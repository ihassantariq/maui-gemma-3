using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Channels;
using Microsoft.ML.OnnxRuntimeGenAI;
using RagCore.Retrieval;

namespace RagCore.Generation;

/// <summary>
/// Generates answers with an on-device Gemma 3 model via OnnxRuntimeGenAI.
/// Expects <paramref name="modelDir"/> to contain genai_config.json plus the
/// decoder/embedding ONNX files and tokenizer files for a text-only build.
/// </summary>
public sealed class GemmaAnswerGenerator
{
    private const int MaxNewTokens = 1024;

    private readonly Model _model;
    private readonly Tokenizer _tokenizer;

    public GemmaAnswerGenerator(string modelDir)
    {
        _model = new Model(modelDir);
        _tokenizer = new Tokenizer(_model);
    }

    public IAsyncEnumerable<string> GenerateAsync(
        string question,
        IReadOnlyList<ScoredChunk> contextChunks,
        CancellationToken cancellationToken = default) =>
        StreamTokens(BuildQuestionPrompt(question, contextChunks), EmptyAnswerFallback, cancellationToken);

    public IAsyncEnumerable<string> SummarizeAsync(
        IReadOnlyList<ScoredChunk> contextChunks,
        CancellationToken cancellationToken = default) =>
        StreamTokens(BuildSummaryPrompt(contextChunks), EmptySummaryFallback, cancellationToken);

    private async IAsyncEnumerable<string> StreamTokens(
        string prompt,
        string emptyFallback,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var channel = Channel.CreateUnbounded<string>(new UnboundedChannelOptions
        {
            SingleWriter = true,
            SingleReader = true
        });

        var generationTask = Task.Run(() => Generate(prompt, emptyFallback, channel.Writer, cancellationToken), cancellationToken);

        await foreach (var token in channel.Reader.ReadAllAsync(cancellationToken))
            yield return token;

        await generationTask;
    }

    // The Q&A prompt's "Question: ...<end_of_turn>" framing leads this small model to often
    // imitate the same Q&A structure and start its reply with "Answer:" even though it
    // was never asked to. Buffer just enough of the start of the stream to detect and
    // strip that prefix before any text reaches the UI. Applied universally (including
    // to summaries) as a generic safety net — harmless even when it never matches.
    private const string AnswerPrefix = "Answer:";
    private const string EmptyAnswerFallback = "I couldn't find a clear answer to that in the document. Could you try rephrasing the question?";
    private const string EmptySummaryFallback = "Couldn't generate a summary for this document. Try again.";

    private void Generate(string prompt, string emptyFallback, ChannelWriter<string> writer, CancellationToken cancellationToken)
    {
        Exception? error = null;
        try
        {
            using var sequences = _tokenizer.Encode(prompt);
            using var generatorParams = new GeneratorParams(_model);
            generatorParams.SetSearchOption("max_length", sequences[0].Length + MaxNewTokens);
            generatorParams.SetSearchOption("temperature", 0.7);
            generatorParams.SetSearchOption("top_p", 0.95);

            using var generator = new Generator(_model, generatorParams);
            using var stream = _tokenizer.CreateStream();

            generator.AppendTokenSequences(sequences);

            var leadingBuffer = string.Empty;
            var checkedPrefix = false;
            var anyTextWritten = false;

            void Write(string text)
            {
                if (string.IsNullOrEmpty(text)) return;
                anyTextWritten = true;
                writer.TryWrite(text);
            }

            while (!generator.IsDone())
            {
                cancellationToken.ThrowIfCancellationRequested();

                generator.GenerateNextToken();

                var sequence = generator.GetSequence(0);
                var newToken = sequence[^1];
                var decoded = stream.Decode(newToken);
                if (!string.IsNullOrEmpty(decoded))
                {
                    // TokenizerStream.Decode only converts a single leading SentencePiece
                    // "meta-space" (U+2581) to a real space; tokens encoding multiple
                    // leading spaces (e.g. list indentation) leave extra "▁" characters.
                    if (decoded.Contains('▁'))
                        decoded = decoded.Replace('▁', ' ');

                    if (!checkedPrefix)
                    {
                        leadingBuffer += decoded;
                        var trimmed = leadingBuffer.TrimStart();
                        if (trimmed.Length < AnswerPrefix.Length)
                            continue;

                        checkedPrefix = true;
                        if (trimmed.StartsWith(AnswerPrefix, StringComparison.OrdinalIgnoreCase))
                            Write(trimmed[AnswerPrefix.Length..].TrimStart());
                        else
                            Write(leadingBuffer);
                        continue;
                    }

                    Write(decoded);
                }
            }

            // Generation finished before enough characters arrived to check the prefix.
            if (!checkedPrefix && leadingBuffer.Length > 0)
                Write(leadingBuffer);

            // The model sometimes emits only "Answer:" (or nothing at all) before hitting
            // its end-of-turn token — a genuinely empty completion, not just the stripped
            // prefix. Surface a fallback instead of leaving the chat bubble blank.
            if (!anyTextWritten)
                Write(emptyFallback);
        }
        catch (Exception ex)
        {
            error = ex;
        }
        finally
        {
            writer.Complete(error);
        }
    }

    private static string BuildQuestionPrompt(string question, IReadOnlyList<ScoredChunk> contextChunks)
    {
        var sb = new StringBuilder();
        sb.Append("<start_of_turn>user\n");
        sb.Append("Answer the question using only the information in the context below. ");
        sb.Append("If the answer is not in the context, say you don't know.\n\n");
        sb.Append("Context:\n");

        foreach (var scored in contextChunks)
            sb.Append($"[page {scored.Chunk.PageNumber}] {scored.Chunk.Text}\n\n");

        sb.Append($"Question: {question}<end_of_turn>\n");
        sb.Append("<start_of_turn>model\n");
        return sb.ToString();
    }

    private static string BuildSummaryPrompt(IReadOnlyList<ScoredChunk> contextChunks)
    {
        // Document content first, instruction last (right before <end_of_turn>) — the
        // model weights text closest to the generation point much more heavily, and
        // putting the task instruction up front (with raw document text as the last
        // thing it sees) made it ask for "the text to summarize" instead of using it.
        var sb = new StringBuilder();
        sb.Append("<start_of_turn>user\n");
        sb.Append("Document:\n");

        foreach (var scored in contextChunks)
            sb.Append($"[page {scored.Chunk.PageNumber}] {scored.Chunk.Text}\n\n");

        sb.Append("Summarize the document above, covering its main topics and key points ");
        sb.Append("in a few short paragraphs.<end_of_turn>\n");
        sb.Append("<start_of_turn>model\n");
        return sb.ToString();
    }
}
