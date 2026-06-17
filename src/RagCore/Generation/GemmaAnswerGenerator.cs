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
public sealed class GemmaAnswerGenerator : IAnswerGenerator
{
    private const int MaxNewTokens = 512;

    private readonly Model _model;
    private readonly Tokenizer _tokenizer;

    public GemmaAnswerGenerator(string modelDir)
    {
        _model = new Model(modelDir);
        _tokenizer = new Tokenizer(_model);
    }

    public async IAsyncEnumerable<string> GenerateAsync(
        string question,
        IReadOnlyList<ScoredChunk> contextChunks,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var prompt = BuildPrompt(question, contextChunks);
        var channel = Channel.CreateUnbounded<string>(new UnboundedChannelOptions
        {
            SingleWriter = true,
            SingleReader = true
        });

        var generationTask = Task.Run(() => Generate(prompt, channel.Writer, cancellationToken), cancellationToken);

        await foreach (var token in channel.Reader.ReadAllAsync(cancellationToken))
            yield return token;

        await generationTask;
    }

    private void Generate(string prompt, ChannelWriter<string> writer, CancellationToken cancellationToken)
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

                    writer.TryWrite(decoded);
                }
            }
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

    private static string BuildPrompt(string question, IReadOnlyList<ScoredChunk> contextChunks)
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
}
