using RagCore.Chunking;
using RagCore.Retrieval;

namespace RagChatApp;

public partial class SummaryPage : ContentPage
{
    // 3000 words is empirically proven safe on-device — every section pass in the
    // earlier map-reduce version used this exact budget without crashing. 8000 words
    // previously caused a hard, unlogged crash (likely an on-device OOM in the native
    // onnxruntime-genai KV-cache allocation). This trades full whole-document coverage
    // for a single, simple, predictable generation call.
    private const int MaxSummaryWords = 3000;

    private readonly AppSession _session;

    public SummaryPage(AppSession session)
    {
        InitializeComponent();
        _session = session;
    }

    protected override async void OnAppearing()
    {
        base.OnAppearing();

        SummaryLabel.FormattedText = MarkdownText.ToFormattedString(_session.Summary.Text);
        LoadingSection.IsVisible = _session.Summary.IsGenerating;

        if (_session.Summary.HasStarted) return;
        if (_session.Generator is null || _session.DocumentChunks is null) return;

        _session.Summary.HasStarted = true;
        _session.Summary.IsGenerating = true;
        LoadingStatusLabel.Text = "Generating summary…";
        LoadingSection.IsVisible = true;

        try
        {
            var contextChunks = BuildSampledContext(_session.DocumentChunks, MaxSummaryWords);
            var fullText = string.Empty;

            await foreach (var token in _session.Generator.SummarizeAsync(contextChunks))
            {
                fullText += token;
                var snapshot = fullText;
                MainThread.BeginInvokeOnMainThread(() =>
                {
                    _session.Summary.Text = snapshot;
                    SummaryLabel.FormattedText = MarkdownText.ToFormattedString(snapshot);
                });
            }
        }
        catch (Exception ex)
        {
            var errorText = $"[Error generating summary: {ex.Message}]";
            _session.Summary.Text = errorText;
            SummaryLabel.FormattedText = MarkdownText.ToFormattedString(errorText);
        }
        finally
        {
            _session.Summary.IsGenerating = false;
            LoadingSection.IsVisible = false;
        }
    }

    // Sample whole chunks evenly across the entire document instead of just taking the
    // first N words. Truncating at the front means anything past ~8-10 pages is never
    // seen at all — a document's key results/conclusions are just as likely to be in
    // the back half as the front. Sampling at chunk granularity (not word-level slicing)
    // keeps every included excerpt complete rather than cut off mid-sentence.
    private static IReadOnlyList<ScoredChunk> BuildSampledContext(IReadOnlyList<TextChunk> chunks, int maxWords)
    {
        if (chunks.Count == 0) return [];

        var avgWordsPerChunk = (double)chunks.Sum(c => c.WordCount) / chunks.Count;
        var chunksToKeep = Math.Max(1, Math.Min(chunks.Count, (int)(maxWords / avgWordsPerChunk)));

        if (chunksToKeep >= chunks.Count)
            return chunks.Select(c => new ScoredChunk(c, 1f)).ToList();

        var result = new List<ScoredChunk>(chunksToKeep);
        for (var i = 0; i < chunksToKeep; i++)
        {
            var index = (int)((long)i * chunks.Count / chunksToKeep);
            result.Add(new ScoredChunk(chunks[index], 1f)); // dummy score; GemmaAnswerGenerator.BuildSummaryPrompt ignores it
        }
        return result;
    }
}
