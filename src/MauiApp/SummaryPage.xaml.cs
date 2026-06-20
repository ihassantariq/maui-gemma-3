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
            var contextChunks = BuildCappedContext(_session.DocumentChunks, MaxSummaryWords);
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

    private static IReadOnlyList<ScoredChunk> BuildCappedContext(IReadOnlyList<TextChunk> chunks, int maxWords)
    {
        var result = new List<ScoredChunk>();
        var budget = maxWords;
        foreach (var chunk in chunks)
        {
            if (budget <= 0) break;
            result.Add(new ScoredChunk(chunk, 1f)); // dummy score; GemmaAnswerGenerator.BuildSummaryPrompt ignores it
            budget -= chunk.WordCount;
        }
        return result;
    }
}
