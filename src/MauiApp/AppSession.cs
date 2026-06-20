using System.Collections.ObjectModel;
using System.ComponentModel;
using RagCore.Chunking;
using RagCore.Embedding;
using RagCore.Generation;
using RagCore.Retrieval;

namespace RagChatApp;

/// <summary>
/// Holds app/document state shared across SetupPage, SummaryPage, and ChatPage.
/// One instance is created at app startup and passed to each page's constructor —
/// this app has no DI container, so this is the deliberately simple substitute.
/// </summary>
public sealed class AppSession
{
    public MiniLmEmbedder? Embedder { get; set; }
    public GemmaAnswerGenerator? Generator { get; set; }
    public InMemoryVectorStore? Store { get; set; }

    /// <summary>Full, page-ordered chunks for the currently indexed PDF (not similarity-filtered).</summary>
    public IReadOnlyList<TextChunk>? DocumentChunks { get; set; }

    public ObservableCollection<ChatMessage> Messages { get; } = [];

    public SummaryState Summary { get; } = new();

    public void ResetForNewDocument()
    {
        Store = new InMemoryVectorStore();
        DocumentChunks = null;
        Messages.Clear();
        Summary.Reset();
    }
}

/// <summary>Bindable holder for summary generation state, mirroring ChatMessage's PropertyChanged pattern.</summary>
public sealed class SummaryState : INotifyPropertyChanged
{
    private string _text = string.Empty;
    private bool _isGenerating;

    public string Text
    {
        get => _text;
        set { if (_text == value) return; _text = value; OnChanged(nameof(Text)); }
    }

    public bool IsGenerating
    {
        get => _isGenerating;
        set { if (_isGenerating == value) return; _isGenerating = value; OnChanged(nameof(IsGenerating)); }
    }

    /// <summary>True once generation has started (or finished) for the current document.</summary>
    public bool HasStarted { get; set; }

    public void Reset()
    {
        Text = string.Empty;
        IsGenerating = false;
        HasStarted = false;
    }

    public event PropertyChangedEventHandler? PropertyChanged;
    private void OnChanged(string name) => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
}
