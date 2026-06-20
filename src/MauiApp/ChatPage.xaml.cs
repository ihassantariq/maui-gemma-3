using System.Collections.ObjectModel;
using System.Windows.Input;

namespace RagChatApp;

public partial class ChatPage : ContentPage
{
    private readonly AppSession _session;
    private bool _isGenerating;

    public ObservableCollection<ChatMessage> Messages => _session.Messages;

    public ICommand AskCommand { get; }

    public ChatPage(AppSession session)
    {
        InitializeComponent();
        _session = session;
        AskCommand = new Command(() => _ = AskAsync());
        BindingContext = this;
    }

    private void OnAskClicked(object? sender, EventArgs e) => _ = AskAsync();

    private async Task AskAsync()
    {
        if (_isGenerating || _session.Embedder is null || _session.Generator is null || _session.Store is null) return;

        var question = QuestionEntry.Text?.Trim();
        if (string.IsNullOrEmpty(question)) return;

        QuestionEntry.Text = string.Empty;
        _isGenerating = true;
        AskButton.IsEnabled = false;

        Messages.Add(new ChatMessage("user", question));

        var assistantMessage = new ChatMessage("assistant", "…");
        Messages.Add(assistantMessage);
        MessagesView.ScrollTo(Messages.Count - 1, position: ScrollToPosition.End, animate: false);

        try
        {
            var queryEmbedding = await Task.Run(() => _session.Embedder.Embed(question));
            var results = _session.Store.Search(queryEmbedding, topK: 3);

            var fullText = string.Empty;
            await foreach (var token in _session.Generator.GenerateAsync(question, results))
            {
                fullText += token;
                var snapshot = fullText;
                MainThread.BeginInvokeOnMainThread(() =>
                {
                    assistantMessage.Text = snapshot;
                    MessagesView.ScrollTo(Messages.Count - 1, position: ScrollToPosition.End, animate: false);
                });
            }
        }
        catch (Exception ex)
        {
            assistantMessage.Text = $"[Error: {ex.Message}]";
        }
        finally
        {
            _isGenerating = false;
            AskButton.IsEnabled = true;
        }
    }
}
