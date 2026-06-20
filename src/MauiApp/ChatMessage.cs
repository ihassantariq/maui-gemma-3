using System.ComponentModel;

namespace RagChatApp;

public sealed class ChatMessage : INotifyPropertyChanged
{
    private string _text;

    public ChatMessage(string role, string text)
    {
        Role = role;
        _text = text;
    }

    public string Role { get; }

    public string Text
    {
        get => _text;
        set
        {
            if (_text == value) return;
            _text = value;
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(Text)));
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(FormattedText)));
        }
    }

    public FormattedString FormattedText => MarkdownText.ToFormattedString(_text, BubbleTextColor);

    public bool IsUser => Role == "user";

    public Color BubbleColor => IsUser
        ? Color.FromArgb("#512BD4")
        : Color.FromArgb("#E1E1E1");

    public Color BubbleTextColor => IsUser
        ? Colors.White
        : Color.FromArgb("#1f1f1f");

    public LayoutOptions BubbleAlign => IsUser
        ? LayoutOptions.End
        : LayoutOptions.Start;

    /// <summary>Speech-bubble "tail" effect: the corner nearest the sender's edge is squared off.</summary>
    public CornerRadius BubbleCorners => IsUser
        ? new CornerRadius(16, 16, 4, 16)
        : new CornerRadius(16, 16, 16, 4);

    public event PropertyChangedEventHandler? PropertyChanged;
}
