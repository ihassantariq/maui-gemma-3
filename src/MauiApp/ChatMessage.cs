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
        }
    }

    public Color RoleColor => Role == "user"
        ? Color.FromArgb("#512BD4")
        : Color.FromArgb("#2D6A4F");

    public LayoutOptions BubbleAlign => Role == "user"
        ? LayoutOptions.End
        : LayoutOptions.Start;

    public event PropertyChangedEventHandler? PropertyChanged;
}
