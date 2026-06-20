namespace RagChatApp;

public partial class AppShell : Shell
{
    public AppShell(AppSession session)
    {
        InitializeComponent();
        SetupShellContent.ContentTemplate = new DataTemplate(() => new SetupPage(session));
        SummaryShellContent.ContentTemplate = new DataTemplate(() => new SummaryPage(session));
        ChatShellContent.ContentTemplate = new DataTemplate(() => new ChatPage(session));
    }
}
