namespace RagChatApp;

public partial class App : Application
{
	public App()
	{
		InitializeComponent();
	}

	protected override Window CreateWindow(IActivationState? activationState)
	{
		var session = new AppSession();
		return new Window(new AppShell(session));
	}
}
