using Microsoft.Extensions.Logging;
using MauiHostingApp = Microsoft.Maui.Hosting.MauiApp;

namespace RagChatApp;

public static class MauiProgram
{
	public static MauiHostingApp CreateMauiApp()
	{
		var builder = MauiHostingApp.CreateBuilder();
		builder
			.UseMauiApp<App>()
			.ConfigureFonts(fonts =>
			{
				fonts.AddFont("OpenSans-Regular.ttf", "OpenSansRegular");
				fonts.AddFont("OpenSans-Semibold.ttf", "OpenSansSemibold");
			});

#if DEBUG
		builder.Logging.AddDebug();
#endif

		return builder.Build();
	}
}
