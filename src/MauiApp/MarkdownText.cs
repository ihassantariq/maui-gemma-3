namespace RagChatApp;

/// <summary>
/// Renders a minimal markdown subset (**bold**) as a FormattedString for Label display.
/// Splitting on "**" alternates plain/bold segments by index, which also handles an
/// unclosed trailing "**" gracefully while text is still streaming in token by token.
/// Strikethrough markers (~~text~~) are stripped out entirely rather than rendered,
/// since the model occasionally emits them and they're just noise in a chat answer.
/// </summary>
public static class MarkdownText
{
    // Label.TextColor does not cascade to Spans inside FormattedText, so the
    // color must be set per-span explicitly when one is needed (e.g. white
    // text on a colored chat bubble).
    public static FormattedString ToFormattedString(string text, Color? textColor = null)
    {
        text = text.Replace("~~", string.Empty);

        var formatted = new FormattedString();
        var segments = text.Split("**");

        for (var i = 0; i < segments.Length; i++)
        {
            if (segments[i].Length == 0) continue;
            var span = new Span
            {
                Text = segments[i],
                FontAttributes = i % 2 == 1 ? FontAttributes.Bold : FontAttributes.None
            };
            if (textColor is not null)
                span.TextColor = textColor;
            formatted.Spans.Add(span);
        }

        return formatted;
    }
}
