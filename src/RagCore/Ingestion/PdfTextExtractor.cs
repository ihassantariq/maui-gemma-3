using System.Text.RegularExpressions;
using Syncfusion.Pdf;
using Syncfusion.Pdf.Parsing;

namespace RagCore.Ingestion;

public static class PdfTextExtractor
{
    public static IReadOnlyList<string> ExtractPageTexts(string pdfPath)
    {
        using var stream = File.OpenRead(pdfPath);
        using var document = new PdfLoadedDocument(stream);

        var pages = new List<string>(document.Pages.Count);
        foreach (PdfLoadedPage page in document.Pages)
        {
            var text = page.ExtractText();
            pages.Add(NormalizeWhitespace(text));
        }

        return pages;
    }

    private static string NormalizeWhitespace(string? text) =>
        string.IsNullOrEmpty(text) ? string.Empty : Regex.Replace(text, @"\s+", " ").Trim();
}
