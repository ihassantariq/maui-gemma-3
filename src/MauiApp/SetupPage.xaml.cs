using System.Security.Cryptography;
using RagCore.Chunking;
using RagCore.Embedding;
using RagCore.Generation;
using RagCore.Ingestion;
using RagCore.Persistence;
using RagCore.Services;

namespace RagChatApp;

public partial class SetupPage : ContentPage
{
    private static readonly string CacheRoot = Path.Combine(FileSystem.AppDataDirectory, ".cache", "maui-gemma-3");
    private static readonly string MiniLmModelDir = Path.Combine(CacheRoot, "models", "all-MiniLM-L6-v2");
    private static readonly string GemmaModelDir = Path.Combine(CacheRoot, "models", "gemma-3-270m-it");
    private static readonly string IndexDir = Path.Combine(CacheRoot, "index");

    private readonly AppSession _session;

    public SetupPage(AppSession session)
    {
        InitializeComponent();
        _session = session;
        _ = Task.Run(InitializeAsync);
    }

    private async Task InitializeAsync()
    {
        try
        {
            var totalFiles = 2 + 6; // MiniLM (2) + Gemma (6 files)
            var completedFiles = 0;

            void UpdateSetup(string status, double progress) =>
                MainThread.BeginInvokeOnMainThread(() =>
                {
                    SetupStatusLabel.Text = status;
                    SetupProgressBar.Progress = progress;
                });

            UpdateSetup("Checking MiniLM model…", 0);
            await ModelDownloader.EnsureMiniLmAsync(MiniLmModelDir, p =>
            {
                if (p.BytesRead >= (p.TotalBytes ?? 1))
                    UpdateSetup($"Downloaded {p.FileName}", (double)++completedFiles / totalFiles);
                else if (p.TotalBytes is > 0)
                    UpdateSetup($"Downloading {p.FileName}: {p.BytesRead * 100 / p.TotalBytes.Value}%",
                        (double)completedFiles / totalFiles);
            });

            UpdateSetup("Checking Gemma model…", (double)completedFiles / totalFiles);
            await ModelDownloader.EnsureGemmaAsync(GemmaModelDir, p =>
            {
                if (p.BytesRead >= (p.TotalBytes ?? 1))
                    UpdateSetup($"Downloaded {p.FileName}", (double)++completedFiles / totalFiles);
                else if (p.TotalBytes is > 0)
                    UpdateSetup($"Downloading {p.FileName}: {p.BytesRead * 100 / p.TotalBytes.Value}%",
                        (double)completedFiles / totalFiles);
            });

            UpdateSetup("Loading embedding model…", 0.9);
            _session.Embedder = await Task.Run(() => new MiniLmEmbedder(MiniLmModelDir));

            UpdateSetup("Loading Gemma model (this may take a few minutes)…", 0.95);
            _session.Generator = await Task.Run(() => new GemmaAnswerGenerator(GemmaModelDir));

            MainThread.BeginInvokeOnMainThread(() =>
            {
                SetupSection.IsVisible = false;
                PickerSection.IsVisible = true;
            });
        }
        catch (Exception ex)
        {
            MainThread.BeginInvokeOnMainThread(() =>
                SetupStatusLabel.Text = $"Setup failed: {ex.Message}");
        }
    }

    private async void OnPickPdfClicked(object? sender, EventArgs e)
    {
        try
        {
            var result = await FilePicker.PickAsync(new PickOptions
            {
                PickerTitle = "Pick a PDF",
                FileTypes = new FilePickerFileType(new Dictionary<DevicePlatform, IEnumerable<string>>
                {
                    // Use */* on Android — strict application/pdf hides files with wrong MIME type
                    // (common for files downloaded via browser or transferred from other devices).
                    // Extension is validated below instead.
                    { DevicePlatform.Android, ["*/*"] },
                    { DevicePlatform.iOS, ["com.adobe.pdf"] },
                    { DevicePlatform.MacCatalyst, ["com.adobe.pdf"] },
                })
            });

            if (result is null) return;

            if (!result.FileName.EndsWith(".pdf", StringComparison.OrdinalIgnoreCase))
            {
                await DisplayAlertAsync("Invalid file", "Please select a PDF file.", "OK");
                return;
            }

            PickPdfButton.IsEnabled = false;
            IndexingCard.IsVisible = true;

            // Android content URIs aren't directly readable via File.OpenRead — copy to cache first.
            string pdfPath;
            if (string.IsNullOrEmpty(result.FullPath))
            {
                var tempPath = Path.Combine(FileSystem.CacheDirectory, result.FileName);
                using var src = await result.OpenReadAsync();
                using var dst = File.Create(tempPath);
                await src.CopyToAsync(dst);
                pdfPath = tempPath;
            }
            else
            {
                pdfPath = result.FullPath;
            }

            await IndexPdfAsync(pdfPath);

            await Shell.Current.GoToAsync("//main");
        }
        catch (Exception ex)
        {
            IndexingStatusLabel.Text = $"Error: {ex.Message}";
            PickPdfButton.IsEnabled = true;
        }
    }

    private async Task IndexPdfAsync(string pdfPath)
    {
        Directory.CreateDirectory(IndexDir);

        var pdfHash = ComputeFileHash(pdfPath);
        var dbPath = Path.Combine(IndexDir, $"{pdfHash}.db");

        IChunkRepository repository = new SqliteChunkRepository();
        _session.ResetForNewDocument();
        var store = _session.Store!;

        void UpdateIndexing(string status, double progress) =>
            MainThread.BeginInvokeOnMainThread(() =>
            {
                IndexingStatusLabel.Text = status;
                IndexingProgressBar.Progress = progress;
            });

        if (repository.Exists(dbPath))
        {
            UpdateIndexing("Loading cached index…", 0.5);
            var items = await Task.Run(() => repository.Load(dbPath));

            if (items.Count == 0)
                throw new InvalidOperationException(
                    "This PDF doesn't contain any extractable text — it may be a scanned or image-only document.");

            store.AddRange(items);
            _session.DocumentChunks = items.Select(i => i.Chunk).ToList();
            UpdateIndexing($"Loaded {store.Count} chunks from cache.", 1.0);
        }
        else
        {
            UpdateIndexing("Extracting text from PDF…", 0.1);
            var pageTexts = await Task.Run(() => PdfTextExtractor.ExtractPageTexts(pdfPath));

            UpdateIndexing($"Chunking {pageTexts.Count} pages…", 0.2);
            var chunks = await Task.Run(() => TextChunker.ChunkPages(pageTexts));
            _session.DocumentChunks = chunks;

            if (chunks.Count == 0)
                throw new InvalidOperationException(
                    "This PDF doesn't contain any extractable text — it may be a scanned or image-only document.");

            var items = new List<(TextChunk Chunk, float[] Embedding)>(chunks.Count);
            for (var i = 0; i < chunks.Count; i++)
            {
                var chunk = chunks[i];
                var embedding = await Task.Run(() => _session.Embedder!.Embed(chunk.Text));
                items.Add((chunk, embedding));
                UpdateIndexing($"Embedding chunk {i + 1}/{chunks.Count}…", 0.2 + 0.7 * (i + 1) / chunks.Count);
            }

            store.AddRange(items);
            await Task.Run(() => repository.Save(dbPath, items));
            UpdateIndexing($"Indexed {chunks.Count} chunks.", 1.0);
        }
    }

    private static string ComputeFileHash(string path)
    {
        using var stream = File.OpenRead(path);
        var hash = SHA256.HashData(stream);
        return Convert.ToHexString(hash).ToLowerInvariant();
    }
}
