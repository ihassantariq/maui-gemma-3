using System.Security.Cryptography;
using RagCore.Chunking;
using RagCore.Embedding;
using RagCore.Generation;
using RagCore.Ingestion;
using RagCore.Persistence;
using RagCore.Retrieval;
using RagCore.Services;

if (args.Length == 0)
{
    Console.WriteLine("Usage: dotnet run --project src/RagConsole -- <path-to-pdf>");
    return 1;
}

var pdfPath = args[0];
if (!File.Exists(pdfPath))
{
    Console.WriteLine($"File not found: {pdfPath}");
    return 1;
}

var syncfusionKey = Environment.GetEnvironmentVariable("SYNCFUSION_LICENSE_KEY");
if (!string.IsNullOrWhiteSpace(syncfusionKey))
{
    Syncfusion.Licensing.SyncfusionLicenseProvider.RegisterLicense(syncfusionKey);
}
else
{
    Console.WriteLine("Warning: SYNCFUSION_LICENSE_KEY is not set. PDF extraction may show a watermark or fail.");
}

var userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
var miniLmModelDir = Path.Combine(userProfile, ".cache", "maui-gemma-3", "models", "all-MiniLM-L6-v2");
var gemmaModelDir = Path.Combine(userProfile, ".cache", "maui-gemma-3", "models", "gemma-3-270m-it");
var indexDir = Path.Combine(userProfile, ".cache", "maui-gemma-3", "index");

Console.WriteLine("Checking MiniLM model cache...");
await ModelDownloader.EnsureMiniLmAsync(miniLmModelDir, p =>
{
    if (p.TotalBytes is > 0 && p.BytesRead < p.TotalBytes)
        Console.Write($"\rDownloading {p.FileName}: {p.BytesRead * 100 / p.TotalBytes.Value}% ({p.BytesRead / 1048576.0:F1} MB / {p.TotalBytes.Value / 1048576.0:F1} MB)");
});
Console.WriteLine();

Console.WriteLine("Checking Gemma 3 model cache (large download on first run)...");
await ModelDownloader.EnsureGemmaAsync(gemmaModelDir, p =>
{
    if (p.TotalBytes is > 0 && p.BytesRead < p.TotalBytes)
        Console.Write($"\rDownloading {p.FileName}: {p.BytesRead * 100 / p.TotalBytes.Value}% ({p.BytesRead / 1048576.0:F1} MB / {p.TotalBytes.Value / 1048576.0:F1} MB)");
});
Console.WriteLine();

// Not disposed: onnxruntime-genai 0.8.3's native Dispose() crashes with
// "libc++abi: mutex lock failed" on macOS arm64 during process exit. The
// process reclaims these resources on exit regardless.
var embedder = new MiniLmEmbedder(miniLmModelDir);

var pdfHash = ComputeFileHash(pdfPath);
var dbPath = Path.Combine(indexDir, $"{pdfHash}.db");

IChunkRepository repository = new SqliteChunkRepository();
var store = new InMemoryVectorStore();

if (repository.Exists(dbPath))
{
    Console.WriteLine("Found cached index for this PDF, loading...");
    var items = repository.Load(dbPath);
    store.AddRange(items);
    Console.WriteLine($"Loaded {store.Count} chunks from cache.");
}
else
{
    Console.WriteLine("Extracting text from PDF...");
    var pageTexts = PdfTextExtractor.ExtractPageTexts(pdfPath);
    Console.WriteLine($"Extracted {pageTexts.Count} pages.");

    var chunks = TextChunker.ChunkPages(pageTexts);
    Console.WriteLine($"Created {chunks.Count} chunks. Embedding...");

    var items = new List<(TextChunk Chunk, float[] Embedding)>(chunks.Count);
    for (var i = 0; i < chunks.Count; i++)
    {
        var embedding = embedder.Embed(chunks[i].Text);
        items.Add((chunks[i], embedding));
        Console.Write($"\rEmbedded {i + 1}/{chunks.Count} chunks");
    }
    Console.WriteLine();

    store.AddRange(items);
    repository.Save(dbPath, items);
    Console.WriteLine($"Saved index to {dbPath}");
}

Console.WriteLine("Loading Gemma 3 model (this can take a while)...");
var generator = new GemmaAnswerGenerator(gemmaModelDir);

Console.WriteLine();
Console.WriteLine("Ask a question about the document (type 'exit' to quit):");

while (true)
{
    Console.Write("\n> ");
    var question = Console.ReadLine();
    if (string.IsNullOrWhiteSpace(question))
        continue;

    if (question.Trim().Equals("exit", StringComparison.OrdinalIgnoreCase))
        break;

    var queryEmbedding = embedder.Embed(question);
    var results = store.Search(queryEmbedding, topK: 3);

    Console.WriteLine("\nRetrieved chunks:");
    foreach (var result in results)
    {
        var snippet = string.Join(' ', result.Chunk.Text.Split(' ').Take(20));
        Console.WriteLine($"  - page {result.Chunk.PageNumber}, score {result.Score:F3}: \"{snippet}...\"");
    }

    Console.WriteLine("\nAnswer:");
    await foreach (var token in generator.GenerateAsync(question, results))
    {
        Console.Write(token);
    }
    Console.WriteLine();
}

return 0;

static string ComputeFileHash(string path)
{
    using var stream = File.OpenRead(path);
    var hash = SHA256.HashData(stream);
    return Convert.ToHexString(hash).ToLowerInvariant();
}
