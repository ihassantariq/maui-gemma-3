namespace RagCore.Services;

public readonly record struct DownloadProgress(string FileName, long BytesRead, long? TotalBytes);

public static class ModelDownloader
{
    private const string MiniLmBaseUrl = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/";

    private static readonly (string RemotePath, string FileName)[] MiniLmFiles =
    [
        ("onnx/model.onnx", "model.onnx"),
        ("vocab.txt", "vocab.txt"),
    ];

    // Self-built Gemma 3 270M-it int4 CPU bundle (gemma3_text, no vision component).
    // Built via onnxruntime-genai 0.11.4 model-builder from google/gemma-3-270m-it.
    private const string GemmaBaseUrl = "https://huggingface.co/ihassantariq/gemma-3-270m-it-onnx-int4/resolve/main/";

    private static readonly (string RemotePath, string FileName)[] GemmaFiles =
    [
        ("genai_config.json", "genai_config.json"),
        ("chat_template.jinja", "chat_template.jinja"),
        ("tokenizer.json", "tokenizer.json"),
        ("tokenizer_config.json", "tokenizer_config.json"),
        ("model.onnx", "model.onnx"),
        ("model.onnx.data", "model.onnx.data"),
    ];

    public static Task EnsureMiniLmAsync(
        string cacheDir,
        Action<DownloadProgress>? onProgress = null,
        CancellationToken cancellationToken = default) =>
        EnsureFilesAsync(MiniLmBaseUrl, MiniLmFiles, cacheDir, onProgress, cancellationToken);

    public static async Task EnsureGemmaAsync(
        string cacheDir,
        Action<DownloadProgress>? onProgress = null,
        CancellationToken cancellationToken = default)
    {
        await EnsureFilesAsync(GemmaBaseUrl, GemmaFiles, cacheDir, onProgress, cancellationToken);
        // onnxruntime-genai 0.8.3 requires temperature to be a number; the builder
        // emits null when the source model has no explicit temperature set.
        PatchGenAiConfigTemperature(Path.Combine(cacheDir, "genai_config.json"));
    }

    private static void PatchGenAiConfigTemperature(string configPath)
    {
        if (!File.Exists(configPath)) return;
        var content = File.ReadAllText(configPath);
        var patched = content.Replace("\"temperature\": null", "\"temperature\": 1.0");
        if (patched != content)
            File.WriteAllText(configPath, patched);
    }

    private static async Task EnsureFilesAsync(
        string baseUrl,
        (string RemotePath, string FileName)[] files,
        string cacheDir,
        Action<DownloadProgress>? onProgress,
        CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(cacheDir);

        using var client = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };

        foreach (var (remotePath, fileName) in files)
        {
            var destPath = Path.Combine(cacheDir, fileName);
            if (File.Exists(destPath))
            {
                var size = new FileInfo(destPath).Length;
                onProgress?.Invoke(new DownloadProgress(fileName, size, size));
                continue;
            }

            var url = baseUrl + remotePath;
            if (!Uri.IsWellFormedUriString(url, UriKind.Absolute))
                throw new InvalidOperationException(
                    $"'{fileName}' not found at:\n  {destPath}\n\nPre-seed via adb push or set a valid download URL.");

            await DownloadFileAsync(client, url, destPath, fileName, onProgress, cancellationToken);
        }
    }

    private static async Task DownloadFileAsync(
        HttpClient client,
        string url,
        string destPath,
        string displayName,
        Action<DownloadProgress>? onProgress,
        CancellationToken cancellationToken)
    {
        var tempPath = destPath + ".download";

        // Resume partial download if temp file exists from a previous interrupted attempt.
        long resumeFrom = File.Exists(tempPath) ? new FileInfo(tempPath).Length : 0;

        using var request = new HttpRequestMessage(HttpMethod.Get, url);
        if (resumeFrom > 0)
            request.Headers.Range = new System.Net.Http.Headers.RangeHeaderValue(resumeFrom, null);

        using var response = await client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken);

        // 416 means the server thinks the file is already complete.
        if (response.StatusCode == System.Net.HttpStatusCode.RequestedRangeNotSatisfiable)
        {
            File.Move(tempPath, destPath, overwrite: true);
            return;
        }

        response.EnsureSuccessStatusCode();

        var isPartial = response.StatusCode == System.Net.HttpStatusCode.PartialContent;
        var contentLength = response.Content.Headers.ContentLength;
        var totalBytes = isPartial ? resumeFrom + contentLength : contentLength;

        await using (var contentStream = await response.Content.ReadAsStreamAsync(cancellationToken))
        await using (var fileStream = new FileStream(tempPath, isPartial ? FileMode.Append : FileMode.Create, FileAccess.Write))
        {
            var buffer = new byte[81920];
            long readBytes = resumeFrom;
            int bytesRead;

            while (true)
            {
                // Per-read timeout: detect stalled connections (CDN drops).
                using var readCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
                readCts.CancelAfter(TimeSpan.FromSeconds(30));
                try
                {
                    bytesRead = await contentStream.ReadAsync(buffer, readCts.Token);
                }
                catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
                {
                    throw new IOException($"Download stalled — no data received for 30 seconds. Restart the app to resume from {readBytes / 1048576.0:F0} MB.");
                }

                if (bytesRead == 0) break;

                await fileStream.WriteAsync(buffer.AsMemory(0, bytesRead), cancellationToken);
                readBytes += bytesRead;
                onProgress?.Invoke(new DownloadProgress(displayName, readBytes, totalBytes));
            }
        }

        File.Move(tempPath, destPath, overwrite: true);
    }
}
