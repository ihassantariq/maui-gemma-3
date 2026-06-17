using RagCore.Retrieval;

namespace RagCore.Generation;

public interface IAnswerGenerator
{
    IAsyncEnumerable<string> GenerateAsync(
        string question,
        IReadOnlyList<ScoredChunk> contextChunks,
        CancellationToken cancellationToken = default);
}
