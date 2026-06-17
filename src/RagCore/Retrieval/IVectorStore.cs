using RagCore.Chunking;

namespace RagCore.Retrieval;

public interface IVectorStore
{
    int Count { get; }

    void Add(TextChunk chunk, float[] embedding);

    void AddRange(IEnumerable<(TextChunk Chunk, float[] Embedding)> items);

    IReadOnlyList<ScoredChunk> Search(float[] queryEmbedding, int topK);
}
