using RagCore.Chunking;

namespace RagCore.Retrieval;

public sealed class InMemoryVectorStore
{
    private readonly List<(TextChunk Chunk, float[] Embedding)> _items = [];

    public int Count => _items.Count;

    public void Add(TextChunk chunk, float[] embedding) => _items.Add((chunk, embedding));

    public void AddRange(IEnumerable<(TextChunk Chunk, float[] Embedding)> items) => _items.AddRange(items);

    public IReadOnlyList<ScoredChunk> Search(float[] queryEmbedding, int topK) =>
        _items
            .Select(item => new ScoredChunk(item.Chunk, VectorMath.CosineSimilarity(queryEmbedding, item.Embedding)))
            .OrderByDescending(s => s.Score)
            .Take(topK)
            .ToList();
}
