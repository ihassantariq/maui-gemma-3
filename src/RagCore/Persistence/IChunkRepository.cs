using RagCore.Chunking;

namespace RagCore.Persistence;

public interface IChunkRepository
{
    bool Exists(string dbPath);

    IReadOnlyList<(TextChunk Chunk, float[] Embedding)> Load(string dbPath);

    void Save(string dbPath, IReadOnlyList<(TextChunk Chunk, float[] Embedding)> items);
}
