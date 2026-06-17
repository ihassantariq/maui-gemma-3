using RagCore.Chunking;

namespace RagCore.Retrieval;

public sealed record ScoredChunk(TextChunk Chunk, float Score);
