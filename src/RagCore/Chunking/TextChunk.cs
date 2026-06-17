namespace RagCore.Chunking;

public sealed record TextChunk(int Index, int PageNumber, string Text, int WordCount);
