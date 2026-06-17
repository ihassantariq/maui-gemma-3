namespace RagCore.Embedding;

public interface IEmbedder
{
    int EmbeddingDimension { get; }

    float[] Embed(string text);

    float[][] EmbedBatch(IReadOnlyList<string> texts);
}
