namespace RagCore.Retrieval;

public static class VectorMath
{
    public static float CosineSimilarity(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        var dot = 0f;
        var normA = 0f;
        var normB = 0f;
        for (var i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        if (normA == 0f || normB == 0f)
            return 0f;

        return dot / (MathF.Sqrt(normA) * MathF.Sqrt(normB));
    }
}
