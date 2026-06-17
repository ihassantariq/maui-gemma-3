using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;

namespace RagCore.Embedding;

/// <summary>
/// Generates sentence embeddings using the all-MiniLM-L6-v2 ONNX export
/// (mean-pooled, L2-normalized last_hidden_state -> 384-dim vector).
/// </summary>
public sealed class MiniLmEmbedder : IEmbedder, IDisposable
{
    private const int MaxSequenceLength = 256;
    private const string InputIdsName = "input_ids";
    private const string AttentionMaskName = "attention_mask";
    private const string TokenTypeIdsName = "token_type_ids";
    private const string LastHiddenStateName = "last_hidden_state";

    private readonly InferenceSession _session;
    private readonly BertTokenizer _tokenizer;

    public int EmbeddingDimension => 384;

    public MiniLmEmbedder(string modelDir)
    {
        var onnxPath = Path.Combine(modelDir, "model.onnx");
        var vocabPath = Path.Combine(modelDir, "vocab.txt");

        _session = new InferenceSession(onnxPath);
        _tokenizer = BertTokenizer.Create(vocabPath, new BertOptions
        {
            LowerCaseBeforeTokenization = true
        });

        foreach (var required in new[] { InputIdsName, AttentionMaskName, TokenTypeIdsName })
        {
            if (!_session.InputMetadata.ContainsKey(required))
                throw new InvalidOperationException(
                    $"Expected ONNX input '{required}' not found. Available inputs: {string.Join(", ", _session.InputMetadata.Keys)}");
        }

        if (!_session.OutputMetadata.ContainsKey(LastHiddenStateName))
            throw new InvalidOperationException(
                $"Expected ONNX output '{LastHiddenStateName}' not found. Available outputs: {string.Join(", ", _session.OutputMetadata.Keys)}");
    }

    public float[] Embed(string text) => EmbedBatch([text])[0];

    public float[][] EmbedBatch(IReadOnlyList<string> texts)
    {
        var tokenIds = texts
            .Select(text => _tokenizer.EncodeToIds(text, MaxSequenceLength, addSpecialTokens: true, out _, out _))
            .ToList();

        var batchSize = texts.Count;
        var maxLen = tokenIds.Max(ids => ids.Count);

        var inputIds = new DenseTensor<long>(new[] { batchSize, maxLen });
        var attentionMask = new DenseTensor<long>(new[] { batchSize, maxLen });
        var tokenTypeIds = new DenseTensor<long>(new[] { batchSize, maxLen });

        for (var b = 0; b < batchSize; b++)
        {
            var ids = tokenIds[b];
            for (var t = 0; t < maxLen; t++)
            {
                if (t < ids.Count)
                {
                    inputIds[new[] { b, t }] = ids[t];
                    attentionMask[new[] { b, t }] = 1;
                }
                else
                {
                    inputIds[new[] { b, t }] = _tokenizer.PaddingTokenId;
                    attentionMask[new[] { b, t }] = 0;
                }

                tokenTypeIds[new[] { b, t }] = 0;
            }
        }

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(InputIdsName, inputIds),
            NamedOnnxValue.CreateFromTensor(AttentionMaskName, attentionMask),
            NamedOnnxValue.CreateFromTensor(TokenTypeIdsName, tokenTypeIds),
        };

        using var results = _session.Run(inputs);
        var lastHiddenState = results.First(r => r.Name == LastHiddenStateName).AsTensor<float>();

        var embeddings = new float[batchSize][];
        for (var b = 0; b < batchSize; b++)
            embeddings[b] = MeanPoolAndNormalize(lastHiddenState, attentionMask, b, maxLen);

        return embeddings;
    }

    private float[] MeanPoolAndNormalize(Tensor<float> lastHiddenState, DenseTensor<long> attentionMask, int batchIndex, int sequenceLength)
    {
        var pooled = new float[EmbeddingDimension];
        var attendedCount = 0;

        for (var t = 0; t < sequenceLength; t++)
        {
            if (attentionMask[new[] { batchIndex, t }] == 0)
                continue;

            attendedCount++;
            for (var d = 0; d < EmbeddingDimension; d++)
                pooled[d] += lastHiddenState[new[] { batchIndex, t, d }];
        }

        if (attendedCount > 0)
        {
            for (var d = 0; d < EmbeddingDimension; d++)
                pooled[d] /= attendedCount;
        }

        var norm = MathF.Sqrt(pooled.Sum(x => x * x));
        if (norm > 0)
        {
            for (var d = 0; d < EmbeddingDimension; d++)
                pooled[d] /= norm;
        }

        return pooled;
    }

    public void Dispose() => _session.Dispose();
}
