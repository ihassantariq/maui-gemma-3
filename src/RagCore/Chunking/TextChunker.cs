namespace RagCore.Chunking;

public static class TextChunker
{
    public static IReadOnlyList<TextChunk> ChunkPages(
        IReadOnlyList<string> pageTexts,
        int targetWords = 300,
        int overlapWords = 50,
        int minWords = 20)
    {
        var chunks = new List<TextChunk>();
        var step = targetWords - overlapWords;

        for (int pageIndex = 0; pageIndex < pageTexts.Count; pageIndex++)
        {
            var words = pageTexts[pageIndex].Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (words.Length == 0)
                continue;

            if (words.Length <= targetWords)
            {
                chunks.Add(new TextChunk(chunks.Count, pageIndex + 1, string.Join(' ', words), words.Length));
                continue;
            }

            var start = 0;
            while (start < words.Length)
            {
                var count = Math.Min(targetWords, words.Length - start);

                // Last window on the page is too small on its own - merge it into the previous chunk.
                if (start > 0 && count < minWords)
                {
                    var previous = chunks[^1];
                    var tailWords = words.Skip(start).Take(count);
                    chunks[^1] = previous with
                    {
                        Text = previous.Text + " " + string.Join(' ', tailWords),
                        WordCount = previous.WordCount + count
                    };
                    break;
                }

                var chunkWords = words.Skip(start).Take(count).ToArray();
                chunks.Add(new TextChunk(chunks.Count, pageIndex + 1, string.Join(' ', chunkWords), chunkWords.Length));

                if (start + count >= words.Length)
                    break;

                start += step;
            }
        }

        return chunks;
    }
}
