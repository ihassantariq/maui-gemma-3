using System.Runtime.InteropServices;
using Microsoft.Data.Sqlite;
using RagCore.Chunking;

namespace RagCore.Persistence;

public sealed class SqliteChunkRepository : IChunkRepository
{
    public bool Exists(string dbPath) => File.Exists(dbPath);

    public IReadOnlyList<(TextChunk Chunk, float[] Embedding)> Load(string dbPath)
    {
        using var connection = OpenConnection(dbPath);

        using var command = connection.CreateCommand();
        command.CommandText = "SELECT idx, page_number, text, word_count, embedding FROM chunks ORDER BY idx;";

        using var reader = command.ExecuteReader();

        var items = new List<(TextChunk Chunk, float[] Embedding)>();
        while (reader.Read())
        {
            var chunk = new TextChunk(
                Index: reader.GetInt32(0),
                PageNumber: reader.GetInt32(1),
                Text: reader.GetString(2),
                WordCount: reader.GetInt32(3));

            var bytes = (byte[])reader[4];
            var embedding = MemoryMarshal.Cast<byte, float>(bytes).ToArray();

            items.Add((chunk, embedding));
        }

        return items;
    }

    public void Save(string dbPath, IReadOnlyList<(TextChunk Chunk, float[] Embedding)> items)
    {
        var directory = Path.GetDirectoryName(dbPath);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);

        if (File.Exists(dbPath))
            File.Delete(dbPath);

        using var connection = OpenConnection(dbPath);

        using var createCommand = connection.CreateCommand();
        createCommand.CommandText = """
            CREATE TABLE chunks (
                idx INTEGER PRIMARY KEY,
                page_number INTEGER NOT NULL,
                text TEXT NOT NULL,
                word_count INTEGER NOT NULL,
                embedding BLOB NOT NULL
            );
            """;
        createCommand.ExecuteNonQuery();

        using var transaction = connection.BeginTransaction();
        using var insertCommand = connection.CreateCommand();
        insertCommand.Transaction = transaction;
        insertCommand.CommandText = """
            INSERT INTO chunks (idx, page_number, text, word_count, embedding)
            VALUES ($idx, $pageNumber, $text, $wordCount, $embedding);
            """;

        var idxParam = insertCommand.CreateParameter();
        idxParam.ParameterName = "$idx";
        insertCommand.Parameters.Add(idxParam);

        var pageNumberParam = insertCommand.CreateParameter();
        pageNumberParam.ParameterName = "$pageNumber";
        insertCommand.Parameters.Add(pageNumberParam);

        var textParam = insertCommand.CreateParameter();
        textParam.ParameterName = "$text";
        insertCommand.Parameters.Add(textParam);

        var wordCountParam = insertCommand.CreateParameter();
        wordCountParam.ParameterName = "$wordCount";
        insertCommand.Parameters.Add(wordCountParam);

        var embeddingParam = insertCommand.CreateParameter();
        embeddingParam.ParameterName = "$embedding";
        insertCommand.Parameters.Add(embeddingParam);

        foreach (var (chunk, embedding) in items)
        {
            idxParam.Value = chunk.Index;
            pageNumberParam.Value = chunk.PageNumber;
            textParam.Value = chunk.Text;
            wordCountParam.Value = chunk.WordCount;
            embeddingParam.Value = MemoryMarshal.AsBytes<float>(embedding).ToArray();

            insertCommand.ExecuteNonQuery();
        }

        transaction.Commit();
    }

    private static SqliteConnection OpenConnection(string dbPath)
    {
        var connection = new SqliteConnection($"Data Source={dbPath}");
        connection.Open();
        return connection;
    }
}
