from dataclasses import dataclass

from psycopg2.extras import Json

from storage.postgres import get_connection


@dataclass(frozen=True)
class StoredChunk:
    id: int
    document_path: str
    chunk_index: int
    content: str
    created_at: str
    embedding: list[float] | None = None
    embedding_model: str | None = None
    token_count: int | None = None


class ChunkRepository:
    def clear_all(self) -> int:
        """Удаляет все чанки и возвращает количество удалённых записей."""
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM chunks")
                deleted_count = cursor.rowcount
        return deleted_count

    def replace_document_chunks(
        self,
        document_path: str,
        chunks: list[str],
        embeddings: list[list[float]],
        embedding_model: str,
        token_counts: list[int],
    ) -> int:
        if not (len(chunks) == len(embeddings) == len(token_counts)):
            raise ValueError(
                "Количество чанков, векторов и значений token_count не совпадает."
            )

        for vector in embeddings:
            if not vector:
                raise ValueError("Обнаружен пустой embedding-вектор.")

        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM chunks WHERE document_path = %s",
                    (document_path,),
                )

                if chunks:
                    rows = []
                    for index, (content, vector) in enumerate(
                        zip(chunks, embeddings)
                    ):
                        vector_literal = "[" + ",".join(
                            format(float(value), ".9g") for value in vector
                        ) + "]"

                        rows.append(
                            (
                                document_path,
                                index,
                                content,
                                Json(vector),
                                vector_literal,
                                embedding_model,
                                token_counts[index],
                            )
                        )

                    cursor.executemany(
                        """
                        INSERT INTO chunks (
                            document_path,
                            chunk_index,
                            content,
                            embedding,
                            embedding_vector,
                            embedding_model,
                            token_count
                        )
                        VALUES (%s, %s, %s, %s, %s::vector, %s, %s)
                        """,
                        rows,
                    )

        return len(chunks)

    def get_all(
        self,
        search: str = "",
        limit: int = 1000,
        offset: int = 0,
    ) -> list[StoredChunk]:
        if limit < 1 or offset < 0:
            raise ValueError("Некорректные limit или offset.")

        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        id,
                        document_path,
                        chunk_index,
                        content,
                        created_at,
                        embedding,
                        embedding_model,
                        token_count
                    FROM chunks
                    WHERE document_path ILIKE %s
                    ORDER BY document_path, chunk_index
                    LIMIT %s OFFSET %s
                    """,
                    (f"%{search}%", limit, offset),
                )
                rows = cursor.fetchall()

        return [
            StoredChunk(
                id=row[0],
                document_path=row[1],
                chunk_index=row[2],
                content=row[3],
                created_at=row[4].isoformat(),
                embedding=row[5],
                embedding_model=row[6],
                token_count=row[7],
            )
            for row in rows
        ]

    def search_similar(
        self,
        query_embedding: list[float],
        embedding_model: str,
        limit: int = 5,
    ) -> list[dict]:
        if not query_embedding:
            raise ValueError("Embedding вопроса пустой.")

        if not 1 <= limit <= 100:
            raise ValueError("Количество результатов должно быть от 1 до 100.")

        vector_literal = "[" + ",".join(
            format(float(value), ".9g") for value in query_embedding
        ) + "]"

        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        id,
                        document_path,
                        chunk_index,
                        content,
                        token_count,
                        embedding_model,
                        embedding_vector <=> %s::vector AS distance
                    FROM chunks
                    WHERE embedding_model = %s
                      AND embedding_vector IS NOT NULL
                    ORDER BY embedding_vector <=> %s::vector
                    LIMIT %s
                    """,
                    (
                        vector_literal,
                        embedding_model,
                        vector_literal,
                        limit,
                    ),
                )
                rows = cursor.fetchall()

        results = []
        for row in rows:
            distance = float(row[6])
            results.append(
                {
                    "id": row[0],
                    "document_path": row[1],
                    "chunk_index": row[2],
                    "content": row[3],
                    "token_count": row[4],
                    "embedding_model": row[5],
                    "distance": distance,
                    "similarity": 1.0 - distance,
                }
            )

        return results

    def count(self) -> int:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM chunks")
                return cursor.fetchone()[0]

    def count_documents(self) -> int:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(DISTINCT document_path) FROM chunks"
                )
                return cursor.fetchone()[0]
