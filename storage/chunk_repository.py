import uuid
from dataclasses import dataclass

from psycopg2.extras import Json

from storage.postgres import get_connection
from ingestion.document_loader import load_document
from pathlib import Path


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
    content_key: str | None = None
    locator: dict | None = None
    metadata: dict | None = None


class ChunkRepository:
    def __init__(self, content_store=None):
        # Kept as an ignored argument for compatibility with callers from the
        # previous version. Chunk text is read from original files, not cache.
        self._source_cache: dict[str, tuple[tuple[int, int], str]] = {}

    def _read_source_range(self, document_path: str, locator: dict | None) -> str:
        locator = locator or {}
        if locator.get("kind") != "document_range":
            raise RuntimeError(
                "У этого чанка старый формат locator. Переиндексируйте документ, "
                "чтобы читать текст напрямую из исходного файла."
            )
        path = Path(document_path)
        try:
            stat = path.stat()
        except OSError as exc:
            raise FileNotFoundError(
                f"Исходный файл недоступен: {document_path}. "
                "Восстановите файл или переиндексируйте документ."
            ) from exc
        signature = (stat.st_mtime_ns, stat.st_size)
        cached = self._source_cache.get(document_path)
        if cached is None or cached[0] != signature:
            document = load_document(path)
            self._source_cache[document_path] = (signature, document.content)
            source_text = document.content
        else:
            source_text = cached[1]
        start = int(locator.get("start_char", -1))
        end = int(locator.get("end_char", -1))
        if start < 0 or end < start or end > len(source_text):
            raise RuntimeError(
                f"Диапазон чанка больше не соответствует файлу {document_path}. "
                "Файл изменился — переиндексируйте документ."
            )
        return source_text[start:end]

    @staticmethod
    def _vector_literal(vector: list[float]) -> str:
        if not vector:
            raise ValueError("Embedding-вектор пуст.")
        return "[" + ",".join(
            format(float(value), ".9g") for value in vector
        ) + "]"

    def is_document_unchanged(self, document_path: str, sha256: str, embedding_model: str) -> bool:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT d.sha256
                    FROM rag_documents d
                    WHERE d.source_uri = %s
                      AND EXISTS (
                          SELECT 1 FROM rag_chunks c
                          WHERE c.document_id = d.id
                            AND c.embedding_model = %s
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM rag_chunks old_c
                          WHERE old_c.document_id = d.id
                            AND old_c.embedding_model = %s
                            AND old_c.locator->>'kind' IS DISTINCT FROM 'document_range'
                      )
                    """,
                    (document_path, embedding_model, embedding_model),
                )
                row = cursor.fetchone()
        return bool(row and row[0] == sha256)

    def replace_document_chunks(
        self,
        *,
        document_path: str,
        file_name: str,
        file_type: str,
        sha256: str,
        file_size: int,
        chunks: list[dict],
        embedding_model: str,
    ) -> int:
        """Атомарно заменяет документ и его чанки; старая схема не затрагивается."""
        document_id = None
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT id FROM rag_documents WHERE source_uri = %s",
                    (document_path,),
                )
                existing = cursor.fetchone()
                if existing:
                    document_id = existing[0]
                    cursor.execute(
                        """
                        UPDATE rag_documents
                        SET file_name=%s, file_type=%s, sha256=%s,
                            file_size=%s, status='indexed',
                            indexed_at=NOW()
                        WHERE id=%s
                        """,
                        (file_name, file_type, sha256, file_size, document_id),
                    )
                    cursor.execute(
                        "DELETE FROM rag_chunks WHERE document_id = %s",
                        (document_id,),
                    )
                else:
                    document_id = str(uuid.uuid4())
                    cursor.execute(
                        """
                        INSERT INTO rag_documents
                            (id, source_uri, file_name, file_type,
                             sha256, file_size, status)
                        VALUES (%s, %s, %s, %s, %s, %s, 'indexed')
                        """,
                        (
                            document_id, document_path, file_name,
                            file_type, sha256, file_size,
                        ),
                    )

                rows = []
                for index, item in enumerate(chunks):
                    vector = item["embedding"]
                    rows.append((
                        document_id,
                        index,
                        item["content_key"],
                        Json(item.get("locator", {})),
                        Json(item.get("metadata", {})),
                        item.get("token_count"),
                        embedding_model,
                        self._vector_literal(vector),
                    ))

                if rows:
                    cursor.executemany(
                        """
                        INSERT INTO rag_chunks (
                            document_id, chunk_index, content_key,
                            locator, metadata, token_count,
                            embedding_model, embedding_vector
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                        """,
                        rows,
                    )
        return len(chunks)

    def clear_all(self) -> int:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM rag_chunks")
                deleted = cursor.fetchone()[0]
                cursor.execute("DELETE FROM rag_chunks")
                cursor.execute("DELETE FROM rag_documents")
        return deleted

    def get_all(self, search: str = "", limit: int = 1000, offset: int = 0):
        if limit < 1 or offset < 0:
            raise ValueError("Некорректные limit или offset.")
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT c.id, d.source_uri, c.chunk_index,
                           c.content_key, c.created_at, c.embedding_model,
                           c.token_count, c.locator, c.metadata
                    FROM rag_chunks c
                    JOIN rag_documents d ON d.id = c.document_id
                    WHERE d.source_uri ILIKE %s OR d.file_name ILIKE %s
                    ORDER BY d.source_uri, c.chunk_index
                    LIMIT %s OFFSET %s
                    """,
                    (f"%{search}%", f"%{search}%", limit, offset),
                )
                rows = cursor.fetchall()

        result = []
        for row in rows:
            content = self._read_source_range(row[1], row[7])
            result.append(StoredChunk(
                id=row[0],
                document_path=row[1],
                chunk_index=row[2],
                content=content,
                created_at=row[4].isoformat(),
                embedding=None,
                embedding_model=row[5],
                token_count=row[6],
                content_key=row[3],
                locator=row[7],
                metadata=row[8],
            ))
        return result

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
        vector_literal = self._vector_literal(query_embedding)
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT c.id, d.source_uri, d.file_name, c.chunk_index,
                           c.content_key, c.token_count, c.embedding_model,
                           c.locator, c.metadata,
                           c.embedding_vector <=> %s::vector AS distance
                    FROM rag_chunks c
                    JOIN rag_documents d ON d.id = c.document_id
                    WHERE c.embedding_model = %s
                    ORDER BY c.embedding_vector <=> %s::vector
                    LIMIT %s
                    """,
                    (vector_literal, embedding_model, vector_literal, limit),
                )
                rows = cursor.fetchall()

        results = []
        for row in rows:
            distance = float(row[9])
            content = self._read_source_range(row[1], row[7])
            results.append({
                "id": row[0],
                "document_path": row[1],
                "file_name": row[2],
                "chunk_index": row[3],
                "content_key": row[4],
                "content": content,
                "token_count": row[5],
                "embedding_model": row[6],
                "locator": row[7],
                "metadata": row[8],
                "distance": distance,
                "similarity": 1.0 - distance,
            })
        return results

    def count(self) -> int:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM rag_chunks")
                return cursor.fetchone()[0]

    def count_documents(self) -> int:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM rag_documents")
                return cursor.fetchone()[0]
