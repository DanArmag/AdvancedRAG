import psycopg2
from psycopg2.extensions import connection

from config import CONFIG


def get_connection() -> connection:
    return psycopg2.connect(CONFIG.postgres.dsn)


def initialize_database() -> None:
    """Создаёт таблицу чанков и при необходимости обновляет её схему."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            # Расширение должно быть создано до использования типа vector.
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id BIGSERIAL PRIMARY KEY,
                    document_path TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding JSONB,
                    embedding_vector vector,
                    embedding_model TEXT,
                    token_count INTEGER,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    CONSTRAINT uq_chunks_document_index
                        UNIQUE (document_path, chunk_index),
                    CONSTRAINT chk_chunks_index
                        CHECK (chunk_index >= 0),
                    CONSTRAINT chk_chunks_token_count
                        CHECK (token_count IS NULL OR token_count >= 0)
                )
                """
            )

            # Миграции для уже существующей таблицы.
            cursor.execute(
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding JSONB"
            )
            cursor.execute(
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_vector vector"
            )
            cursor.execute(
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_model TEXT"
            )
            cursor.execute(
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS token_count INTEGER"
            )
            cursor.execute(
                "ALTER TABLE chunks DROP CONSTRAINT IF EXISTS chk_chunks_content_length"
            )

            cursor.execute(
                """
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1
                        FROM pg_constraint
                        WHERE conname = 'chk_chunks_token_count'
                          AND conrelid = 'chunks'::regclass
                    ) THEN
                        ALTER TABLE chunks
                            ADD CONSTRAINT chk_chunks_token_count
                            CHECK (token_count IS NULL OR token_count >= 0);
                    END IF;
                END
                $$;
                """
            )

            # Переносим старые JSONB-векторы в колонку pgvector.
            cursor.execute(
                """
                UPDATE chunks
                SET embedding_vector = embedding::text::vector
                WHERE embedding IS NOT NULL
                  AND embedding_vector IS NULL
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_document_path
                ON chunks (document_path)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_created_at
                ON chunks (created_at DESC)
                """
            )
