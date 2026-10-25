import psycopg2
from psycopg2.extensions import connection

from config import CONFIG


def get_connection() -> connection:
    return psycopg2.connect(CONFIG.postgres.dsn)


def initialize_database() -> None:
    """Создаёт новую схему RAG, не изменяя прежнюю таблицу chunks."""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rag_documents (
                    id UUID PRIMARY KEY,
                    source_uri TEXT NOT NULL UNIQUE,
                    file_name TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    file_size BIGINT NOT NULL CHECK (file_size >= 0),
                    status TEXT NOT NULL DEFAULT 'indexed',
                    indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    id BIGSERIAL PRIMARY KEY,
                    document_id UUID NOT NULL
                        REFERENCES rag_documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
                    content_key TEXT NOT NULL DEFAULT 'source',
                    locator JSONB NOT NULL DEFAULT '{}'::jsonb,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    token_count INTEGER CHECK (
                        token_count IS NULL OR token_count >= 0
                    ),
                    embedding_model TEXT NOT NULL,
                    embedding_vector vector NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE (document_id, chunk_index, embedding_model)
                )
            """)

            cursor.execute("ALTER TABLE rag_chunks ALTER COLUMN content_key SET DEFAULT 'source'")

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_rag_chunks_document_id
                ON rag_chunks (document_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_rag_chunks_model
                ON rag_chunks (embedding_model)
            """)
            # Векторный индекс намеренно не создаём автоматически:
            # модели могут иметь разные размерности. После выбора одной
            # embedding-модели можно добавить HNSW индекс под её размерность.
