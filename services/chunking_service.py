from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from api.embedding_client import EmbeddingClient
from api.lm_studio import LMStudioClient
from config import CONFIG
from ingestion.document_loader import load_documents
from ingestion.semantic_chunker import SemanticChunker
from storage.chunk_repository import ChunkRepository


ProgressCallback = Callable[[str, int, int, str], None]


@dataclass(frozen=True)
class ProcessingResult:
    documents_processed: int
    chunks_created: int
    documents_skipped: int = 0


class ChunkingService:
    def __init__(self, repository=None, batch_size=16):
        self.repository = repository or ChunkRepository()
        self.embedding_client = EmbeddingClient()
        self.lm_client = LMStudioClient(CONFIG.lm_studio)
        self.chunker = SemanticChunker()
        self.batch_size = batch_size

    def process_folder(self, folder_path, on_progress=None):
        def report(stage, current, total, filename=""):
            if on_progress:
                on_progress(stage, current, total, filename)

        documents = load_documents(folder_path)
        if not documents:
            return ProcessingResult(0, 0, 0)

        models = self.lm_client.get_loaded_models()
        if not models.llm:
            raise RuntimeError(
                "Для смыслового разбиения необходимо загрузить LLM."
            )
        if not models.embedding:
            raise RuntimeError("Не загружена embedding-модель.")

        processed = 0
        skipped = 0
        total_chunks = 0
        total_documents = len(documents)

        for doc_index, document in enumerate(documents, start=1):
            filename = document.path
            report("Проверка документа", doc_index - 1, total_documents, filename)

            if self.repository.is_document_unchanged(
                document.path, document.sha256, models.embedding
            ):
                skipped += 1
                report("Файл не изменился — пропуск", doc_index, total_documents, filename)
                continue

            chunks = self.chunker.split(
                text=document.content,
                model=models.llm,
                on_progress=lambda current, tokens, total: report(
                    f"Семантическое разбиение (чанк {current}, около {tokens} токенов)",
                    doc_index - 1,
                    total_documents,
                    filename,
                ),
            )

            token_counts = [self.chunker.count_tokens(chunk) for chunk in chunks]
            vectors = []
            for start in range(0, len(chunks), self.batch_size):
                batch = chunks[start:start + self.batch_size]
                report(
                    f"Создание embeddings: {start + 1}–{start + len(batch)} "
                    f"из {len(chunks)}",
                    doc_index - 1,
                    total_documents,
                    filename,
                )
                vectors.extend(self.embedding_client.embed_texts(
                    model=models.embedding,
                    texts=batch,
                ))

            chunk_records = []
            offset = 0
            for index, (text, vector, token_count) in enumerate(
                zip(chunks, vectors, token_counts)
            ):
                end_offset = offset + len(text)
                overlapping_units = [
                    unit for unit in document.units
                    if unit.start_char < end_offset and unit.end_char > offset
                ]

                metadata = {}
                for unit in overlapping_units:
                    for key, value in unit.metadata.items():
                        if value is not None:
                            metadata.setdefault(key, value)

                # Locator points into the original source document's extracted
                # text. Chunk text itself is never written to a sidecar/cache.
                locator = {
                    "kind": "document_range",
                    "start_char": offset,
                    "end_char": end_offset,
                }
                content_key = "source"
                chunk_records.append({
                    "content_key": content_key,
                    "locator": locator,
                    "metadata": metadata,
                    "token_count": token_count,
                    "embedding": vector,
                })
                offset = end_offset

            report("Сохранение поискового индекса", doc_index - 1, total_documents, filename)
            self.repository.replace_document_chunks(
                document_path=document.path,
                file_name=Path(document.path).name,
                file_type=document.file_type,
                sha256=document.sha256,
                file_size=document.file_size,
                chunks=chunk_records,
                embedding_model=models.embedding,
            )

            processed += 1
            total_chunks += len(chunks)
            report("Документ проиндексирован", doc_index, total_documents, filename)

        report("Готово", total_documents, total_documents)
        return ProcessingResult(processed, total_chunks, skipped)
