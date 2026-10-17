from dataclasses import dataclass
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
            return ProcessingResult(0, 0)

        models = self.lm_client.get_loaded_models()
        if not models.llm:
            raise RuntimeError(
                "Для смыслового разбиения необходимо загрузить LLM."
            )
        if not models.embedding:
            raise RuntimeError("Не загружена embedding-модель.")

        total_documents = len(documents)
        processed_documents = 0
        total_chunks = 0

        for document_index, document in enumerate(documents, start=1):
            filename = document.path
            report(
                "Семантическое разбиение",
                document_index - 1,
                total_documents,
                filename,
            )

            chunks = self.chunker.split(
                text=document.content,
                model=models.llm,
                on_progress=lambda current, tokens, total: report(
                    f"Выбор логических границ "
                    f"(чанк {current}, около {tokens} токенов)",
                    document_index - 1,
                    total_documents,
                    filename,
                ),
            )

            # Считаем токены один раз на чанк, а не на каждом пакете.
            token_counts = [
                self.chunker.count_tokens(chunk) for chunk in chunks
            ]

            vectors = []
            for start in range(0, len(chunks), self.batch_size):
                batch = chunks[start:start + self.batch_size]
                report(
                    f"Создание embeddings: "
                    f"{start + 1}–{start + len(batch)} из {len(chunks)}",
                    document_index - 1,
                    total_documents,
                    filename,
                )
                vectors.extend(
                    self.embedding_client.embed_texts(
                        model=models.embedding,
                        texts=batch,
                    )
                )

            report(
                "Сохранение в PostgreSQL",
                document_index - 1,
                total_documents,
                filename,
            )
            self.repository.replace_document_chunks(
                document_path=document.path,
                chunks=chunks,
                embeddings=vectors,
                embedding_model=models.embedding,
                token_counts=token_counts,
            )

            processed_documents += 1
            total_chunks += len(chunks)
            report(
                "Документы обработаны",
                processed_documents,
                total_documents,
                filename,
            )

        report("Готово", total_documents, total_documents)
        return ProcessingResult(processed_documents, total_chunks)
