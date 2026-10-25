from api.embedding_client import EmbeddingClient
from api.lm_studio import LMStudioClient
from config import CONFIG
from storage.chunk_repository import ChunkRepository


class SemanticSearchService:
    """Ищет в pgvector; текст чанка читается из исходного документа."""

    def __init__(self, repository=None):
        self.repository = repository or ChunkRepository()
        self.embedding_client = EmbeddingClient()
        self.lm_client = LMStudioClient(CONFIG.lm_studio)

    def search(self, query: str, limit: int = 5) -> list[dict]:
        query = query.strip()
        if not query:
            raise ValueError("Введите вопрос для поиска.")
        if not 1 <= limit <= 100:
            raise ValueError("Количество результатов должно быть от 1 до 100.")

        models = self.lm_client.get_loaded_models()
        if not models.embedding:
            raise RuntimeError("В LM Studio не загружена embedding-модель.")

        vectors = self.embedding_client.embed_texts(
            model=models.embedding,
            texts=[query],
        )
        if not vectors or not vectors[0]:
            raise RuntimeError("Не удалось получить embedding для запроса.")

        return self.repository.search_similar(
            query_embedding=vectors[0],
            embedding_model=models.embedding,
            limit=limit,
        )
