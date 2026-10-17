
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LMStudioConfig:
    host: str = "127.0.0.1"
    port: int = 1234

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def models_url(self) -> str:
        return f"{self.base_url}/api/v0/models"

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/v1/chat/completions"


@dataclass(frozen=True)
class PostgresConfig:
    dsn: str = field(
        default_factory=lambda: os.getenv(
            "RAG_DATABASE_URL",
            "postgresql://postgres:postgres@localhost:5432/rag",
        )
    )


@dataclass(frozen=True)
class AppConfig:
    lm_studio: LMStudioConfig = field(
        default_factory=LMStudioConfig
    )

    postgres: PostgresConfig = field(
        default_factory=PostgresConfig
    )

    # Настройки семантического чанкинга
    target_chunk_tokens: int = 500
    min_chunk_ratio: float = 0.90
    max_chunk_ratio: float = 1.10
    chunk_overlap_tokens: int = 40
    semantic_boundary_check: bool = True


CONFIG = AppConfig()
