from dataclasses import dataclass


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
class AppConfig:
    lm_studio: LMStudioConfig = LMStudioConfig()


CONFIG = AppConfig()