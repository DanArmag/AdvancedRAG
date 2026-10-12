import json
from typing import Callable, List

from urllib.request import Request, urlopen

from config import LMStudioConfig
from models.chat import LoadedModels


class LMStudioError(Exception):
    pass


class LMStudioClient:

    def __init__(self, config: LMStudioConfig):
        self.config = config

    def get_loaded_models(self) -> LoadedModels:
        """
        Получает список моделей из LM Studio
        и возвращает только загруженные.
        """

        request = Request(
            self.config.models_url,
            method="GET",
            headers={
                "Content-Type": "application/json"
            }
        )

        try:
            with urlopen(request, timeout=3) as response:
                data = json.loads(
                    response.read().decode("utf-8")
                )

        except Exception as exc:
            raise LMStudioError(
                f"Не удалось подключиться к LM Studio: {exc}"
            ) from exc

        llm = None
        embedding = None

        for model in data.get("data", []):

            if model.get("state") != "loaded":
                continue

            model_type = model.get("type")
            model_id = model.get("id")

            if model_type == "llm":
                llm = model_id

            elif model_type in ("embedding", "embeddings"):
                embedding = model_id

        return LoadedModels(
            llm=llm,
            embedding=embedding
        )

    def stream_chat(
        self,
        model: str,
        messages: List[dict],
        on_token: Callable[[str], None],
        on_finished: Callable[[], None],
        on_error: Callable[[str], None],
    ):
        """
        Отправляет сообщение в LM Studio
        и получает ответ потоково.
        """

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": 0.7,
        }

        body = json.dumps(payload).encode("utf-8")

        request = Request(
            self.config.chat_url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            }
        )

        try:
            with urlopen(request, timeout=None) as response:

                while True:
                    line = response.readline()

                    if not line:
                        break

                    line = line.decode(
                        "utf-8"
                    ).strip()

                    if not line:
                        continue

                    if not line.startswith("data:"):
                        continue

                    data = line[5:].strip()

                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices", [])

                    if not choices:
                        continue

                    delta = choices[0].get(
                        "delta",
                        {}
                    )

                    text = delta.get("content")

                    if text:
                        on_token(text)

            on_finished()

        except Exception as exc:
            on_error(str(exc))