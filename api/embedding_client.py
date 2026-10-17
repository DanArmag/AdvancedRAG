
import json
import urllib.error
import urllib.request

from config import CONFIG


class EmbeddingClient:
    def __init__(self, timeout: int = 180):
        self.url = f"{CONFIG.lm_studio.base_url}/v1/embeddings"
        self.timeout = timeout

    def embed_texts(
        self,
        model: str,
        texts: list[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        payload = json.dumps({
            "model": model,
            "input": texts,
            "encoding_format": "float",
        }).encode("utf-8")

        request = urllib.request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout,
            ) as response:
                result = json.loads(
                    response.read().decode("utf-8")
                )
        except urllib.error.HTTPError as exc:
            details = exc.read().decode(
                "utf-8", errors="replace"
            )
            raise RuntimeError(
                f"Ошибка embedding API HTTP {exc.code}: {details}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Не удалось подключиться к LM Studio: {exc}"
            ) from exc
        except TimeoutError as exc:
            raise RuntimeError(
                "Превышено время ожидания embedding-модели."
            ) from exc

        data = result.get("data", [])

        # API может вернуть результаты не в исходном порядке.
        data.sort(key=lambda item: item.get("index", 0))

        if len(data) != len(texts):
            raise RuntimeError(
                f"Модель вернула {len(data)} векторов "
                f"для {len(texts)} текстов."
            )

        vectors = [item.get("embedding") for item in data]

        if any(not isinstance(v, list) or not v for v in vectors):
            raise RuntimeError(
                "Embedding API вернул пустой или некорректный вектор."
            )

        if any(
            not all(isinstance(x, (int, float)) for x in vector)
            for vector in vectors
        ):
            raise RuntimeError(
                "Embedding API вернул нечисловые значения."
            )

        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise RuntimeError(
                "Векторы одного пакета имеют разную размерность."
            )

        return vectors
