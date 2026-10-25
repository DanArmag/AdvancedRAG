import hashlib
import json
from pathlib import Path
from typing import Any


class ChunkContentStore:
    """Файловый кэш содержимого чанков, не хранящийся в PostgreSQL."""

    def __init__(self, root: str | Path | None = None):
        if root is None:
            root = Path(__file__).resolve().parents[1] / ".rag_cache" / "chunks"
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, payload: dict[str, Any]) -> str:
        serialized = json.dumps(
            payload, ensure_ascii=False, sort_keys=True
        ).encode("utf-8")
        key = hashlib.sha256(serialized).hexdigest()
        relative = Path(key[:2]) / f"{key}.json"
        target = (self.root / relative).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError("Недопустимый ключ содержимого")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            temp = target.with_suffix(".tmp")
            temp.write_bytes(serialized)
            temp.replace(target)
        return relative.as_posix()

    def get(self, key: str) -> dict[str, Any]:
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Недопустимый путь к содержимому")
        if not path.is_file():
            raise FileNotFoundError(
                f"Не найден кэш чанка {key}. Повторно проиндексируйте документ."
            )
        return json.loads(path.read_text(encoding="utf-8"))
