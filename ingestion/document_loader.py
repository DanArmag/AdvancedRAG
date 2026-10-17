
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LoadedDocument:
    path: str
    content: str


TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".tex",
    ".py", ".pyi", ".js", ".jsx", ".mjs", ".cjs",
    ".ts", ".tsx", ".java", ".kt", ".kts",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".hh",
    ".cs", ".go", ".rs", ".php", ".rb", ".swift",
    ".scala", ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd",
    ".sql", ".r", ".R", ".jl", ".lua", ".pl",
    ".html", ".htm", ".css", ".scss", ".sass", ".less",
    ".xml", ".svg", ".json", ".jsonl",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".env", ".properties", ".graphql", ".gql",
    ".proto", ".dockerfile", ".makefile",
    ".log", ".csv", ".tsv",
}

SPECIAL_TEXT_NAMES = {
    "dockerfile", "makefile", "justfile",
    ".gitignore", ".dockerignore", ".editorconfig",
    ".env", ".env.example", ".env.template",
}

IGNORED_DIRS = {
    ".git", ".venv", "venv", "__pycache__",
    "node_modules", "dist", "build",
    ".idea", ".vscode", ".pytest_cache",
}

MAX_FILE_BYTES = 10 * 1024 * 1024


def _is_supported_file(path: Path) -> bool:
    return (
        path.suffix.lower() in TEXT_EXTENSIONS
        or path.name.lower() in SPECIAL_TEXT_NAMES
    )


def _read_text_file(path: Path) -> str:
    raw = path.read_bytes()

    if len(raw) > MAX_FILE_BYTES:
        raise ValueError(
            f"Файл превышает лимит 10 МБ: {path}"
        )

    if b"\x00" in raw:
        raise ValueError(
            f"Обнаружен бинарный файл: {path}"
        )

    for encoding in ("utf-8-sig", "cp1251", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue

    raise UnicodeError(f"Не удалось декодировать: {path}")


def load_documents(folder_path: str) -> list[LoadedDocument]:
    root = Path(folder_path).expanduser().resolve()

    if not root.is_dir():
        raise NotADirectoryError(
            f"Папка не существует: {root}"
        )

    documents = []

    for current_dir, dirnames, filenames in __import__(
        "os"
    ).walk(root):
        dirnames[:] = sorted(
            name for name in dirnames
            if name.lower() not in IGNORED_DIRS
        )

        for filename in sorted(filenames):
            path = Path(current_dir) / filename

            if not _is_supported_file(path):
                continue

            try:
                content = _read_text_file(path)
            except (OSError, ValueError, UnicodeError) as exc:
                raise RuntimeError(
                    f"Ошибка чтения {path}: {exc}"
                ) from exc

            if content.strip():
                documents.append(
                    LoadedDocument(
                        path=str(path.resolve()),
                        content=content,
                    )
                )

    return documents
