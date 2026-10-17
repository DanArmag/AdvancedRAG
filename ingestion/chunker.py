
def split_into_chunks(
    text: str,
    chunk_size: int = 400,
) -> list[str]:
    """
    Разбивает текст на последовательные фрагменты
    длиной не более chunk_size символов.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size должен быть больше нуля")

    text = text.strip()

    if not text:
        return []

    return [
        text[start:start + chunk_size]
        for start in range(0, len(text), chunk_size)
    ]
