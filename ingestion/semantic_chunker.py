
import json
import re
import urllib.error
import urllib.request

import tiktoken

from config import CONFIG


class SemanticChunker:
    def __init__(self):
        self.target = CONFIG.target_chunk_tokens
        self.minimum = int(
            self.target * CONFIG.min_chunk_ratio
        )
        self.maximum = int(
            self.target * CONFIG.max_chunk_ratio
        )
        self.overlap = CONFIG.chunk_overlap_tokens
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def _fallback_index(self, candidate_texts: list[str]) -> int:
        """Выбирает допустимую границу без участия LLM."""
        if not candidate_texts:
            return 0

        return min(
            range(len(candidate_texts)),
            key=lambda i: abs(
                self.count_tokens(candidate_texts[i]) - self.target
            ),
        )

    def _units(self, text: str) -> list[str]:
        # Сохраняем разделители, чтобы восстановить исходный текст.
        parts = re.split(r"(\n+|(?<=[.!?])\s+)", text)
        return [part for part in parts if part]

    def _ask_llm(
            self,
            model: str,
            current_text: str,
            candidate_texts: list[str],
    ) -> int:
        candidates = []

        for i, candidate in enumerate(candidate_texts):
            boundary = len(candidate)

            candidates.append({
                "index": i,
                "ending": candidate[-160:],
                "next": current_text[boundary:boundary + 160],
            })

        prompt = f"""
    Ты выбираешь границу смыслового чанка.

    Цель: около {self.target} токенов.
    Допустимый диапазон: от {self.minimum} до {self.maximum}.

    Выбирай границу, которая:
    - не разрывает предложение, если есть альтернатива;
    - сохраняет функцию, класс или логический блок кода;
    - не отделяет комментарий от связанного с ним кода;
    - не разрывает список, таблицу или структурированный блок;
    - по возможности приближается к целевому размеру.

    Выбирай только один из предложенных индексов.

    Варианты:
    {json.dumps(candidates, ensure_ascii=False)}

    Ответь строго JSON:
    {{"index": 0}}
    """

        payload = json.dumps({
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Выбери лучшую границу текста. "
                        "Возвращай только валидный JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 50,
            "stream": False,
        }).encode("utf-8")

        request = urllib.request.Request(
            CONFIG.lm_studio.chat_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                result = json.loads(response.read().decode("utf-8"))

        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ошибка обращения к LLM при выборе границ: "
                f"HTTP {exc.code} {exc.reason}\n"
                f"Ответ LM Studio: {details}"
            ) from exc

        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"Ошибка обращения к LLM при выборе границ: {exc}"
            ) from exc

        content = result["choices"][0]["message"]["content"]

        try:
            # Удаляем возможные Markdown-ограждения.
            cleaned = content.strip()
            cleaned = re.sub(
                r"^```(?:json)?\s*|\s*```$",
                "",
                cleaned,
                flags=re.IGNORECASE,
            ).strip()

            parsed = json.loads(cleaned)

            # Принимаем только ожидаемый формат:
            # {"index": 0}
            if not isinstance(parsed, dict) or set(parsed.keys()) != {"index"}:
                raise ValueError("Ожидался JSON только с полем 'index'.")

            raw_index = parsed["index"]

            # Не принимаем bool и дробные числа вместо целого индекса.
            if isinstance(raw_index, bool):
                raise ValueError("Индекс не может быть boolean.")

            if isinstance(raw_index, int):
                index = raw_index
            elif isinstance(raw_index, str) and raw_index.strip().isdigit():
                index = int(raw_index.strip())
            else:
                raise ValueError("Индекс должен быть целым числом.")

            if not 0 <= index < len(candidate_texts):
                raise ValueError(f"Индекс вне допустимого диапазона: {index}")

            return index

        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            fallback = self._fallback_index(candidate_texts)

            print(
                "[SemanticChunker] Некорректный ответ LLM. "
                f"Используется резервная граница {fallback}. "
                f"Причина: {exc}. Ответ модели: {content[:500]!r}"
            )

            return fallback

    def split(
        self,
        text: str,
        model: str,
        on_progress=None,
    ) -> list[str]:
        if not text.strip():
            return []

        units = self._units(text)
        chunks = []
        start = 0
        total_units = len(units)

        while start < total_units:
            candidates = []
            candidate_token_counts = []
            combined = ""

            for end in range(start, total_units):
                combined += units[end]
                token_count = self.count_tokens(combined)

                # Не рассматриваем границы слишком далеко за максимумом.
                if token_count > self.maximum:
                    break

                if token_count >= self.minimum:
                    candidates.append(combined)
                    candidate_token_counts.append(token_count)

            if not candidates:
                # Длинный неделимый блок: выбираем максимально возможную
                # границу, даже если она меньше минимального размера.
                combined = ""
                end = start

                while end < total_units:
                    proposed = combined + units[end]
                    if combined and self.count_tokens(proposed) > self.maximum:
                        break
                    combined = proposed
                    end += 1

                if end == start:
                    end += 1
                    combined = units[start]

                chunks.append(combined)
                start = end
                continue

            # Текст, который LLM оценивает при выборе границы.
            available_text = "".join(units[start:])
            candidate_texts = []

            for candidate in candidates:
                candidate_texts.append(candidate)

            if CONFIG.semantic_boundary_check:
                # Ограничиваем число вариантов, отправляемых модели.
                max_candidates = 8
                candidate_indices = list(range(len(candidates)))

                if len(candidate_indices) > max_candidates:
                    last = len(candidate_indices) - 1
                    candidate_indices = sorted({
                        round(i * last / (max_candidates - 1))
                        for i in range(max_candidates)
                    })

                sampled_candidates = [
                    candidates[i] for i in candidate_indices
                ]

                selected = self._ask_llm(
                    model=model,
                    current_text=available_text,
                    candidate_texts=sampled_candidates,
                )

                selected = candidate_indices[selected]

            else:
                selected = min(
                    range(len(candidates)),
                    key=lambda i: abs(
                        candidate_token_counts[i] - self.target
                    ),
                )

            chosen = candidates[selected]
            chunks.append(chosen)

            consumed = len(chosen)
            consumed_text = "".join(units[start:])

            # Ищем точную границу в исходных единицах.
            boundary = start
            chars = 0

            while boundary < total_units and chars < consumed:
                chars += len(units[boundary])
                boundary += 1

            if chars != consumed:
                raise RuntimeError(
                    "Не удалось сопоставить границу чанка с исходным текстом."
                )

            if on_progress:
                on_progress(
                    len(chunks),
                    self.count_tokens(chosen),
                    total_units,
                )

            start = boundary

        # Проверяем, что все фрагменты в сумме сохраняют исходный текст.
        if "".join(chunks) != text:
            raise RuntimeError(
                "Ошибка целостности: чанки не совпадают с исходным текстом."
            )

        return chunks
