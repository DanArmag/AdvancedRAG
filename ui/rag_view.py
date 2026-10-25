import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from services.chunking_service import ChunkingService
from services.search_service import SemanticSearchService
from storage.chunk_repository import ChunkRepository


class RagView(ttk.Frame):
    def __init__(self, parent, on_busy=None):
        super().__init__(parent, padding=12)

        self.on_busy = on_busy or (lambda busy: None)
        self.repository = ChunkRepository()
        self.service = ChunkingService(self.repository)
        self.semantic_search = SemanticSearchService(self.repository)

        self.folder_var = tk.StringVar()
        self.search_var = tk.StringVar()
        self.semantic_query_var = tk.StringVar()
        self.semantic_limit_var = tk.StringVar(value="5")
        self.status_var = tk.StringVar(
            value="Выберите папку с документами."
        )
        self.stats_var = tk.StringVar(value="")

        self._chunk_by_iid = {}
        self._semantic_results_mode = False
        self._search_in_progress = False

        self._create_widgets()
        self.after(100, self.refresh_chunks)

    def _create_widgets(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)

        title = ttk.Label(
            self,
            text="Управление документами RAG",
            font=("Segoe UI", 17, "bold"),
        )
        title.grid(row=0, column=0, sticky="w", pady=(0, 12))

        folder_frame = ttk.LabelFrame(
            self,
            text="Источник документов",
            padding=10,
        )
        folder_frame.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        folder_frame.columnconfigure(0, weight=1)

        ttk.Entry(
            folder_frame,
            textvariable=self.folder_var,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 8))

        ttk.Button(
            folder_frame,
            text="Обзор…",
            command=self.choose_folder,
        ).grid(row=0, column=1, padx=(0, 8))

        self.process_button = ttk.Button(
            folder_frame,
            text="Разбить на чанки",
            command=self.start_processing,
        )
        self.process_button.grid(row=0, column=2)

        ttk.Label(
            folder_frame,
            text=(
                "Выберите папку с документами. Файлы обрабатываются "
                "согласно поддерживаемым форматам загрузчика."
            ),
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 0))

        self.progress = ttk.Progressbar(
            folder_frame,
            mode="determinate",
            maximum=100,
        )
        self.progress.grid(
            row=2,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(10, 4),
        )

        self.progress_label = ttk.Label(
            folder_frame,
            text="Ожидание обработки",
        )
        self.progress_label.grid(
            row=3,
            column=0,
            columnspan=3,
            sticky="w",
        )

        semantic_frame = ttk.LabelFrame(
            self,
            text="Семантический поиск по документам",
            padding=10,
        )
        semantic_frame.grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(0, 12),
        )
        semantic_frame.columnconfigure(0, weight=1)

        self.semantic_entry = ttk.Entry(
            semantic_frame,
            textvariable=self.semantic_query_var,
        )
        self.semantic_entry.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(0, 8),
        )
        self.semantic_entry.bind(
            "<Return>",
            lambda _event: self.start_semantic_search(),
        )

        ttk.Label(
            semantic_frame,
            text="Результатов:",
        ).grid(row=0, column=1, padx=(0, 5))

        self.semantic_limit_box = ttk.Combobox(
            semantic_frame,
            textvariable=self.semantic_limit_var,
            values=("3", "5", "10", "20"),
            width=5,
            state="readonly",
        )
        self.semantic_limit_box.grid(
            row=0,
            column=2,
            padx=(0, 8),
        )

        self.semantic_button = ttk.Button(
            semantic_frame,
            text="Искать по смыслу",
            command=self.start_semantic_search,
        )
        self.semantic_button.grid(row=0, column=3)

        self.semantic_hint = ttk.Label(
            semantic_frame,
            text=(
                "Пример: «Как приложение подключается к базе данных?»"
            ),
        )
        self.semantic_hint.grid(
            row=1,
            column=0,
            columnspan=4,
            sticky="w",
            pady=(7, 0),
        )

        content = ttk.Frame(self)
        content.grid(row=3, column=0, sticky="nsew")
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2)
        content.rowconfigure(1, weight=1)

        ttk.Label(
            content,
            text="Сохранённые чанки / результаты поиска",
            font=("Segoe UI", 12, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 6))

        toolbar = ttk.Frame(content)
        toolbar.grid(row=0, column=1, sticky="ew", pady=(0, 6))
        toolbar.columnconfigure(0, weight=1)

        search_entry = ttk.Entry(
            toolbar,
            textvariable=self.search_var,
        )
        search_entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        search_entry.bind(
            "<Return>",
            lambda _event: self.refresh_chunks(),
        )

        ttk.Button(
            toolbar,
            text="Найти файл",
            command=self.refresh_chunks,
        ).grid(row=0, column=1, padx=(0, 6))

        ttk.Button(
            toolbar,
            text="Обновить",
            command=self.refresh_chunks,
        ).grid(row=0, column=2)

        ttk.Button(
            toolbar,
            text="Очистить БД",
            command=self.clear_database,
        ).grid(row=0, column=3, padx=(8, 0))

        table_frame = ttk.Frame(content)
        table_frame.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=(0, 12),
        )
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        columns = ("id", "document", "index", "length", "tokens")
        self.tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            selectmode="browse",
        )

        self.tree.heading("id", text="ID")
        self.tree.heading("document", text="Документ")
        self.tree.heading("index", text="№")
        self.tree.heading("length", text="Символов")
        self.tree.heading("tokens", text="Токены")

        self.tree.column("id", width=65, anchor="center", stretch=False)
        self.tree.column("document", width=260)
        self.tree.column("index", width=55, anchor="center", stretch=False)
        self.tree.column("length", width=85, anchor="center", stretch=False)
        self.tree.column(
            "tokens",
            width=75,
            anchor="center",
            stretch=False,
        )

        scrollbar = ttk.Scrollbar(
            table_frame,
            orient="vertical",
            command=self.tree.yview,
        )
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.tree.bind("<<TreeviewSelect>>", self._show_selected_chunk)

        details = ttk.LabelFrame(
            content,
            text="Содержимое чанка",
            padding=8,
        )
        details.grid(row=1, column=1, sticky="nsew")
        details.rowconfigure(0, weight=1)
        details.columnconfigure(0, weight=1)

        self.detail_text = tk.Text(
            details,
            wrap="word",
            font=("Segoe UI", 10),
            relief="flat",
            padx=10,
            pady=10,
            state="disabled",
        )
        detail_scroll = ttk.Scrollbar(
            details,
            orient="vertical",
            command=self.detail_text.yview,
        )
        self.detail_text.configure(yscrollcommand=detail_scroll.set)
        self.detail_text.grid(row=0, column=0, sticky="nsew")
        detail_scroll.grid(row=0, column=1, sticky="ns")

        footer = ttk.Frame(self)
        footer.grid(row=4, column=0, sticky="ew", pady=(10, 0))

        ttk.Label(
            footer,
            textvariable=self.status_var,
        ).pack(side="left", fill="x", expand=True)

        ttk.Label(
            footer,
            textvariable=self.stats_var,
        ).pack(side="right")

    def choose_folder(self):
        folder = filedialog.askdirectory(
            title="Выберите папку с документами"
        )
        if folder:
            self.folder_var.set(folder)

    def clear_database(self):
        confirmed = messagebox.askyesno(
            "Подтверждение очистки",
            "Удалить все документы и чанки НОВОГО RAG-индекса?\nСтарая таблица chunks не изменится.\n\n"
            "Это действие нельзя отменить.",
            icon="warning",
            parent=self,
        )
        if not confirmed:
            return

        self.process_button.configure(state="disabled")
        self.status_var.set("Очистка базы данных...")
        threading.Thread(
            target=self._clear_database_worker,
            daemon=True,
        ).start()

    def _clear_database_worker(self):
        try:
            deleted_count = self.repository.clear_all()
            self.after(
                0,
                lambda: self._clear_database_finished(deleted_count),
            )
        except Exception as exc:
            error = str(exc)
            self.after(
                0,
                lambda: self._clear_database_failed(error),
            )

    def _clear_database_finished(self, deleted_count):
        self.process_button.configure(state="normal")
        self.progress.configure(value=0)
        self.progress_label.configure(text="Ожидание обработки")
        self.status_var.set(
            f"Новый индекс очищен. Удалено чанков: {deleted_count}."
        )
        self.refresh_chunks()

    def _clear_database_failed(self, error):
        self.process_button.configure(state="normal")
        self.status_var.set("Не удалось очистить базу данных.")
        messagebox.showerror(
            "Ошибка очистки",
            error,
            parent=self,
        )

    def start_processing(self):
        folder = self.folder_var.get().strip()
        if not folder:
            messagebox.showwarning(
                "Папка не выбрана",
                "Сначала выберите папку с документами.",
                parent=self,
            )
            return

        if not Path(folder).is_dir():
            messagebox.showerror(
                "Ошибка",
                "Указанная папка не существует.",
                parent=self,
            )
            return

        self.process_button.configure(state="disabled")
        self.progress.configure(value=0, maximum=100)
        self.progress_label.configure(text="Подготовка...")
        self.status_var.set("Запуск фоновой обработки...")
        self.on_busy(True)

        threading.Thread(
            target=self._process_worker,
            args=(folder,),
            daemon=True,
        ).start()

    def _process_worker(self, folder):
        def on_progress(stage, current, total, filename):
            self.after(
                0,
                lambda s=stage, c=current, t=total, f=filename:
                    self._update_progress(s, c, t, f),
            )

        try:
            result = self.service.process_folder(
                folder,
                on_progress=on_progress,
            )
            self.after(
                0,
                lambda: self._processing_finished(result),
            )
        except Exception as exc:
            error = str(exc)
            self.after(
                0,
                lambda: self._processing_failed(error),
            )

    def _update_progress(self, stage, current, total, filename):
        if total > 0:
            percent = current / total * 100
            self.progress.configure(value=percent)
            self.progress_label.configure(
                text=f"{stage}: {current} из {total} ({percent:.1f}%)"
            )
        else:
            self.progress_label.configure(text=stage)

        if filename:
            self.status_var.set(f"{stage} — {Path(filename).name}")
        else:
            self.status_var.set(stage)

    def _processing_finished(self, result):
        self.process_button.configure(state="normal")
        self.on_busy(False)
        self.progress.configure(value=100)
        self.progress_label.configure(text="Обработка завершена")
        self.status_var.set(
            f"Готово: документов — {result.documents_processed}, "
            f"чанков — {result.chunks_created}."
        )
        self.refresh_chunks()

    def _processing_failed(self, error):
        self.process_button.configure(state="normal")
        self.on_busy(False)
        self.status_var.set("Обработка завершилась с ошибкой.")
        messagebox.showerror(
            "Ошибка обработки",
            error,
            parent=self,
        )

    def start_semantic_search(self):
        query = self.semantic_query_var.get().strip()
        if not query:
            messagebox.showwarning(
                "Пустой запрос",
                "Введите вопрос для семантического поиска.",
                parent=self,
            )
            return

        if self._search_in_progress:
            return

        try:
            limit = int(self.semantic_limit_var.get())
        except ValueError:
            messagebox.showerror(
                "Ошибка",
                "Выберите количество результатов.",
                parent=self,
            )
            return

        self._search_in_progress = True
        self.semantic_button.configure(state="disabled")
        self.semantic_hint.configure(
            text="Ищу подходящие фрагменты. Подождите..."
        )
        self.status_var.set(
            "Создание embedding вопроса и поиск в PostgreSQL..."
        )

        threading.Thread(
            target=self._semantic_search_worker,
            args=(query, limit),
            daemon=True,
        ).start()

    def _semantic_search_worker(self, query, limit):
        try:
            results = self.semantic_search.search(query=query, limit=limit)
            self.after(
                0,
                lambda: self._display_semantic_results(results, query),
            )
        except Exception as exc:
            error = str(exc)
            self.after(
                0,
                lambda: self._semantic_search_failed(error),
            )

    def _display_semantic_results(self, results, query):
        self._search_in_progress = False
        self.semantic_button.configure(state="normal")
        self._semantic_results_mode = True

        for item in self.tree.get_children():
            self.tree.delete(item)
        self._chunk_by_iid = {}

        self.tree.heading("length", text="Сходство")
        for result in results:
            iid = f"semantic_{result['id']}"
            self._chunk_by_iid[iid] = result
            self.tree.insert(
                "",
                "end",
                iid=iid,
                values=(
                    result["id"],
                    Path(result["document_path"]).name,
                    result["chunk_index"],
                    f"{result['similarity']:.4f}",
                    result["token_count"]
                    if result["token_count"] is not None
                    else "—",
                ),
            )

        self.stats_var.set(f"Результатов: {len(results)}")
        if results:
            self._set_detail(
                f"Запрос: {query}\n\n"
                "Выберите результат в таблице, чтобы увидеть текст чанка."
            )
            self.status_var.set(
                f"Поиск завершён. Найдено фрагментов: {len(results)}."
            )
            self.semantic_hint.configure(
                text="Результаты отсортированы по косинусному расстоянию."
            )
        else:
            self._set_detail(
                "Подходящие фрагменты не найдены.\n\n"
                "Проверьте, что документы проиндексированы и у них "
                "заполнена колонка embedding_vector."
            )
            self.status_var.set("Поиск завершён: результатов нет.")
            self.semantic_hint.configure(
                text="Попробуйте переформулировать вопрос."
            )

    def _semantic_search_failed(self, error):
        self._search_in_progress = False
        self.semantic_button.configure(state="normal")
        self.status_var.set("Семантический поиск завершился с ошибкой.")
        self.semantic_hint.configure(
            text="Поиск не выполнен. Подробности показаны в окне ошибки."
        )
        messagebox.showerror(
            "Ошибка семантического поиска",
            error,
            parent=self,
        )

    def refresh_chunks(self):
        # Обычное обновление возвращает таблицу к списку всех чанков.
        threading.Thread(
            target=self._refresh_worker,
            args=(self.search_var.get().strip(),),
            daemon=True,
        ).start()

    def _refresh_worker(self, search):
        try:
            chunks = self.repository.get_all(
                search=search,
                limit=1000,
                offset=0,
            )
            total_chunks = self.repository.count()
            total_documents = self.repository.count_documents()
            self.after(
                0,
                lambda: self._display_chunks(
                    chunks,
                    total_chunks,
                    total_documents,
                ),
            )
        except Exception as exc:
            error = str(exc)
            self.after(0, lambda: self._display_error(error))

    def _display_chunks(self, chunks, total_chunks, total_documents):
        self._semantic_results_mode = False
        self.tree.heading("length", text="Символов")

        for item in self.tree.get_children():
            self.tree.delete(item)
        self._chunk_by_iid = {}

        for chunk in chunks:
            iid = str(chunk.id)
            self._chunk_by_iid[iid] = chunk
            self.tree.insert(
                "",
                "end",
                iid=iid,
                values=(
                    chunk.id,
                    Path(chunk.document_path).name,
                    chunk.chunk_index,
                    len(chunk.content),
                    chunk.token_count
                    if chunk.token_count is not None
                    else "—",
                ),
            )

        self.stats_var.set(
            f"В БД: {total_documents} документов · {total_chunks} чанков"
        )
        self._set_detail(
            "Выберите чанк в таблице, чтобы увидеть его текст."
        )
        self.status_var.set(
            f"Показано {len(chunks)} чанков. При необходимости уточните поиск."
        )

    def _show_selected_chunk(self, _event=None):
        selected = self.tree.selection()
        if not selected:
            return

        item = self._chunk_by_iid.get(selected[0])
        if item is None:
            return

        # Результат семантического поиска представлен словарём.
        if isinstance(item, dict):
            token_count = (
                item["token_count"]
                if item["token_count"] is not None
                else "не рассчитано"
            )
            self._set_detail(
                f"Документ: {item['document_path']}\n"
                f"Чанк №: {item['chunk_index']}\n"
                f"Embedding-модель: {item['embedding_model']}\n"
                f"Косинусное расстояние: {item['distance']:.6f}\n"
                f"Оценка сходства: {item['similarity']:.4f}\n"
                f"Токенов: {token_count}\n\n"
                f"Текст чанка:\n{item['content']}"
            )
            return

        chunk = item
        if chunk.embedding:
            vector_preview = ", ".join(
                f"{value:.5f}" for value in chunk.embedding[:10]
            )
            embedding_info = (
                f"Embedding-модель: {chunk.embedding_model}\n"
                f"Размерность вектора: {len(chunk.embedding)}\n"
                f"Первые 10 значений: [{vector_preview}]\n"
            )
        else:
            embedding_info = "Embedding отсутствует\n"

        token_count = (
            chunk.token_count
            if chunk.token_count is not None
            else "не рассчитано"
        )
        self._set_detail(
            f"Документ: {chunk.document_path}\n"
            f"Чанк №: {chunk.chunk_index}\n"
            f"Длина: {len(chunk.content)} символов\n"
            f"Создан: {chunk.created_at}\n"
            f"{embedding_info}\n"
            f"Токенов: {token_count}\n"
            f"Текст чанка:\n{chunk.content}"
        )

    def _set_detail(self, text):
        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", "end")
        self.detail_text.insert("1.0", text)
        self.detail_text.configure(state="disabled")

    def _display_error(self, error):
        self.status_var.set("Не удалось загрузить чанки из БД.")
        messagebox.showerror(
            "Ошибка PostgreSQL",
            error,
            parent=self,
        )
