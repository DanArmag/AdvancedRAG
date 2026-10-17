
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from api.lm_studio import LMStudioClient, LMStudioError
from config import CONFIG
from models.chat import ChatHistory
from storage.postgres import initialize_database

from ui.chat_view import ChatView
from ui.input_bar import InputBar
from ui.model_info import ModelInfo
from ui.rag_view import RagView


class ChatApp:

    def __init__(self, root: tk.Tk):
        self.root = root

        self.root.title("Adaptive Evidence RAG")
        self.root.geometry("1200x780")
        self.root.minsize(850, 550)

        self.client = LMStudioClient(CONFIG.lm_studio)
        self.history = ChatHistory()
        self.chat_model = None
        self.generating = False

        try:
            initialize_database()
        except Exception as exc:
            messagebox.showerror(
                "PostgreSQL недоступен",
                "Не удалось подключиться к PostgreSQL "
                "или создать таблицу chunks.\n\n"
                f"{exc}\n\n"
                "Проверь Docker Compose и RAG_DATABASE_URL.",
                parent=self.root,
            )

        self._create_ui()
        self.refresh_models()

    def _create_ui(self):
        self._create_header()

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)

        # -------------------------------
        # Chat tab
        # -------------------------------
        self.chat_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.chat_tab, text="Чат")

        self.model_info = ModelInfo(
            self.chat_tab,
            on_refresh=self.refresh_models,
        )
        self.model_info.pack(side="top", fill="x")

        self.chat_view = ChatView(self.chat_tab)
        self.chat_view.pack(fill="both", expand=True)

        self.input_bar = InputBar(
            self.chat_tab,
            on_send=self.send_message,
        )
        self.input_bar.pack(side="bottom", fill="x")

        self.root.bind(
            "<Control-Return>",
            lambda event: self.send_message(),
        )

        # -------------------------------
        # RAG tab
        # -------------------------------
        self.rag_view = RagView(self.notebook)
        self.notebook.add(self.rag_view, text="Документы и чанки")

    def _create_header(self):
        header = tk.Frame(
            self.root,
            bg="#202124",
            height=65,
        )
        header.pack(side="top", fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="Adaptive Evidence RAG",
            bg="#202124",
            fg="white",
            font=("Segoe UI", 16, "bold"),
        ).pack(side="left", padx=18)

        self.status_label = tk.Label(
            header,
            text="Подключение...",
            bg="#202124",
            fg="#aaaaaa",
            font=("Segoe UI", 9),
        )
        self.status_label.pack(side="right", padx=18)

    # ===================================
    # LM Studio
    # ===================================
    def refresh_models(self):
        self.status_label.config(text="Проверка LM Studio...")

        threading.Thread(
            target=self._refresh_models_worker,
            daemon=True,
        ).start()

    def _refresh_models_worker(self):
        try:
            models = self.client.get_loaded_models()
            self.root.after(
                0,
                lambda: self._models_loaded(models),
            )
        except LMStudioError as exc:
            error = str(exc)
            self.root.after(
                0,
                lambda: self._models_error(error),
            )

    def _models_loaded(self, models):
        self.model_info.update_models(models)
        self.chat_model = models.llm
        self.status_label.config(text="LM Studio подключён")

        if not self.chat_model:
            self.chat_view.append(
                "⚠ Модель генерации не загружена.\n"
                "Загрузи LLM в LM Studio.\n\n",
                "system",
            )

    def _models_error(self, error):
        self.model_info.set_error()
        self.status_label.config(text="LM Studio недоступен")
        self.chat_view.append(
            f"⚠ Не удалось подключиться к LM Studio.\n"
            f"{error}\n\n",
            "system",
        )

    # ===================================
    # Chat
    # ===================================
    def send_message(self):
        if self.generating:
            return

        if not self.chat_model:
            messagebox.showwarning(
                "Нет модели",
                "В LM Studio не найдена загруженная модель.",
                parent=self.root,
            )
            return

        text = self.input_bar.get_text()
        if not text:
            return

        self.input_bar.clear()

        self.chat_view.append("Вы:\n", "user")
        self.chat_view.append(text + "\n\n")
        self.history.add_user_message(text)

        self.chat_view.append("Модель:\n", "assistant")

        self.generating = True
        self.input_bar.set_enabled(False)
        self.input_bar.set_generating(True)

        threading.Thread(
            target=self._generation_worker,
            daemon=True,
        ).start()

    def _generation_worker(self):
        response_parts = []

        def on_token(token: str):
            response_parts.append(token)
            self.root.after(
                0,
                lambda token=token: self.chat_view.append(
                    token, "assistant"
                ),
            )

        def on_finished():
            response = "".join(response_parts)
            self.history.add_assistant_message(response)
            self.root.after(0, self._generation_finished)

        def on_error(error: str):
            self.root.after(
                0,
                lambda error=error: self._generation_error(error),
            )

        self.client.stream_chat(
            model=self.chat_model,
            messages=self.history.get_messages(),
            on_token=on_token,
            on_finished=on_finished,
            on_error=on_error,
        )

    def _generation_finished(self):
        self.chat_view.append("\n\n")
        self.generating = False
        self.input_bar.set_enabled(True)
        self.input_bar.set_generating(False)

    def _generation_error(self, error):
        self.chat_view.append(
            f"\n\n⚠ Ошибка генерации:\n{error}\n\n",
            "system",
        )
        self.generating = False
        self.input_bar.set_enabled(True)
        self.input_bar.set_generating(False)
