import tkinter as tk
from tkinter import ttk

from models.chat import LoadedModels


class ModelInfo(tk.Frame):

    def __init__(
        self,
        parent,
        on_refresh
    ):
        super().__init__(
            parent,
            bg="#f3f3f3",
            height=70
        )

        self.on_refresh = on_refresh

        self.pack_propagate(False)

        self._create_widgets()

    def _create_widgets(self):

        # --------------------------
        # LLM
        # --------------------------

        llm_frame = tk.Frame(
            self,
            bg="#f3f3f3"
        )

        llm_frame.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(15, 5),
            pady=10
        )

        tk.Label(
            llm_frame,
            text="Модель генерации",
            bg="#f3f3f3",
            fg="#666666",
            font=("Segoe UI", 9)
        ).pack(
            anchor="w"
        )

        self.llm_label = tk.Label(
            llm_frame,
            text="Проверка...",
            bg="#f3f3f3",
            fg="#111111",
            font=("Segoe UI", 10, "bold")
        )

        self.llm_label.pack(
            anchor="w"
        )

        # --------------------------
        # Embedding
        # --------------------------

        embedding_frame = tk.Frame(
            self,
            bg="#f3f3f3"
        )

        embedding_frame.pack(
            side="left",
            fill="both",
            expand=True,
            padx=5,
            pady=10
        )

        tk.Label(
            embedding_frame,
            text="Embedding модель",
            bg="#f3f3f3",
            fg="#666666",
            font=("Segoe UI", 9)
        ).pack(
            anchor="w"
        )

        self.embedding_label = tk.Label(
            embedding_frame,
            text="Проверка...",
            bg="#f3f3f3",
            fg="#111111",
            font=("Segoe UI", 10, "bold")
        )

        self.embedding_label.pack(
            anchor="w"
        )

        # --------------------------
        # Refresh
        # --------------------------

        refresh_button = ttk.Button(
            self,
            text="Обновить",
            command=self.on_refresh
        )

        refresh_button.pack(
            side="right",
            padx=15
        )

    def update(
        self,
        models: LoadedModels
    ):

        self.llm_label.config(
            text=models.llm or "Не загружена"
        )

        self.embedding_label.config(
            text=models.embedding or "Не загружена"
        )

    def set_error(self):

        self.llm_label.config(
            text="Нет соединения"
        )

        self.embedding_label.config(
            text="Нет соединения"
        )