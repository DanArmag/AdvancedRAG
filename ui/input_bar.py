import tkinter as tk
from tkinter import ttk


class InputBar(tk.Frame):

    def __init__(
        self,
        parent,
        on_send
    ):
        super().__init__(
            parent,
            bg="#f3f3f3",
            height=100
        )

        self.on_send = on_send

        self.pack_propagate(False)

        self._create_widgets()

    def _create_widgets(self):

        self.input_text = tk.Text(
            self,
            height=3,
            font=("Segoe UI", 11),
            wrap="word",
            relief="solid",
            borderwidth=1
        )

        self.input_text.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(15, 8),
            pady=12
        )

        self.input_text.bind(
            "<Return>",
            self._on_enter
        )

        self.send_button = ttk.Button(
            self,
            text="Отправить",
            command=self.on_send
        )

        self.send_button.pack(
            side="right",
            padx=(5, 15),
            pady=12
        )

    def _on_enter(self, event):

        self.on_send()

        return "break"

    def get_text(self) -> str:

        return self.input_text.get(
            "1.0",
            "end-1c"
        ).strip()

    def clear(self):

        self.input_text.delete(
            "1.0",
            "end"
        )

    def set_enabled(self, enabled: bool):

        state = "normal" if enabled else "disabled"

        self.input_text.config(
            state=state
        )

        self.send_button.config(
            state=state
        )

    def set_generating(self, generating: bool):

        if generating:
            self.send_button.config(
                text="Генерация..."
            )
        else:
            self.send_button.config(
                text="Отправить"
            )