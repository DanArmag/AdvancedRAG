import tkinter as tk


class ChatView(tk.Frame):

    def __init__(self, parent):
        super().__init__(
            parent,
            bg="white"
        )

        self._create_widgets()
        self._configure_tags()

    def _create_widgets(self):

        self.text = tk.Text(
            self,
            wrap="word",
            font=("Segoe UI", 11),
            bg="white",
            fg="#111111",
            relief="flat",
            borderwidth=0,
            padx=18,
            pady=15,
            state="disabled"
        )

        self.text.pack(
            side="left",
            fill="both",
            expand=True
        )

        scrollbar = tk.Scrollbar(
            self,
            command=self.text.yview
        )

        scrollbar.pack(
            side="right",
            fill="y"
        )

        self.text.configure(
            yscrollcommand=scrollbar.set
        )

    def _configure_tags(self):

        self.text.tag_configure(
            "user",
            foreground="#1565c0",
            font=("Segoe UI", 11, "bold")
        )

        self.text.tag_configure(
            "assistant",
            foreground="#222222",
            font=("Segoe UI", 11)
        )

        self.text.tag_configure(
            "system",
            foreground="#777777",
            font=("Segoe UI", 9, "italic")
        )

    def append(
        self,
        text: str,
        tag: str | None = None
    ):

        self.text.configure(
            state="normal"
        )

        if tag:
            self.text.insert(
                "end",
                text,
                tag
            )
        else:
            self.text.insert(
                "end",
                text
            )

        self.text.configure(
            state="disabled"
        )

        self.text.see("end")

    def clear(self):

        self.text.configure(
            state="normal"
        )

        self.text.delete(
            "1.0",
            "end"
        )

        self.text.configure(
            state="disabled"
        )