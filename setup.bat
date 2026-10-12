@echo off
setlocal EnableExtensions
cd /d "%~dp0"

title LM Studio Chat - Setup

echo.
echo ========================================
echo       LM Studio Chat - Project Setup
echo ========================================
echo.
echo Creating project...
echo.

set "PS_FILE=%TEMP%\lmstudio_setup_%RANDOM%.ps1"

(
    echo $ErrorActionPreference = "Stop"
    echo Set-Location -LiteralPath "%~dp0"
    echo.
    echo New-Item -ItemType Directory -Force -Path "api" ^| Out-Null
    echo New-Item -ItemType Directory -Force -Path "models" ^| Out-Null
    echo New-Item -ItemType Directory -Force -Path "ui" ^| Out-Null
    echo.
    echo $files = @{}
    echo.
    echo $files["main.py"] = @'
import tkinter as tk

from ui.app import ChatApp


def main():
    root = tk.Tk()
    ChatApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
'@
    echo.
    echo $files["config.py"] = @'
from dataclasses import dataclass


@dataclass(frozen=True)
class LMStudioConfig:
    host: str = "127.0.0.1"
    port: int = 1234

    @property
    def base_url(self) -^> str:
        return f"http://{self.host}:{self.port}"

    @property
    def models_url(self) -^> str:
        return f"{self.base_url}/api/v0/models"

    @property
    def chat_url(self) -^> str:
        return f"{self.base_url}/v1/chat/completions"


@dataclass(frozen=True)
class AppConfig:
    lm_studio: LMStudioConfig = LMStudioConfig()


CONFIG = AppConfig()
'@
    echo.
    echo $files["api/__init__.py"] = ""
    echo.
    echo $files["api/lm_studio.py"] = @'
import json
from typing import Callable, List

from urllib.request import Request, urlopen

from config import LMStudioConfig
from models.chat import LoadedModels


class LMStudioError(Exception):
    pass


class LMStudioClient:

    def __init__(self, config: LMStudioConfig):
        self.config = config

    def get_loaded_models(self) -^> LoadedModels:

        request = Request(
            self.config.models_url,
            method="GET",
            headers={
                "Content-Type": "application/json"
            }
        )

        try:
            with urlopen(request, timeout=3) as response:
                data = json.loads(
                    response.read().decode("utf-8")
                )

        except Exception as exc:
            raise LMStudioError(
                f"Не удалось подключиться к LM Studio: {exc}"
            ) from exc

        llm = None
        embedding = None

        for model in data.get("data", []):

            if model.get("state") != "loaded":
                continue

            model_type = model.get("type")
            model_id = model.get("id")

            if model_type == "llm":
                llm = model_id

            elif model_type in ("embedding", "embeddings"):
                embedding = model_id

        return LoadedModels(
            llm=llm,
            embedding=embedding
        )

    def stream_chat(
        self,
        model: str,
        messages: List[dict],
        on_token: Callable[[str], None],
        on_finished: Callable[[], None],
        on_error: Callable[[str], None],
    ):

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": 0.7,
        }

        body = json.dumps(payload).encode("utf-8")

        request = Request(
            self.config.chat_url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            }
        )

        try:

            with urlopen(request, timeout=None) as response:

                while True:

                    line = response.readline()

                    if not line:
                        break

                    line = line.decode("utf-8").strip()

                    if not line:
                        continue

                    if not line.startswith("data:"):
                        continue

                    data = line[5:].strip()

                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices", [])

                    if not choices:
                        continue

                    delta = choices[0].get(
                        "delta",
                        {}
                    )

                    text = delta.get("content")

                    if text:
                        on_token(text)

            on_finished()

        except Exception as exc:
            on_error(str(exc))
'@
    echo.
    echo $files["models/__init__.py"] = ""
    echo.
    echo $files["models/chat.py"] = @'
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Message:
    role: str
    content: str


@dataclass
class LoadedModels:
    llm: Optional[str] = None
    embedding: Optional[str] = None


class ChatHistory:

    def __init__(self):
        self._messages: List[Message] = []

    def add_user_message(self, content: str):
        self._messages.append(
            Message(
                role="user",
                content=content
            )
        )

    def add_assistant_message(self, content: str):
        self._messages.append(
            Message(
                role="assistant",
                content=content
            )
        )

    def clear(self):
        self._messages.clear()

    def get_messages(self) -^> List[dict]:
        return [
            {
                "role": message.role,
                "content": message.content
            }
            for message in self._messages
        ]

    @property
    def messages(self) -^> List[Message]:
        return list(self._messages)
'@
    echo.
    echo $files["ui/__init__.py"] = ""
    echo.
    echo $files["ui/chat_view.py"] = @'
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

    def append(self, text: str, tag=None):

        self.text.configure(state="normal")

        if tag:
            self.text.insert("end", text, tag)
        else:
            self.text.insert("end", text)

        self.text.configure(state="disabled")
        self.text.see("end")

    def clear(self):

        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")
'@
    echo.
    echo $files["ui/input_bar.py"] = @'
import tkinter as tk
from tkinter import ttk


class InputBar(tk.Frame):

    def __init__(self, parent, on_send):

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

    def get_text(self) -^> str:

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

        self.input_text.config(state=state)
        self.send_button.config(state=state)

    def set_generating(self, generating: bool):

        self.send_button.config(
            text="Генерация..." if generating
            else "Отправить"
        )
'@
    echo.
    echo $files["ui/model_info.py"] = @'
import tkinter as tk
from tkinter import ttk

from models.chat import LoadedModels


class ModelInfo(tk.Frame):

    def __init__(self, parent, on_refresh):

        super().__init__(
            parent,
            bg="#f3f3f3",
            height=70
        )

        self.on_refresh = on_refresh

        self.pack_propagate(False)

        self._create_widgets()

    def _create_widgets(self):

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
        ).pack(anchor="w")

        self.llm_label = tk.Label(
            llm_frame,
            text="Проверка...",
            bg="#f3f3f3",
            fg="#111111",
            font=("Segoe UI", 10, "bold")
        )

        self.llm_label.pack(anchor="w")

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
        ).pack(anchor="w")

        self.embedding_label = tk.Label(
            embedding_frame,
            text="Проверка...",
            bg="#f3f3f3",
            fg="#111111",
            font=("Segoe UI", 10, "bold")
        )

        self.embedding_label.pack(anchor="w")

        ttk.Button(
            self,
            text="Обновить",
            command=self.on_refresh
        ).pack(
            side="right",
            padx=15
        )

    def update(self, models: LoadedModels):

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
'@
    echo.
    echo $files["ui/app.py"] = @'
import threading
import tkinter as tk
from tkinter import messagebox

from api.lm_studio import (
    LMStudioClient,
    LMStudioError
)

from config import CONFIG
from models.chat import ChatHistory

from ui.chat_view import ChatView
from ui.input_bar import InputBar
from ui.model_info import ModelInfo


class ChatApp:

    def __init__(self, root: tk.Tk):

        self.root = root

        self.root.title("LM Studio Chat")
        self.root.geometry("1000x720")
        self.root.minsize(700, 500)

        self.client = LMStudioClient(
            CONFIG.lm_studio
        )

        self.history = ChatHistory()

        self.chat_model = None
        self.generating = False

        self._create_ui()
        self.refresh_models()

    def _create_ui(self):

        self._create_header()

        self.model_info = ModelInfo(
            self.root,
            on_refresh=self.refresh_models
        )

        self.model_info.pack(
            side="top",
            fill="x"
        )

        self.chat_view = ChatView(
            self.root
        )

        self.chat_view.pack(
            fill="both",
            expand=True
        )

        self.input_bar = InputBar(
            self.root,
            on_send=self.send_message
        )

        self.input_bar.pack(
            side="bottom",
            fill="x"
        )

        self.root.bind(
            "<Control-Return>",
            lambda event: self.send_message()
        )

    def _create_header(self):

        header = tk.Frame(
            self.root,
            bg="#202124",
            height=75
        )

        header.pack(
            side="top",
            fill="x"
        )

        header.pack_propagate(False)

        tk.Label(
            header,
            text="LM Studio Chat",
            bg="#202124",
            fg="white",
            font=("Segoe UI", 16, "bold")
        ).pack(
            side="left",
            padx=18
        )

        self.status_label = tk.Label(
            header,
            text="Подключение...",
            bg="#202124",
            fg="#aaaaaa",
            font=("Segoe UI", 9)
        )

        self.status_label.pack(
            side="right",
            padx=18
        )

    def refresh_models(self):

        self.status_label.config(
            text="Проверка LM Studio..."
        )

        threading.Thread(
            target=self._refresh_models_worker,
            daemon=True
        ).start()

    def _refresh_models_worker(self):

        try:

            models = self.client.get_loaded_models()

            self.root.after(
                0,
                lambda: self._models_loaded(models)
            )

        except LMStudioError as exc:

            self.root.after(
                0,
                lambda: self._models_error(str(exc))
            )

    def _models_loaded(self, models):

        self.model_info.update(models)

        self.chat_model = models.llm

        self.status_label.config(
            text="LM Studio подключён"
        )

        if not self.chat_model:

            self.chat_view.append(
                "⚠ Модель генерации не загружена.\\n"
                "Загрузи LLM в LM Studio.\\n\\n",
                "system"
            )

    def _models_error(self, error):

        self.model_info.set_error()

        self.status_label.config(
            text="LM Studio недоступен"
        )

        self.chat_view.append(
            "⚠ Не удалось подключиться к LM Studio.\\n"
            f"{error}\\n\\n",
            "system"
        )

    def send_message(self):

        if self.generating:
            return

        if not self.chat_model:

            messagebox.showwarning(
                "Нет модели",
                "В LM Studio не найдена "
                "загруженная модель генерации."
            )

            return

        text = self.input_bar.get_text()

        if not text:
            return

        self.input_bar.clear()

        self.chat_view.append(
            "Вы:\\n",
            "user"
        )

        self.chat_view.append(
            text + "\\n\\n"
        )

        self.history.add_user_message(text)

        self.chat_view.append(
            "Модель:\\n",
            "user"
        )

        self.generating = True

        self.input_bar.set_enabled(False)
        self.input_bar.set_generating(True)

        threading.Thread(
            target=self._generation_worker,
            daemon=True
        ).start()

    def _generation_worker(self):

        response_parts = []

        def on_token(token: str):

            response_parts.append(token)

            self.root.after(
                0,
                lambda t=token:
                    self.chat_view.append(t)
            )

        def on_finished():

            response = "".join(response_parts)

            self.history.add_assistant_message(
                response
            )

            self.root.after(
                0,
                self._generation_finished
            )

        def on_error(error: str):

            self.root.after(
                0,
                lambda:
                    self._generation_error(error)
            )

        self.client.stream_chat(
            model=self.chat_model,
            messages=self.history.get_messages(),
            on_token=on_token,
            on_finished=on_finished,
            on_error=on_error
        )

    def _generation_finished(self):

        self.chat_view.append("\\n\\n")

        self.generating = False

        self.input_bar.set_enabled(True)
        self.input_bar.set_generating(False)

    def _generation_error(self, error):

        self.chat_view.append(
            "\\n\\n"
            "⚠ Ошибка генерации:\\n"
            f"{error}\\n\\n",
            "system"
        )

        self.generating = False

        self.input_bar.set_enabled(True)
        self.input_bar.set_generating(False)
'@
    echo.
    echo foreach ($item in $files.GetEnumerator()) {
    echo     $path = Join-Path (Get-Location) $item.Key
    echo     $parent = Split-Path $path -Parent
    echo.
    echo     if (!(Test-Path $parent)) {
    echo         New-Item -ItemType Directory -Force -Path $parent ^| Out-Null
    echo     }
    echo.
    echo     [System.IO.File]::WriteAllText(
    echo         $path,
    echo         $item.Value,
    echo         [System.Text.UTF8Encoding]::new($false)
    echo     )
    echo }
    echo.
    echo Write-Host ""
    echo Write-Host "Project created successfully."
    echo Write-Host ""
) > "%PS_FILE%"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%PS_FILE%"
set "RESULT=%ERRORLEVEL%"

del "%PS_FILE%" >nul 2>&1

if not "%RESULT%"=="0" (
    echo.
    echo ========================================
    echo ERROR: Project creation failed.
    echo ========================================
    echo.
    pause
    exit /b %RESULT%
)

echo.
echo ========================================
echo        PROJECT CREATED SUCCESSFULLY
echo ========================================
echo.
echo Files:
echo.
echo   main.py
echo   config.py
echo.
echo   api\
echo       __init__.py
echo       lm_studio.py
echo.
echo   models\
echo       __init__.py
echo       chat.py
echo.
echo   ui\
echo       __init__.py
echo       app.py
echo       chat_view.py
echo       input_bar.py
echo       model_info.py
echo.
echo ========================================
echo.
echo To start:
echo.
echo   python main.py
echo.
pause

endlocal