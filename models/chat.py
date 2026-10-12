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

    def get_messages(self) -> List[dict]:
        return [
            {
                "role": message.role,
                "content": message.content
            }
            for message in self._messages
        ]

    @property
    def messages(self) -> List[Message]:
        return list(self._messages)