import json
import os
from pathlib import Path
from typing import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

"""
`__file__` 是当前文件路径，`.resolve()` 转绝对路径，`parents[0]` 是`core` 、`[1]` 是`internal` 、
`[2]` 是`llmops-api` 项目根。不用`"./storage"` 是因为那会跟着启动目录变
"""
MEMORY_STORAGE_DIR = Path(__file__).resolve().parents[2] / "storage" / "memory"

class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, storage_dir, session_id, window_rounds=3) -> None:
        self.storage_dir = Path(storage_dir)
        self.session_id = session_id
        # 一轮 = human + ai = 2 条，所以`k=3` 换成`3 * 2 = 6` 条
        self.window_rounds = window_rounds
        # `mkdir(parents=True, exist_ok=True)` ：目录不存在就建，存在也不报错
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        # 每个会话一个`<session_id>.json`
        self.file_path = self.storage_dir / f"{self.session_id}.json"

    @property
    def messages(self) -> list[BaseMessage]:
        if not self.file_path.exists():
            return []

        try:
            raw = json.loads(self.file_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

        return messages_from_dict(raw)

    def clear(self) -> None:
        if self.file_path.exists():
            self.file_path.unlink()

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        all_messages = list(self.messages) + list(messages)

        if self.window_rounds > 0:
            max_messages = self.window_rounds * 2
            # 保留最后 max_messages 个，超出自动丢弃最早的
            all_messages = all_messages[-max_messages:]

        self.file_path.write_text(
            # `ensure_ascii=False` 让中文直接可读
            json.dumps([message_to_dict(m) for m in all_messages], ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

def get_file_chat_message_history(session_id: str) -> FileChatMessageHistory:
    return FileChatMessageHistory(MEMORY_STORAGE_DIR, session_id)