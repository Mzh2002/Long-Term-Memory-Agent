"""Layer 1: Working Memory — short-term conversational context.

Working memory holds the active conversation context: recent messages,
current task state, and any in-flight instructions. It operates entirely
in-memory and acts as the agent's "scratchpad" during a session.
Analogous to a human's short-term / working memory or the model's
context window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import tiktoken

from memory_agent.config import MemoryConfig


@dataclass
class Message:
    """A single message in the conversation."""

    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class WorkingMemory:
    """Manages the agent's short-term conversational context.

    Keeps a sliding window of recent messages plus a persistent system
    prompt. When the window exceeds the configured token or message limit,
    older messages are evicted (and optionally summarized into episodic memory).
    """

    def __init__(self, config: MemoryConfig) -> None:
        self._config = config
        self._system_prompt: str = ""
        self._messages: list[Message] = []
        self._scratchpad: dict[str, Any] = {}
        self._encoder = tiktoken.encoding_for_model(config.chat_model)

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        self._system_prompt = value

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    def add_message(self, role: str, content: str, **metadata: Any) -> list[Message] | None:
        """Add a message and return any evicted messages (for consolidation)."""
        self._messages.append(Message(role=role, content=content, metadata=metadata))
        return self._trim()

    def get_context_window(self) -> list[dict[str, str]]:
        """Return the current context as OpenAI-compatible message dicts."""
        context: list[dict[str, str]] = []
        if self._system_prompt:
            context.append({"role": "system", "content": self._system_prompt})
        for msg in self._messages:
            context.append({"role": msg.role, "content": msg.content})
        return context

    def set_scratchpad(self, key: str, value: Any) -> None:
        """Store transient task state in the scratchpad."""
        self._scratchpad[key] = value

    def get_scratchpad(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from the scratchpad."""
        return self._scratchpad.get(key, default)

    def clear_scratchpad(self) -> None:
        self._scratchpad.clear()

    def token_count(self) -> int:
        """Estimate total tokens in the current context window."""
        total = 0
        if self._system_prompt:
            total += len(self._encoder.encode(self._system_prompt))
        for msg in self._messages:
            total += len(self._encoder.encode(msg.content)) + 4  # role overhead
        return total

    def _trim(self) -> list[Message] | None:
        """Evict oldest messages if context exceeds limits.

        Returns evicted messages so they can be consolidated into
        episodic memory.
        """
        evicted: list[Message] = []

        # Trim by message count
        while len(self._messages) > self._config.max_context_messages:
            evicted.append(self._messages.pop(0))

        # Trim by token count
        while self.token_count() > self._config.max_context_tokens and len(self._messages) > 2:
            evicted.append(self._messages.pop(0))

        return evicted if evicted else None

    def clear(self) -> list[Message]:
        """Clear all messages, returning them for potential consolidation."""
        evicted = list(self._messages)
        self._messages.clear()
        self._scratchpad.clear()
        return evicted
