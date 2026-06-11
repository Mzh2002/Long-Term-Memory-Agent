"""Configuration for the memory agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MemoryConfig:
    """Configuration for the 4-layer memory system."""

    # OpenAI settings
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Working memory settings
    max_context_messages: int = 20
    max_context_tokens: int = 4000

    # Episodic memory settings
    max_episodes_retrieved: int = 5
    episode_similarity_threshold: float = 0.7

    # Semantic memory settings
    max_facts_retrieved: int = 10
    fact_similarity_threshold: float = 0.75

    # Procedural memory settings
    max_procedures_retrieved: int = 3

    # Storage settings
    storage_dir: Path = field(default_factory=lambda: Path.home() / ".memory_agent")
    db_name: str = "memory.db"

    @property
    def db_path(self) -> Path:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        return self.storage_dir / self.db_name

    @classmethod
    def from_env(cls) -> MemoryConfig:
        """Create config from environment variables."""
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            chat_model=os.getenv("MEMORY_AGENT_CHAT_MODEL", "gpt-4o-mini"),
            embedding_model=os.getenv("MEMORY_AGENT_EMBEDDING_MODEL", "text-embedding-3-small"),
            max_context_messages=int(os.getenv("MEMORY_AGENT_MAX_CONTEXT", "20")),
            max_context_tokens=int(os.getenv("MEMORY_AGENT_MAX_TOKENS", "4000")),
        )
