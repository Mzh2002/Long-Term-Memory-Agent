"""Layer 2: Episodic Memory — records of past interactions and events.

Episodic memory stores summaries of past conversations and significant
events. Each episode is timestamped, summarised, and embedded for
semantic retrieval. When the working memory overflows, evicted messages
are consolidated here. This gives the agent a sense of *personal history*
— it can recall "what happened" in previous sessions.
"""

from __future__ import annotations

from typing import Any

from memory_agent.config import MemoryConfig
from memory_agent.embeddings import EmbeddingEngine
from memory_agent.storage.sqlite_store import SQLiteStore


class EpisodicMemory:
    """Manages the agent's long-term record of past interactions."""

    def __init__(
        self, config: MemoryConfig, store: SQLiteStore, embeddings: EmbeddingEngine
    ) -> None:
        self._config = config
        self._store = store
        self._embeddings = embeddings

    def store_episode(self, summary: str, content: str, importance: float = 0.5) -> int:
        """Persist a new episode with its embedding.

        Args:
            summary: Short description of the episode.
            content: Full text content of the episode.
            importance: Float 0-1 indicating how significant this episode is.

        Returns:
            The episode's database ID.
        """
        embedding = self._embeddings.embed(summary + " " + content[:500])
        return self._store.add_episode(
            summary=summary,
            content=content,
            embedding=embedding,
            importance=importance,
        )

    def recall(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Retrieve episodes most relevant to the query.

        Uses cosine similarity over embeddings, filtered by the
        configured threshold.
        """
        top_k = top_k or self._config.max_episodes_retrieved
        query_embedding = self._embeddings.embed(query)
        all_episodes = self._store.get_all_episodes()

        results = self._embeddings.find_similar(
            query_embedding=query_embedding,
            candidates=all_episodes,
            threshold=self._config.episode_similarity_threshold,
            top_k=top_k,
        )

        recalled = []
        for episode, score in results:
            self._store.update_episode_access(episode["id"])
            recalled.append({
                "id": episode["id"],
                "timestamp": episode["timestamp"],
                "summary": episode["summary"],
                "content": episode["content"],
                "importance": episode["importance"],
                "relevance_score": round(score, 4),
            })
        return recalled

    def format_for_context(self, episodes: list[dict[str, Any]]) -> str:
        """Format recalled episodes as context text for the LLM."""
        if not episodes:
            return ""
        lines = ["[Recalled Episodes]"]
        for ep in episodes:
            lines.append(
                f"- [{ep['timestamp'][:10]}] (relevance={ep['relevance_score']}) "
                f"{ep['summary']}"
            )
        return "\n".join(lines)
