"""Layer 3: Semantic Memory — structured knowledge and facts.

Semantic memory stores factual knowledge about the world and the user
as subject-predicate-object triples (knowledge graph style). Facts are
extracted from conversations and persisted with embeddings for retrieval.
This gives the agent durable *knowledge* — it knows "what is true"
independent of any single conversation.
"""

from __future__ import annotations

from typing import Any

from memory_agent.config import MemoryConfig
from memory_agent.embeddings import EmbeddingEngine
from memory_agent.storage.sqlite_store import SQLiteStore


class SemanticMemory:
    """Manages the agent's factual knowledge base."""

    def __init__(
        self, config: MemoryConfig, store: SQLiteStore, embeddings: EmbeddingEngine
    ) -> None:
        self._config = config
        self._store = store
        self._embeddings = embeddings

    def store_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        source: str | None = None,
    ) -> int:
        """Store or update a knowledge triple.

        Args:
            subject: Entity the fact is about (e.g. "User").
            predicate: Relationship (e.g. "prefers").
            obj: Object of the relation (e.g. "Python").
            confidence: How certain we are about this fact (0-1).
            source: Where this fact was learned from.

        Returns:
            The fact's database ID.
        """
        fact_text = f"{subject} {predicate} {obj}"
        embedding = self._embeddings.embed(fact_text)
        return self._store.add_fact(
            subject=subject,
            predicate=predicate,
            obj=obj,
            confidence=confidence,
            source=source,
            embedding=embedding,
        )

    def recall(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Retrieve facts most relevant to the query."""
        top_k = top_k or self._config.max_facts_retrieved
        query_embedding = self._embeddings.embed(query)
        all_facts = self._store.get_all_facts()

        results = self._embeddings.find_similar(
            query_embedding=query_embedding,
            candidates=all_facts,
            threshold=self._config.fact_similarity_threshold,
            top_k=top_k,
        )

        return [
            {
                "id": fact["id"],
                "subject": fact["subject"],
                "predicate": fact["predicate"],
                "object": fact["object"],
                "confidence": fact["confidence"],
                "relevance_score": round(score, 4),
            }
            for fact, score in results
        ]

    def lookup(self, subject: str) -> list[dict[str, Any]]:
        """Direct lookup of facts by subject name."""
        return self._store.get_facts_by_subject(subject)

    def format_for_context(self, facts: list[dict[str, Any]]) -> str:
        """Format recalled facts as context text for the LLM."""
        if not facts:
            return ""
        lines = ["[Known Facts]"]
        for f in facts:
            lines.append(
                f"- {f['subject']} {f['predicate']} {f['object']} "
                f"(confidence={f['confidence']})"
            )
        return "\n".join(lines)
