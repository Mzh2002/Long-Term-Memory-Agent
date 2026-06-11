"""Layer 4: Procedural Memory — learned behaviours and action patterns.

Procedural memory stores reusable procedures the agent has learned —
multi-step action sequences, best practices, and behavioural rules.
Each procedure has a trigger condition and a sequence of steps.
This gives the agent *know-how* — it remembers "how to do things"
and can improve over time based on success/failure feedback.
"""

from __future__ import annotations

from typing import Any

from memory_agent.config import MemoryConfig
from memory_agent.embeddings import EmbeddingEngine
from memory_agent.storage.sqlite_store import SQLiteStore


class ProceduralMemory:
    """Manages the agent's learned procedures and action patterns."""

    def __init__(
        self, config: MemoryConfig, store: SQLiteStore, embeddings: EmbeddingEngine
    ) -> None:
        self._config = config
        self._store = store
        self._embeddings = embeddings

    def store_procedure(
        self,
        name: str,
        description: str,
        steps: list[str],
        trigger_condition: str | None = None,
    ) -> int:
        """Store or update a procedure.

        Args:
            name: Unique name for the procedure.
            description: What this procedure does.
            steps: Ordered list of action steps.
            trigger_condition: When this procedure should be triggered.

        Returns:
            The procedure's database ID.
        """
        embed_text = f"{name}: {description}. Trigger: {trigger_condition or 'manual'}"
        embedding = self._embeddings.embed(embed_text)
        return self._store.add_procedure(
            name=name,
            description=description,
            steps=steps,
            trigger_condition=trigger_condition,
            embedding=embedding,
        )

    def recall(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Retrieve procedures most relevant to the query."""
        top_k = top_k or self._config.max_procedures_retrieved
        query_embedding = self._embeddings.embed(query)
        all_procedures = self._store.get_all_procedures()

        results = self._embeddings.find_similar(
            query_embedding=query_embedding,
            candidates=all_procedures,
            threshold=0.6,
            top_k=top_k,
        )

        return [
            {
                "id": proc["id"],
                "name": proc["name"],
                "description": proc["description"],
                "steps": proc["steps"],
                "trigger_condition": proc["trigger_condition"],
                "relevance_score": round(score, 4),
            }
            for proc, score in results
        ]

    def record_outcome(self, name: str, success: bool) -> None:
        """Record whether a procedure execution was successful."""
        self._store.record_procedure_outcome(name, success)

    def format_for_context(self, procedures: list[dict[str, Any]]) -> str:
        """Format recalled procedures as context text for the LLM."""
        if not procedures:
            return ""
        lines = ["[Relevant Procedures]"]
        for proc in procedures:
            steps_str = " → ".join(proc["steps"])
            lines.append(f"- {proc['name']}: {proc['description']}")
            lines.append(f"  Steps: {steps_str}")
        return "\n".join(lines)
