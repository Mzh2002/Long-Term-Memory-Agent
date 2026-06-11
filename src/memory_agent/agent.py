"""The Memory Agent — orchestrates the 4-layer memory system.

This is the main entry point for interacting with the agent. It:
1. Accepts user input
2. Retrieves relevant context from all long-term memory layers
3. Injects that context into the working memory / system prompt
4. Calls the LLM for a response
5. Consolidates evicted working-memory messages into long-term storage
"""

from __future__ import annotations

from openai import OpenAI

from memory_agent.config import MemoryConfig
from memory_agent.consolidation import MemoryConsolidator
from memory_agent.embeddings import EmbeddingEngine
from memory_agent.memory.episodic import EpisodicMemory
from memory_agent.memory.procedural import ProceduralMemory
from memory_agent.memory.semantic import SemanticMemory
from memory_agent.memory.working import WorkingMemory
from memory_agent.storage.sqlite_store import SQLiteStore

_BASE_SYSTEM_PROMPT = """\
You are an AI assistant with a 4-layer long-term memory system:

1. **Working Memory**: Your current conversation context (what you see now).
2. **Episodic Memory**: Summaries of past conversations you can recall.
3. **Semantic Memory**: Facts and knowledge you've learned about the user and the world.
4. **Procedural Memory**: Learned procedures and best practices.

When relevant context is retrieved from your long-term memory, it appears
in brackets below. Use this information naturally — reference past
conversations, apply known facts, and follow learned procedures when
appropriate. If you learn new facts or preferences from the user,
acknowledge them so they can be stored for future sessions.
"""


class MemoryAgent:
    """AI agent with a 4-layer long-term memory system."""

    def __init__(self, config: MemoryConfig | None = None) -> None:
        self._config = config or MemoryConfig.from_env()
        self._client = OpenAI(api_key=self._config.openai_api_key)

        # Storage
        self._store = SQLiteStore(self._config.db_path)

        # Embeddings
        self._embeddings = EmbeddingEngine(self._config)

        # Memory layers
        self.working = WorkingMemory(self._config)
        self.episodic = EpisodicMemory(self._config, self._store, self._embeddings)
        self.semantic = SemanticMemory(self._config, self._store, self._embeddings)
        self.procedural = ProceduralMemory(self._config, self._store, self._embeddings)

        # Consolidation
        self._consolidator = MemoryConsolidator(
            self._config, self.episodic, self.semantic, self.procedural
        )

        # Set base system prompt
        self.working.system_prompt = _BASE_SYSTEM_PROMPT

    def chat(self, user_input: str) -> str:
        """Process a user message and return the agent's response.

        This is the main interaction loop:
        1. Retrieve relevant long-term memories
        2. Build augmented context
        3. Get LLM response
        4. Consolidate any evicted messages
        """
        # Retrieve from long-term memory
        memory_context = self._retrieve_memories(user_input)

        # Build augmented system prompt
        augmented_prompt = _BASE_SYSTEM_PROMPT
        if memory_context:
            augmented_prompt += "\n\n" + memory_context
        self.working.system_prompt = augmented_prompt

        # Add user message (may evict old messages)
        evicted = self.working.add_message("user", user_input)

        # Consolidate evicted messages into long-term memory
        if evicted:
            self._consolidator.consolidate(evicted)

        # Get LLM response
        context = self.working.get_context_window()
        response = self._client.chat.completions.create(
            model=self._config.chat_model,
            messages=context,
            temperature=0.7,
        )
        assistant_reply = response.choices[0].message.content or ""

        # Store assistant response in working memory
        self.working.add_message("assistant", assistant_reply)

        return assistant_reply

    def _retrieve_memories(self, query: str) -> str:
        """Retrieve relevant context from all long-term memory layers."""
        sections: list[str] = []

        episodes = self.episodic.recall(query)
        if episodes:
            sections.append(self.episodic.format_for_context(episodes))

        facts = self.semantic.recall(query)
        if facts:
            sections.append(self.semantic.format_for_context(facts))

        procedures = self.procedural.recall(query)
        if procedures:
            sections.append(self.procedural.format_for_context(procedures))

        return "\n\n".join(sections)

    def save_fact(self, subject: str, predicate: str, obj: str, confidence: float = 1.0) -> int:
        """Manually store a fact in semantic memory."""
        return self.semantic.store_fact(subject, predicate, obj, confidence)

    def save_procedure(
        self, name: str, description: str, steps: list[str], trigger: str | None = None
    ) -> int:
        """Manually store a procedure in procedural memory."""
        return self.procedural.store_procedure(name, description, steps, trigger)

    def end_session(self) -> dict:
        """End the current session, consolidating all remaining working memory."""
        remaining = self.working.clear()
        result = self._consolidator.consolidate(remaining)
        return result

    def get_memory_stats(self) -> dict:
        """Return statistics about the current memory state."""
        return {
            "working_memory": {
                "messages": len(self.working.messages),
                "tokens": self.working.token_count(),
            },
            "episodic_memory": {
                "total_episodes": len(self._store.get_all_episodes()),
            },
            "semantic_memory": {
                "total_facts": len(self._store.get_all_facts()),
            },
            "procedural_memory": {
                "total_procedures": len(self._store.get_all_procedures()),
            },
        }

    def close(self) -> None:
        """Clean up resources."""
        self._store.close()
