"""Memory consolidation — moves information between memory layers.

Consolidation is the process of extracting durable knowledge from the
working memory stream and persisting it into the appropriate long-term
layer. This mirrors how human brains consolidate short-term experiences
into long-term memory during sleep / idle periods.

Three consolidation paths:
  working → episodic   (conversation summaries)
  working → semantic   (extracted facts / preferences)
  working → procedural (learned action patterns)
"""

from __future__ import annotations

import json

from openai import OpenAI

from memory_agent.config import MemoryConfig
from memory_agent.memory.episodic import EpisodicMemory
from memory_agent.memory.procedural import ProceduralMemory
from memory_agent.memory.semantic import SemanticMemory
from memory_agent.memory.working import Message

_CONSOLIDATION_PROMPT = """\
You are a memory consolidation module. Analyse the following conversation
segment and extract structured information for long-term storage.

Return a JSON object with three keys:
{
  "episode": {
    "summary": "<one-sentence summary of the conversation>",
    "importance": <float 0-1>
  },
  "facts": [
    {"subject": "...", "predicate": "...", "object": "...", "confidence": <float 0-1>}
  ],
  "procedures": [
    {
      "name": "...",
      "description": "...",
      "steps": ["step1", "step2", ...],
      "trigger": "..."
    }
  ]
}

Rules:
- Only extract facts that are explicitly stated or strongly implied.
- Facts should be about the user, their preferences, or important entities.
- Only extract procedures if the conversation describes a repeatable process.
- If nothing fits a category, use an empty list / null.
- Return ONLY the JSON, no markdown fences or explanation.
"""


class MemoryConsolidator:
    """Extracts durable knowledge from working memory into long-term layers."""

    def __init__(
        self,
        config: MemoryConfig,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        procedural: ProceduralMemory,
    ) -> None:
        self._config = config
        self._client = OpenAI(api_key=config.openai_api_key)
        self._episodic = episodic
        self._semantic = semantic
        self._procedural = procedural

    def consolidate(self, messages: list[Message]) -> dict:
        """Consolidate a batch of evicted messages into long-term memory.

        Returns a summary dict of what was stored.
        """
        if not messages:
            return {"episodes": 0, "facts": 0, "procedures": 0}

        conversation_text = "\n".join(
            f"{m.role}: {m.content}" for m in messages
        )

        response = self._client.chat.completions.create(
            model=self._config.chat_model,
            messages=[
                {"role": "system", "content": _CONSOLIDATION_PROMPT},
                {"role": "user", "content": conversation_text},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {"episodes": 0, "facts": 0, "procedures": 0, "error": "parse_failed"}

        stored = {"episodes": 0, "facts": 0, "procedures": 0}

        # Store episode
        episode = data.get("episode")
        if episode and episode.get("summary"):
            self._episodic.store_episode(
                summary=episode["summary"],
                content=conversation_text,
                importance=episode.get("importance", 0.5),
            )
            stored["episodes"] = 1

        # Store facts
        for fact in data.get("facts", []):
            if fact.get("subject") and fact.get("predicate") and fact.get("object"):
                self._semantic.store_fact(
                    subject=fact["subject"],
                    predicate=fact["predicate"],
                    obj=fact["object"],
                    confidence=fact.get("confidence", 1.0),
                    source="consolidation",
                )
                stored["facts"] += 1

        # Store procedures
        for proc in data.get("procedures", []):
            if proc.get("name") and proc.get("steps"):
                self._procedural.store_procedure(
                    name=proc["name"],
                    description=proc.get("description", ""),
                    steps=proc["steps"],
                    trigger_condition=proc.get("trigger"),
                )
                stored["procedures"] += 1

        return stored
