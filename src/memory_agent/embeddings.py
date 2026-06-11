"""Embedding utilities for vector similarity search across memory layers."""

from __future__ import annotations

import numpy as np
from openai import OpenAI

from memory_agent.config import MemoryConfig


class EmbeddingEngine:
    """Generates and compares text embeddings using OpenAI's embedding API."""

    def __init__(self, config: MemoryConfig) -> None:
        self._client = OpenAI(api_key=config.openai_api_key)
        self._model = config.embedding_model

    def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text."""
        response = self._client.embeddings.create(input=[text], model=self._model)
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in a single API call."""
        if not texts:
            return []
        response = self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

    @staticmethod
    def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a = np.array(vec_a)
        b = np.array(vec_b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def find_similar(
        self,
        query_embedding: list[float],
        candidates: list[dict],
        embedding_key: str = "embedding",
        threshold: float = 0.0,
        top_k: int = 5,
    ) -> list[tuple[dict, float]]:
        """Find the top-k most similar items from candidates.

        Returns list of (item, similarity_score) tuples sorted by descending score.
        """
        scored = []
        for item in candidates:
            item_embedding = item.get(embedding_key)
            if item_embedding is None:
                continue
            score = EmbeddingEngine.cosine_similarity(query_embedding, item_embedding)
            if score >= threshold:
                scored.append((item, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
