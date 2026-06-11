"""Tests for the embedding utilities (cosine similarity, find_similar)."""

from memory_agent.embeddings import EmbeddingEngine


def test_cosine_similarity_identical():
    vec = [1.0, 0.0, 0.0]
    assert abs(EmbeddingEngine.cosine_similarity(vec, vec) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert abs(EmbeddingEngine.cosine_similarity(a, b)) < 1e-6


def test_cosine_similarity_opposite():
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert abs(EmbeddingEngine.cosine_similarity(a, b) - (-1.0)) < 1e-6


def test_cosine_similarity_zero_vector():
    a = [0.0, 0.0]
    b = [1.0, 2.0]
    assert EmbeddingEngine.cosine_similarity(a, b) == 0.0


def test_find_similar():
    candidates = [
        {"text": "cat", "embedding": [1.0, 0.0, 0.0]},
        {"text": "dog", "embedding": [0.9, 0.1, 0.0]},
        {"text": "car", "embedding": [0.0, 0.0, 1.0]},
    ]
    query = [1.0, 0.0, 0.0]

    # Use a dummy engine instance (we only need the static/instance method)
    results = EmbeddingEngine.find_similar(
        None,  # type: ignore[arg-type]
        query_embedding=query,
        candidates=candidates,
        top_k=2,
    )
    assert len(results) == 2
    assert results[0][0]["text"] == "cat"
    assert results[1][0]["text"] == "dog"


def test_find_similar_with_threshold():
    candidates = [
        {"text": "close", "embedding": [1.0, 0.0]},
        {"text": "far", "embedding": [0.0, 1.0]},
    ]
    query = [1.0, 0.0]
    results = EmbeddingEngine.find_similar(
        None,  # type: ignore[arg-type]
        query_embedding=query,
        candidates=candidates,
        threshold=0.5,
        top_k=10,
    )
    assert len(results) == 1
    assert results[0][0]["text"] == "close"
