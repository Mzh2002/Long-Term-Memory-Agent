"""Tests for the SQLiteStore backend."""

import tempfile
from pathlib import Path

from memory_agent.storage.sqlite_store import SQLiteStore


def _make_store() -> SQLiteStore:
    """Create a temporary store for testing."""
    tmp = tempfile.mktemp(suffix=".db")
    return SQLiteStore(Path(tmp))


def test_add_and_retrieve_episode():
    store = _make_store()
    eid = store.add_episode("Test summary", "Test content", embedding=[0.1, 0.2], importance=0.8)
    episodes = store.get_all_episodes()
    assert len(episodes) == 1
    assert episodes[0]["id"] == eid
    assert episodes[0]["summary"] == "Test summary"
    assert episodes[0]["embedding"] == [0.1, 0.2]
    store.close()


def test_add_and_retrieve_fact():
    store = _make_store()
    fid = store.add_fact("User", "likes", "Python", confidence=0.9, embedding=[0.3, 0.4])
    facts = store.get_all_facts()
    assert len(facts) == 1
    assert facts[0]["id"] == fid
    assert facts[0]["subject"] == "User"
    assert facts[0]["predicate"] == "likes"
    assert facts[0]["object"] == "Python"
    store.close()


def test_fact_upsert():
    store = _make_store()
    id1 = store.add_fact("User", "likes", "Python", confidence=0.5)
    id2 = store.add_fact("User", "likes", "Python", confidence=0.9)
    assert id1 == id2
    facts = store.get_all_facts()
    assert len(facts) == 1
    assert facts[0]["confidence"] == 0.9
    store.close()


def test_add_and_retrieve_procedure():
    store = _make_store()
    pid = store.add_procedure(
        name="greet_user",
        description="How to greet the user",
        steps=["Say hello", "Ask how they are"],
        trigger_condition="new session",
        embedding=[0.5, 0.6],
    )
    procedures = store.get_all_procedures()
    assert len(procedures) == 1
    assert procedures[0]["id"] == pid
    assert procedures[0]["steps"] == ["Say hello", "Ask how they are"]
    store.close()


def test_procedure_outcome_tracking():
    store = _make_store()
    store.add_procedure("test_proc", "desc", ["step1"])
    store.record_procedure_outcome("test_proc", success=True)
    store.record_procedure_outcome("test_proc", success=True)
    store.record_procedure_outcome("test_proc", success=False)
    # Verify by raw query
    row = store._conn.execute(
        "SELECT success_count, failure_count FROM procedures WHERE name = ?", ("test_proc",)
    ).fetchone()
    assert row["success_count"] == 2
    assert row["failure_count"] == 1
    store.close()


def test_get_facts_by_subject():
    store = _make_store()
    store.add_fact("User", "likes", "Python")
    store.add_fact("User", "lives_in", "NYC")
    store.add_fact("Bot", "runs_on", "GPT-4")
    results = store.get_facts_by_subject("User")
    assert len(results) == 2
    store.close()
