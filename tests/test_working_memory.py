"""Tests for the WorkingMemory layer."""

from memory_agent.config import MemoryConfig
from memory_agent.memory.working import WorkingMemory


def test_add_message():
    config = MemoryConfig(max_context_messages=5, max_context_tokens=10000)
    wm = WorkingMemory(config)
    wm.add_message("user", "Hello")
    assert len(wm.messages) == 1
    assert wm.messages[0].role == "user"
    assert wm.messages[0].content == "Hello"


def test_context_window_includes_system_prompt():
    config = MemoryConfig(max_context_messages=10, max_context_tokens=10000)
    wm = WorkingMemory(config)
    wm.system_prompt = "You are helpful."
    wm.add_message("user", "Hi")
    ctx = wm.get_context_window()
    assert ctx[0]["role"] == "system"
    assert ctx[0]["content"] == "You are helpful."
    assert ctx[1]["role"] == "user"


def test_eviction_by_message_count():
    config = MemoryConfig(max_context_messages=3, max_context_tokens=100000)
    wm = WorkingMemory(config)
    for i in range(5):
        wm.add_message("user", f"Message {i}")
    assert len(wm.messages) == 3


def test_eviction_returns_evicted_messages():
    config = MemoryConfig(max_context_messages=2, max_context_tokens=100000)
    wm = WorkingMemory(config)
    wm.add_message("user", "First")
    wm.add_message("user", "Second")
    evicted = wm.add_message("user", "Third")
    assert evicted is not None
    assert len(evicted) == 1
    assert evicted[0].content == "First"


def test_scratchpad():
    config = MemoryConfig()
    wm = WorkingMemory(config)
    wm.set_scratchpad("task", "testing")
    assert wm.get_scratchpad("task") == "testing"
    assert wm.get_scratchpad("missing") is None
    wm.clear_scratchpad()
    assert wm.get_scratchpad("task") is None


def test_clear_returns_all_messages():
    config = MemoryConfig(max_context_messages=10, max_context_tokens=100000)
    wm = WorkingMemory(config)
    wm.add_message("user", "A")
    wm.add_message("assistant", "B")
    cleared = wm.clear()
    assert len(cleared) == 2
    assert len(wm.messages) == 0
