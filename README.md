# Long-Term Memory Agent

A Python-based AI agent with a **4-layer memory system** inspired by [OpenAI's memory architecture](https://openai.com/index/memory-and-new-controls-for-chatgpt/). The agent can remember facts, recall past conversations, and learn procedures across sessions.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Memory Agent                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Layer 1: Working Memory (in-context)                   │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Sliding window of recent messages + scratchpad    │  │
│  │ Evicts → consolidation when full                  │  │
│  └───────────────────────────────────────────────────┘  │
│                         │                               │
│            ┌────────────┼────────────┐                  │
│            ▼            ▼            ▼                  │
│  Layer 2: Episodic   Layer 3: Semantic  Layer 4: Procedural │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ Past convos │  │ Facts &      │  │ Learned       │  │
│  │ & events    │  │ knowledge    │  │ procedures    │  │
│  │ (what       │  │ (what is     │  │ (how to do    │  │
│  │  happened)  │  │  true)       │  │  things)      │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
│            │            │            │                  │
│            └────────────┴────────────┘                  │
│                         │                               │
│              ┌──────────┴──────────┐                    │
│              │   SQLite + Embeddings│                    │
│              │   (persistent store) │                    │
│              └─────────────────────┘                    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Memory Layers

| Layer | Name | Purpose | Persistence |
|-------|------|---------|-------------|
| 1 | **Working Memory** | Current conversation context (sliding window) | Session only |
| 2 | **Episodic Memory** | Past conversation summaries and events | Long-term (SQLite) |
| 3 | **Semantic Memory** | Facts, preferences, and knowledge triples | Long-term (SQLite) |
| 4 | **Procedural Memory** | Learned multi-step procedures | Long-term (SQLite) |

### How it works

1. **User sends a message** → the agent retrieves relevant context from all 3 long-term layers using vector similarity search.
2. **Context is injected** into the working memory alongside the conversation.
3. **LLM generates a response** using the augmented context.
4. **When working memory overflows**, evicted messages are *consolidated* — an LLM call extracts episodes, facts, and procedures and stores them in the appropriate long-term layer.
5. **On session end**, all remaining context is consolidated for future recall.

## Installation

```bash
# Clone the repository
git clone https://github.com/Mzh2002/Long-Term-Memory-Agent.git
cd Long-Term-Memory-Agent

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the package
pip install -e ".[dev]"

# Set your OpenAI API key
export OPENAI_API_KEY="your-key-here"
```

## Usage

### Interactive Chat

```bash
memory-agent
```

This starts an interactive session. The agent will:
- Remember facts you share across sessions
- Recall relevant past conversations
- Learn and apply procedures

### Commands (in chat)

| Command | Description |
|---------|-------------|
| `/stats` | Show memory statistics |
| `/facts` | List all stored semantic facts |
| `/quit` | End session and consolidate memory |

### CLI Options

```bash
memory-agent --model gpt-4o          # Use a different model
memory-agent --storage-dir ./my_data  # Custom storage location
memory-agent --stats                  # Show memory stats and exit
```

### Programmatic Usage

```python
from memory_agent.agent import MemoryAgent
from memory_agent.config import MemoryConfig

config = MemoryConfig(openai_api_key="sk-...")
agent = MemoryAgent(config)

# Chat
response = agent.chat("My name is Alice and I'm a data scientist.")
print(response)

# The agent now remembers your name and profession

# Manually store a fact
agent.save_fact("User", "name_is", "Alice")
agent.save_fact("User", "profession_is", "data scientist")

# Store a procedure
agent.save_procedure(
    name="data_analysis",
    description="Steps for exploratory data analysis",
    steps=["Load data", "Check shape & dtypes", "Handle nulls", "Visualize distributions"],
    trigger="user asks about analyzing data"
)

# Check memory stats
print(agent.get_memory_stats())

# End session (consolidates remaining working memory)
agent.end_session()
agent.close()
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `OPENAI_API_KEY` | (required) | OpenAI API key |
| `MEMORY_AGENT_CHAT_MODEL` | `gpt-4o-mini` | Chat completion model |
| `MEMORY_AGENT_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `MEMORY_AGENT_MAX_CONTEXT` | `20` | Max messages in working memory |
| `MEMORY_AGENT_MAX_TOKENS` | `4000` | Max tokens in working memory |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/
```

## Project Structure

```
src/memory_agent/
├── __init__.py          # Package metadata
├── agent.py             # Main MemoryAgent orchestrator
├── config.py            # Configuration dataclass
├── consolidation.py     # Memory consolidation (working → long-term)
├── embeddings.py        # Vector embedding utilities
├── main.py              # CLI entry point
├── memory/
│   ├── __init__.py
│   ├── working.py       # Layer 1: Working Memory
│   ├── episodic.py      # Layer 2: Episodic Memory
│   ├── semantic.py      # Layer 3: Semantic Memory
│   └── procedural.py    # Layer 4: Procedural Memory
└── storage/
    ├── __init__.py
    └── sqlite_store.py  # SQLite persistence backend
```

## License

MIT
