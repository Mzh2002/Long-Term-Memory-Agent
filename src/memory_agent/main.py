"""CLI entry point for the Memory Agent."""

from __future__ import annotations

import argparse
import sys

from memory_agent.agent import MemoryAgent
from memory_agent.config import MemoryConfig


def main() -> None:
    """Run the memory agent in interactive chat mode."""
    parser = argparse.ArgumentParser(
        description="Long-Term Memory Agent — a 4-layer memory system inspired by OpenAI"
    )
    parser.add_argument(
        "--model", default=None, help="Chat model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--storage-dir", default=None, help="Directory for persistent memory storage"
    )
    parser.add_argument(
        "--stats", action="store_true", help="Show memory statistics and exit"
    )
    args = parser.parse_args()

    config = MemoryConfig.from_env()
    if args.model:
        config.chat_model = args.model
    if args.storage_dir:
        from pathlib import Path

        config.storage_dir = Path(args.storage_dir)

    if not config.openai_api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)

    agent = MemoryAgent(config)

    if args.stats:
        stats = agent.get_memory_stats()
        print("\n📊 Memory Statistics:")
        print(f"  Working Memory: {stats['working_memory']['messages']} messages, "
              f"{stats['working_memory']['tokens']} tokens")
        print(f"  Episodic Memory: {stats['episodic_memory']['total_episodes']} episodes")
        print(f"  Semantic Memory: {stats['semantic_memory']['total_facts']} facts")
        print(f"  Procedural Memory: {stats['procedural_memory']['total_procedures']} procedures")
        agent.close()
        return

    print("╔══════════════════════════════════════════════════╗")
    print("║   Long-Term Memory Agent (4-Layer Architecture) ║")
    print("╠══════════════════════════════════════════════════╣")
    print("║  Commands:                                      ║")
    print("║    /stats  — Show memory statistics             ║")
    print("║    /facts  — List stored semantic facts         ║")
    print("║    /quit   — End session and consolidate memory ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            if user_input == "/quit":
                print("\nConsolidating session memory...")
                result = agent.end_session()
                print(f"  Stored: {result['episodes']} episodes, "
                      f"{result['facts']} facts, {result['procedures']} procedures")
                break

            if user_input == "/stats":
                stats = agent.get_memory_stats()
                print(f"\n  Working: {stats['working_memory']['messages']} msgs, "
                      f"{stats['working_memory']['tokens']} tokens")
                print(f"  Episodic: {stats['episodic_memory']['total_episodes']} episodes")
                print(f"  Semantic: {stats['semantic_memory']['total_facts']} facts")
                proc_count = stats['procedural_memory']['total_procedures']
                print(f"  Procedural: {proc_count} procedures\n")
                continue

            if user_input == "/facts":
                facts = agent._store.get_all_facts()
                if not facts:
                    print("\n  No facts stored yet.\n")
                else:
                    print(f"\n  Stored facts ({len(facts)}):")
                    for f in facts:
                        print(f"    • {f['subject']} {f['predicate']} {f['object']}")
                    print()
                continue

            response = agent.chat(user_input)
            print(f"\nAgent: {response}\n")

    except KeyboardInterrupt:
        print("\n\nSession interrupted. Consolidating memory...")
        result = agent.end_session()
        print(f"  Stored: {result['episodes']} episodes, "
              f"{result['facts']} facts, {result['procedures']} procedures")

    agent.close()
    print("Goodbye!")


if __name__ == "__main__":
    main()
