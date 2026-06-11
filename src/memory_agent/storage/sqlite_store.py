"""SQLite-based persistent storage for memory layers."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SQLiteStore:
    """Persistent storage backend using SQLite for all memory layers."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize database tables for all memory layers."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                summary TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT,
                importance REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT
            );

            CREATE TABLE IF NOT EXISTS semantic_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                source TEXT,
                embedding TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS procedures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL,
                steps TEXT NOT NULL,
                trigger_condition TEXT,
                embedding TEXT,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp);
            CREATE INDEX IF NOT EXISTS idx_semantic_subject ON semantic_facts(subject);
            CREATE INDEX IF NOT EXISTS idx_procedures_name ON procedures(name);
        """)
        self._conn.commit()

    def add_episode(
        self,
        summary: str,
        content: str,
        embedding: list[float] | None = None,
        importance: float = 0.5,
    ) -> int:
        """Store an episodic memory entry."""
        now = datetime.now(timezone.utc).isoformat()
        cursor = self._conn.execute(
            """INSERT INTO episodes (timestamp, summary, content, embedding, importance)
               VALUES (?, ?, ?, ?, ?)""",
            (now, summary, content, json.dumps(embedding) if embedding else None, importance),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_all_episodes(self) -> list[dict[str, Any]]:
        """Get all episodes for similarity search."""
        cursor = self._conn.execute(
            "SELECT id, timestamp, summary, content, embedding, importance FROM episodes"
        )
        rows = cursor.fetchall()
        results = []
        for row in rows:
            entry = dict(row)
            if entry["embedding"]:
                entry["embedding"] = json.loads(entry["embedding"])
            results.append(entry)
        return results

    def update_episode_access(self, episode_id: int) -> None:
        """Update access count and timestamp for an episode."""
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE episodes SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
            (now, episode_id),
        )
        self._conn.commit()

    def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        source: str | None = None,
        embedding: list[float] | None = None,
    ) -> int:
        """Store a semantic fact (knowledge triple)."""
        now = datetime.now(timezone.utc).isoformat()
        # Check if fact already exists and update it
        existing = self._conn.execute(
            "SELECT id FROM semantic_facts WHERE subject = ? AND predicate = ? AND object = ?",
            (subject, predicate, obj),
        ).fetchone()

        if existing:
            self._conn.execute(
                """UPDATE semantic_facts SET confidence = ?, source = ?, embedding = ?,
                   updated_at = ? WHERE id = ?""",
                (
                    confidence,
                    source,
                    json.dumps(embedding) if embedding else None,
                    now,
                    existing["id"],
                ),
            )
            self._conn.commit()
            return existing["id"]

        cursor = self._conn.execute(
            """INSERT INTO semantic_facts
               (subject, predicate, object, confidence, source, embedding, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                subject,
                predicate,
                obj,
                confidence,
                source,
                json.dumps(embedding) if embedding else None,
                now,
                now,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_all_facts(self) -> list[dict[str, Any]]:
        """Get all semantic facts for similarity search."""
        cursor = self._conn.execute(
            "SELECT id, subject, predicate, object, confidence, source, embedding "
            "FROM semantic_facts"
        )
        rows = cursor.fetchall()
        results = []
        for row in rows:
            entry = dict(row)
            if entry["embedding"]:
                entry["embedding"] = json.loads(entry["embedding"])
            results.append(entry)
        return results

    def get_facts_by_subject(self, subject: str) -> list[dict[str, Any]]:
        """Get facts related to a specific subject."""
        cursor = self._conn.execute(
            "SELECT * FROM semantic_facts WHERE subject LIKE ?",
            (f"%{subject}%",),
        )
        return [dict(row) for row in cursor.fetchall()]

    def add_procedure(
        self,
        name: str,
        description: str,
        steps: list[str],
        trigger_condition: str | None = None,
        embedding: list[float] | None = None,
    ) -> int:
        """Store a procedural memory entry."""
        now = datetime.now(timezone.utc).isoformat()
        # Upsert
        existing = self._conn.execute(
            "SELECT id FROM procedures WHERE name = ?", (name,)
        ).fetchone()

        if existing:
            self._conn.execute(
                """UPDATE procedures SET description = ?, steps = ?, trigger_condition = ?,
                   embedding = ?, updated_at = ? WHERE id = ?""",
                (
                    description,
                    json.dumps(steps),
                    trigger_condition,
                    json.dumps(embedding) if embedding else None,
                    now,
                    existing["id"],
                ),
            )
            self._conn.commit()
            return existing["id"]

        cursor = self._conn.execute(
            """INSERT INTO procedures
               (name, description, steps, trigger_condition, embedding, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                name,
                description,
                json.dumps(steps),
                trigger_condition,
                json.dumps(embedding) if embedding else None,
                now,
                now,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_all_procedures(self) -> list[dict[str, Any]]:
        """Get all procedures for similarity search."""
        cursor = self._conn.execute(
            "SELECT id, name, description, steps, trigger_condition, embedding FROM procedures"
        )
        rows = cursor.fetchall()
        results = []
        for row in rows:
            entry = dict(row)
            entry["steps"] = json.loads(entry["steps"])
            if entry["embedding"]:
                entry["embedding"] = json.loads(entry["embedding"])
            results.append(entry)
        return results

    def record_procedure_outcome(self, name: str, success: bool) -> None:
        """Track procedure execution outcomes."""
        col = "success_count" if success else "failure_count"
        self._conn.execute(
            f"UPDATE procedures SET {col} = {col} + 1 WHERE name = ?",  # noqa: S608
            (name,),
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
