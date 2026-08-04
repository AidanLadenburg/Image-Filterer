"""SQLite registry of ingestion runs.

Each run is a row here plus its own folder under ``data/runs/<folder>/``. The DB
holds status + live progress so the UI can poll it. Writes are serialized behind
a lock; reads open a short-lived connection (SQLite handles concurrent readers).
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL,
    folder     TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'queued',   -- queued | ingesting | ready | error
    created_at TEXT NOT NULL,
    n_images   INTEGER NOT NULL DEFAULT 0,
    n_bursts   INTEGER NOT NULL DEFAULT 0,
    progress   REAL    NOT NULL DEFAULT 0,        -- 0..100
    message    TEXT    NOT NULL DEFAULT '',
    error      TEXT    NOT NULL DEFAULT ''
);
"""


class RunDB:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._wlock = threading.Lock()
        with closing(self._connect()):  # ensures directory + schema
            pass

    def _connect(self) -> sqlite3.Connection:
        # Self-healing: recreate the data directory + schema if they've gone
        # missing (e.g. the folder was deleted while the server was running), so
        # a stray `rm` degrades to an empty registry instead of 500-ing forever.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA)  # CREATE TABLE IF NOT EXISTS — idempotent, cheap
        return conn

    # NOTE: `with sqlite3.connect(...) as c` commits but does NOT close the
    # connection — that leaks a file descriptor per call, and with clients polling
    # /api/active + /api/state every couple seconds it exhausts the FD limit
    # ("Too many open files"). `closing(...)` guarantees the handle is closed;
    # writes commit explicitly.

    def create(self, name: str, folder: str) -> int:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._wlock, closing(self._connect()) as c:
            cur = c.execute(
                "INSERT INTO runs (name, folder, status, created_at, message) "
                "VALUES (?, ?, 'queued', ?, 'Queued')",
                (name, folder, ts),
            )
            c.commit()
            return int(cur.lastrowid)

    def update(self, run_id: int, **fields) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._wlock, closing(self._connect()) as c:
            c.execute(f"UPDATE runs SET {cols} WHERE id = ?", (*fields.values(), run_id))
            c.commit()

    def get(self, run_id: int) -> Optional[Dict]:
        with closing(self._connect()) as c:
            row = c.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            return dict(row) if row else None

    def list(self, include_unready: bool = True) -> List[Dict]:
        q = "SELECT * FROM runs"
        if not include_unready:
            q += " WHERE status = 'ready'"
        q += " ORDER BY id DESC"
        with closing(self._connect()) as c:
            return [dict(r) for r in c.execute(q).fetchall()]

    def delete(self, run_id: int) -> None:
        with self._wlock, closing(self._connect()) as c:
            c.execute("DELETE FROM runs WHERE id = ?", (run_id,))
            c.commit()
