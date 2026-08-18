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
        self._schema_ready = False
        # One long-lived writer connection. Two reasons it must persist:
        # closing the last connection to a WAL database triggers a checkpoint
        # (an fsync, ~20 ms — paid on every progress update during an ingest),
        # and keeping one open means short-lived reader connections never
        # trigger it either.
        self._wconn: Optional[sqlite3.Connection] = None
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create the directory, the schema, and put the DB in WAL mode. Once."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(str(self.path), timeout=30)) as conn:
            # WAL lets readers proceed while a writer holds the lock. Without it
            # an ingest — which writes progress continuously — starves every
            # viewer polling the API, and the whole UI stalls until it finishes.
            conn.execute("PRAGMA journal_mode=WAL")
            # NORMAL is the right durability trade here: the registry is
            # reconstructible bookkeeping, not the photos themselves.
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(_SCHEMA)
            conn.commit()
        self._schema_ready = True

    def _connect(self) -> sqlite3.Connection:
        # Deliberately does NOT run the schema script: executescript() implicitly
        # opens a write transaction, so doing it here made every *read* take a
        # write lock. Recovery from a deleted DB is handled by _retry below.
        if not self._schema_ready:
            self._ensure_schema()
        conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        conn.row_factory = sqlite3.Row
        # journal_mode persists in the file, but synchronous is per-connection —
        # left at the default FULL, every progress update costs an fsync (~21 ms),
        # which an ingest performs continuously.
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _writer(self) -> sqlite3.Connection:
        """The persistent write connection; caller must hold ``_wlock``."""
        if self._wconn is None:
            if not self._schema_ready:
                self._ensure_schema()
            self._wconn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
            self._wconn.row_factory = sqlite3.Row
            self._wconn.execute("PRAGMA synchronous=NORMAL")
        return self._wconn

    def _write(self, fn):
        """Run ``fn(conn)`` on the writer, rebuilding the registry if it vanished."""
        try:
            return fn(self._writer())
        except sqlite3.Error as exc:
            if "no such table" not in str(exc) and "unable to open" not in str(exc) \
               and "disk I/O" not in str(exc):
                raise
            try:
                if self._wconn is not None:
                    self._wconn.close()
            except sqlite3.Error:
                pass
            self._wconn = None
            self._schema_ready = False
            self._ensure_schema()
            return fn(self._writer())

    def _retry(self, fn):
        """Run ``fn(conn)``; if the registry vanished underneath us, rebuild it once.

        Preserves the old self-healing behaviour (a stray `rm` degrades to an
        empty registry rather than 500-ing forever) without paying for it on
        every single query.
        """
        try:
            with closing(self._connect()) as c:
                return fn(c)
        except sqlite3.OperationalError as exc:
            if "no such table" not in str(exc) and "unable to open" not in str(exc):
                raise
            self._schema_ready = False
            self._ensure_schema()
            with closing(self._connect()) as c:
                return fn(c)

    # NOTE: `with sqlite3.connect(...) as c` commits but does NOT close the
    # connection — that leaks a file descriptor per call, and with clients polling
    # /api/active + /api/state every couple seconds it exhausts the FD limit
    # ("Too many open files"). `closing(...)` guarantees the handle is closed;
    # writes commit explicitly.

    def create(self, name: str, folder: str) -> int:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        def go(c):
            cur = c.execute(
                "INSERT INTO runs (name, folder, status, created_at, message) "
                "VALUES (?, ?, 'queued', ?, 'Queued')",
                (name, folder, ts),
            )
            c.commit()
            return int(cur.lastrowid)
        with self._wlock:
            return self._write(go)

    def update(self, run_id: int, **fields) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k} = ?" for k in fields)
        def go(c):
            c.execute(f"UPDATE runs SET {cols} WHERE id = ?", (*fields.values(), run_id))
            c.commit()
        with self._wlock:
            self._write(go)

    def get(self, run_id: int) -> Optional[Dict]:
        def go(c):
            row = c.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            return dict(row) if row else None
        return self._retry(go)

    def list(self, include_unready: bool = True) -> List[Dict]:
        q = "SELECT * FROM runs"
        if not include_unready:
            q += " WHERE status = 'ready'"
        q += " ORDER BY id DESC"
        return self._retry(lambda c: [dict(r) for r in c.execute(q).fetchall()])

    def delete(self, run_id: int) -> None:
        def go(c):
            c.execute("DELETE FROM runs WHERE id = ?", (run_id,))
            c.commit()
        with self._wlock:
            self._write(go)
