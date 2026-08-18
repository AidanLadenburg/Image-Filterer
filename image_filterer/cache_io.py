"""Content-hashed feature cache.

Keyed on (sha1(file bytes), feature_kind, encoder_id) so that re-running stages
or swapping encoders does not recompute features that already exist.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def file_sha1(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


class Sha1Memo:
    """Remembers a file's sha1 against ``(path, size, mtime_ns)``.

    Every cache lookup in this project is keyed by content hash, which means
    deciding "have I already processed this file?" costs a full read of the file.
    That is fine once. It is ruinous for a folder that gets rescanned on a timer:
    re-checking a 3,000-frame shoot means reading ~30 GB, every pass, purely to
    discover there is nothing to do.

    Size and mtime are not a cryptographic guarantee, but they are the same
    signal make(1), rsync and every build system rely on, and the failure mode is
    contained: an edit that preserves both would have to be byte-length-identical
    and timestamp-preserving. A mismatch simply falls back to hashing.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS sha1_memo (
        path     TEXT PRIMARY KEY,
        size     INTEGER NOT NULL,
        mtime_ns INTEGER NOT NULL,
        sha1     TEXT NOT NULL
    );
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        with closing(self._connect()):
            pass

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        conn.executescript(self._SCHEMA)
        return conn

    def sha1_many(self, paths: Sequence[Path]) -> List[str]:
        """sha1 for each path, hashing only what the memo can't vouch for."""
        if not paths:
            return []
        keys = [str(Path(p)) for p in paths]
        stats: Dict[str, Optional[tuple]] = {}
        for k in keys:
            try:
                st = Path(k).stat()
                stats[k] = (st.st_size, st.st_mtime_ns)
            except OSError:
                stats[k] = None

        known: Dict[str, str] = {}
        with closing(self._connect()) as c:
            # Chunked so we stay under SQLite's variable limit on big folders.
            for i in range(0, len(keys), 500):
                part = keys[i:i + 500]
                q = f"SELECT path, size, mtime_ns, sha1 FROM sha1_memo WHERE path IN ({','.join('?' * len(part))})"
                for path, size, mtime_ns, digest in c.execute(q, part):
                    st = stats.get(path)
                    if st is not None and st[0] == size and st[1] == mtime_ns:
                        known[path] = digest

        fresh: List[tuple] = []
        out: List[str] = []
        for k in keys:
            digest = known.get(k)
            if digest is None:
                digest = file_sha1(Path(k))
                st = stats.get(k)
                if st is not None:
                    fresh.append((k, st[0], st[1], digest))
            out.append(digest)

        if fresh:
            with self._lock, closing(self._connect()) as c:
                c.executemany(
                    "INSERT OR REPLACE INTO sha1_memo (path, size, mtime_ns, sha1) VALUES (?, ?, ?, ?)",
                    fresh)
                c.commit()
        return out

    def sha1(self, path: Path) -> str:
        return self.sha1_many([path])[0]


@dataclass
class FeatureCache:
    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def _entry_dir(self, sha1: str) -> Path:
        sub = self.root / sha1[:2] / sha1[2:4]
        sub.mkdir(parents=True, exist_ok=True)
        return sub

    def _entry_path(self, sha1: str, kind: str, encoder_id: str) -> Path:
        # encoder_id is sanitized for filesystem
        eid = encoder_id.replace("/", "__").replace(":", "_")
        return self._entry_dir(sha1) / f"{sha1}.{kind}.{eid}.npz"

    def has(self, sha1: str, kind: str, encoder_id: str = "none") -> bool:
        return self._entry_path(sha1, kind, encoder_id).exists()

    def load(self, sha1: str, kind: str, encoder_id: str = "none") -> Dict[str, np.ndarray]:
        with np.load(self._entry_path(sha1, kind, encoder_id), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def save(
        self,
        sha1: str,
        kind: str,
        arrays: Dict[str, np.ndarray],
        encoder_id: str = "none",
    ) -> None:
        np.savez_compressed(self._entry_path(sha1, kind, encoder_id), **arrays)

    # --- side-channel: per-image scalar metadata that is cheap to recompute ---

    def save_meta(self, sha1: str, kind: str, payload: Dict[str, Any]) -> None:
        path = self._entry_dir(sha1) / f"{sha1}.{kind}.meta.json"
        path.write_text(json.dumps(payload))

    def load_meta(self, sha1: str, kind: str) -> Optional[Dict[str, Any]]:
        path = self._entry_dir(sha1) / f"{sha1}.{kind}.meta.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())
