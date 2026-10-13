"""Content-hashed feature cache.

Keyed on (sha1(file bytes), feature_kind, encoder_id) so that re-running stages
or swapping encoders does not recompute features that already exist.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import uuid
from collections import OrderedDict
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


class MemoryLRU:
    """Process-wide, byte-capped LRU in front of the on-disk cache.

    Entries are content-addressed (sha1 + kind + encoder), so they never go
    stale. Live hot-folder runs re-load every photo of the run on each ingest;
    without this, a 10k-photo run spends ~20 s per ingest re-reading .npz
    files. ~8 KB per photo, so the default 1 GB covers ~100k photos; past the
    cap the least recently used entries drop back to disk reads."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self._d: "OrderedDict[Any, tuple]" = OrderedDict()
        self._lock = threading.Lock()
        self.bytes = 0
        self.hits = 0
        self.misses = 0

    def get(self, key):
        with self._lock:
            hit = self._d.get(key)
            if hit is None:
                self.misses += 1
                return None
            self._d.move_to_end(key)
            self.hits += 1
            return hit[0]

    def __contains__(self, key) -> bool:
        with self._lock:
            return key in self._d

    def put(self, key, value, nbytes: int) -> None:
        if nbytes > self.max_bytes:
            return
        with self._lock:
            old = self._d.pop(key, None)
            if old is not None:
                self.bytes -= old[1]
            self._d[key] = (value, nbytes)
            self.bytes += nbytes
            while self.bytes > self.max_bytes:
                _, (_, n) = self._d.popitem(last=False)
                self.bytes -= n

    def clear(self) -> None:
        with self._lock:
            self._d.clear(); self.bytes = 0; self.hits = 0; self.misses = 0

    def stats(self) -> Dict[str, float]:
        with self._lock:
            return {"mem_cache_mb": round(self.bytes / 2**20, 1), "mem_cache_entries": len(self._d),
                    "mem_cache_hits": self.hits, "mem_cache_misses": self.misses}


MEMORY = MemoryLRU(int(float(os.environ.get("IMAGE_FILTERER_MEM_CACHE_MB", "1024")) * 2**20))
_ENTRY_OVERHEAD = 400            # dict/key/ndarray headers, roughly


def _frozen(arrays: Dict[str, np.ndarray]) -> tuple:
    """Read-only copies (callers share them) and their size in bytes."""
    out = {}
    for k, a in arrays.items():
        a = np.array(a, copy=True)
        a.setflags(write=False)
        out[k] = a
    return out, sum(a.nbytes for a in out.values()) + _ENTRY_OVERHEAD


def _atomic_write(path: Path, write) -> None:
    """Write via a temp file in the same directory, then rename into place.

    The cache is documented as shareable across machines, i.e. read
    concurrently by other ingests/hot-folder polls/training runs. A direct
    write leaves a window where a reader loads a truncated ``.npz`` or
    ``.json``; ``os.replace`` is atomic on POSIX, so a reader sees either the
    old file or the new one, never a partial one. The temp name is
    per-call-unique so concurrent writers of the same entry never collide.
    """
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def config_namespace(base: str, values: dict, defaults: dict) -> str:
    """Keep the production namespace; distinguish changed measurement inputs."""
    if values == defaults:
        return base
    digest = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:16]
    return f"{base}+cfg-{digest}"


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
        for k in keys:                  # in-memory first: a live run asks for every photo, every ingest
            st = stats.get(k)
            if st is not None:
                digest = MEMORY.get(("sha1", k) + st)
                if digest is not None:
                    known[k] = digest
        ask = [k for k in keys if k not in known]
        with closing(self._connect()) as c:
            # Chunked so we stay under SQLite's variable limit on big folders.
            for i in range(0, len(ask), 500):
                part = ask[i:i + 500]
                q = f"SELECT path, size, mtime_ns, sha1 FROM sha1_memo WHERE path IN ({','.join('?' * len(part))})"
                for path, size, mtime_ns, digest in c.execute(q, part):
                    st = stats.get(path)
                    if st is not None and st[0] == size and st[1] == mtime_ns:
                        known[path] = digest
                        MEMORY.put(("sha1", path) + st, digest, 200 + len(path))

        fresh: List[tuple] = []
        out: List[str] = []
        for k in keys:
            digest = known.get(k)
            if digest is None:
                digest = file_sha1(Path(k))
                st = stats.get(k)
                if st is not None:
                    fresh.append((k, st[0], st[1], digest))
                    MEMORY.put(("sha1", k) + st, digest, 200 + len(k))
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

    def _entry_dir(self, sha1: str, create: bool = False) -> Path:
        sub = self.root / sha1[:2] / sha1[2:4]
        if create:                       # only writers; reads stay syscall-free on memory hits
            sub.mkdir(parents=True, exist_ok=True)
        return sub

    def _entry_path(self, sha1: str, kind: str, encoder_id: str) -> Path:
        # encoder_id is sanitized for filesystem
        eid = encoder_id.replace("/", "__").replace(":", "_")
        return self._entry_dir(sha1) / f"{sha1}.{kind}.{eid}.npz"

    def has(self, sha1: str, kind: str, encoder_id: str = "none") -> bool:
        path = self._entry_path(sha1, kind, encoder_id)
        return path in MEMORY or path.exists()

    def load(self, sha1: str, kind: str, encoder_id: str = "none") -> Dict[str, np.ndarray]:
        path = self._entry_path(sha1, kind, encoder_id)
        hit = MEMORY.get(path)
        if hit is not None:
            return dict(hit)
        with np.load(path, allow_pickle=False) as z:
            arrays, nbytes = _frozen({k: z[k] for k in z.files})
        MEMORY.put(path, arrays, nbytes)
        return dict(arrays)

    def save(
        self,
        sha1: str,
        kind: str,
        arrays: Dict[str, np.ndarray],
        encoder_id: str = "none",
    ) -> None:
        def _write(tmp: Path) -> None:
            # np.savez_compressed appends ".npz" to any filename that lacks
            # it, which would silently write past the atomic-rename target.
            # A file handle bypasses that extension-guessing entirely.
            with open(tmp, "wb") as f:
                np.savez_compressed(f, **arrays)

        self._entry_dir(sha1, create=True)
        path = self._entry_path(sha1, kind, encoder_id)
        _atomic_write(path, _write)
        MEMORY.put(path, *_frozen(arrays))

    # --- side-channel: per-image scalar metadata that is cheap to recompute ---

    def save_meta(self, sha1: str, kind: str, payload: Dict[str, Any]) -> None:
        path = self._entry_dir(sha1, create=True) / f"{sha1}.{kind}.meta.json"
        text = json.dumps(payload)
        _atomic_write(path, lambda tmp: tmp.write_text(text))
        MEMORY.put(path, text, len(text) + _ENTRY_OVERHEAD)

    def load_meta(self, sha1: str, kind: str) -> Optional[Dict[str, Any]]:
        path = self._entry_dir(sha1) / f"{sha1}.{kind}.meta.json"
        text = MEMORY.get(path)
        if text is None:
            if not path.exists():
                return None
            text = path.read_text()
            MEMORY.put(path, text, len(text) + _ENTRY_OVERHEAD)
        return json.loads(text)
