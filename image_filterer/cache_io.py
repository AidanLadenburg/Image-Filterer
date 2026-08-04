"""Content-hashed feature cache.

Keyed on (sha1(file bytes), feature_kind, encoder_id) so that re-running stages
or swapping encoders does not recompute features that already exist.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

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
