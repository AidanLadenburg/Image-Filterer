"""Star ownership shared across every cached generation of a run."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from .pipeline import atomic_write


class StarStore:
    def __init__(self, run_dir: Path):
        self.path = run_dir / "stars.json"
        self.lock = threading.RLock()
        try:
            blob = json.loads(self.path.read_text())
        except FileNotFoundError:
            blob = {}
        # Do not silently overwrite corrupt picks with an empty selection.
        if isinstance(blob.get("stars"), dict):
            self.stars = {str(p): set(map(str, owners))
                          for p, owners in blob["stars"].items() if owners}
        else:
            self.stars = {str(p): {"legacy"} for p in blob.get("starred", [])}

    def _commit(self, stars):
        payload = json.dumps({"version": 2,
                              "stars": {p: sorted(o) for p, o in stars.items() if o}})
        atomic_write(self.path, lambda tmp: tmp.write_text(payload))
        # Preserve the dictionary identity held by older run snapshots. Only
        # update memory after persistence succeeds.
        self.stars.clear()
        self.stars.update(stars)

    def set(self, path: str, viewer: str, wanted: bool):
        with self.lock:
            stars = {p: set(o) for p, o in self.stars.items()}
            owners = stars.setdefault(path, set())
            if wanted:
                owners.add(viewer)
            else:
                owners.discard(viewer)
            if not owners:
                stars.pop(path, None)
            self._commit(stars)

    def clear(self, viewer: str) -> int:
        with self.lock:
            count = sum(viewer in o for o in self.stars.values())
            self._commit({p: o - {viewer} for p, o in self.stars.items()
                          if o - {viewer}})
            return count
