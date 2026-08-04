"""Feature extraction with progress reporting.

Thin wrapper over :class:`features.FeatureExtractor` that yields per-chunk
progress, so the ingest bar can move during the slowest phase of a run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Sequence

from .cache_io import FeatureCache
from .features import FeatureBundle, FeatureExtractor

from .config import Config

ProgressCB = Callable[[int, int], None]  # (done, total)


def load_or_extract(
    cfg: Config,
    paths: Sequence[Path],
    *,
    desc: str = "features",
    chunk_size: int = 64,
    progress_cb: Optional[ProgressCB] = None,
) -> List[FeatureBundle]:
    """Extract (or load-from-cache) features. Calls ``progress_cb(done, total)``
    after each chunk so callers can drive a progress bar."""
    cache = FeatureCache(cfg.cache_dir)
    ex = FeatureExtractor(cfg.features, cache)
    try:
        if progress_cb is None:
            return ex.extract_paths(list(paths), desc=desc, chunk_size=chunk_size)
        out: List[FeatureBundle] = []
        n = len(paths)
        for start in range(0, n, chunk_size):
            sub = list(paths[start : start + chunk_size])
            out.extend(ex.extract_paths(sub, desc=desc, chunk_size=chunk_size))
            progress_cb(min(start + len(sub), n), n)
        return out
    finally:
        ex.close()
