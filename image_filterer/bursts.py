"""EXIF + embedding-aware burst clustering.

Camera identity is EXIF-only — we deliberately don't fall back to filename
prefixes because the user's filename conventions may not be stable going forward.

  - Canon bodies expose ``Exif.Photo.BodySerialNumber`` cleanly → per-body unique
  - Sony ILCE-1 / ILCE-1M2 don't expose any per-body identifier in EXIF (verified
    via pyexiv2; ``Sony1.SonyModelID`` is the same for every ILCE-1). The best
    available per-rig discriminator is ``Make/Model/LensModel`` — two photographers
    with the same body+lens combo will be merged, but the embedding-similarity
    step below splits their bursts whenever their frames diverge visually.
  - Other / unknown bodies fall back to ``Make/Model``.

Within a single camera id, two consecutive frames belong to the same burst when:
    dt ≤ short_gap_seconds                                          (always merge)
    OR (dt ≤ long_gap_seconds AND cosine_sim ≥ similarity_threshold)

When no embedding is provided, falls back to the old time-only rule with
``gap_seconds``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import ExifTags, Image

try:
    import pyexiv2  # for Sony lens-model reads (PIL doesn't always surface it)
    _HAS_PYEXIV2 = True
except Exception:
    _HAS_PYEXIV2 = False


@dataclass
class FrameMeta:
    path: Path
    camera_id: str
    captured_at: datetime  # millisecond resolution where SubsecTimeOriginal exists


def _camera_id_from_exif(named: Dict[str, object], path: Path) -> str:
    """EXIF-only camera identity. No filename fallback by design."""
    body = str(named.get("BodySerialNumber") or "").strip()
    if body:
        return f"body:{body}"

    make = str(named.get("Make") or "").strip()
    model = str(named.get("Model") or "").strip()
    lens_model = str(named.get("LensModel") or "").strip()

    # pyexiv2 reaches into Sony maker notes for things PIL doesn't surface
    if not lens_model and _HAS_PYEXIV2:
        try:
            img = pyexiv2.Image(str(path))
            exif = img.read_exif()
            img.close()
            lens_model = str(exif.get("Exif.Photo.LensModel") or exif.get("Exif.Image.LensModel") or "").strip()
        except Exception:
            pass

    if make and model and lens_model:
        return f"rig:{make}/{model}/{lens_model}"
    if make and model:
        return f"model:{make}/{model}"
    return f"unknown:{path.name}"  # last-resort singleton


def read_frame_meta(path: Path) -> Optional[FrameMeta]:
    try:
        im = Image.open(path)
        exif = im._getexif() or {}
        named = {ExifTags.TAGS.get(k, str(k)): v for k, v in exif.items()}
        dt_str = named.get("DateTimeOriginal")
        if not dt_str:
            return None
        dt = datetime.strptime(str(dt_str), "%Y:%m:%d %H:%M:%S")
        subsec_raw = str(named.get("SubsecTimeOriginal", "0")).strip()
        subsec = int(subsec_raw) if subsec_raw.isdigit() else 0
        if subsec >= 100:
            dt += timedelta(microseconds=subsec * 1000)
        else:
            dt += timedelta(microseconds=subsec * 10000)
        return FrameMeta(path=path, camera_id=_camera_id_from_exif(named, path), captured_at=dt)
    except Exception:
        return None


# ----------------------------------------------------------------- clustering


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity. Assumes the encoder already L2-normalized its output."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cluster_bursts(
    metas: Sequence[FrameMeta],
    *,
    embeddings: Optional[Dict[Path, np.ndarray]] = None,
    short_gap_seconds: float = 0.6,
    long_gap_seconds: float = 8.0,
    similarity_threshold: float = 0.92,
    gap_seconds_fallback: float = 2.5,
) -> List[List[FrameMeta]]:
    """Camera-strict, time + (optional) embedding-similarity aware burst clustering.

    If ``embeddings`` is None, falls back to time-only with ``gap_seconds_fallback``.
    """
    if not metas:
        return []
    ordered = sorted(metas, key=lambda m: (m.camera_id, m.captured_at))
    clusters: List[List[FrameMeta]] = []
    current: List[FrameMeta] = [ordered[0]]

    for m in ordered[1:]:
        prev = current[-1]
        if m.camera_id != prev.camera_id:
            clusters.append(current)
            current = [m]
            continue

        dt = (m.captured_at - prev.captured_at).total_seconds()
        if embeddings is None:
            same_burst = dt <= gap_seconds_fallback
        elif dt <= short_gap_seconds:
            same_burst = True
        elif dt <= long_gap_seconds:
            a = embeddings.get(prev.path)
            b = embeddings.get(m.path)
            if a is None or b is None:
                # Missing embedding: be conservative — fall back to short-gap rule.
                same_burst = dt <= short_gap_seconds
            else:
                sim = _cosine(a, b)
                same_burst = sim >= similarity_threshold
        else:
            same_burst = False

        if same_burst:
            current.append(m)
        else:
            clusters.append(current)
            current = [m]
    if current:
        clusters.append(current)
    return clusters


def assign_burst_ids(
    paths: Sequence[Path],
    *,
    embeddings: Optional[Dict[Path, np.ndarray]] = None,
    short_gap_seconds: float = 0.6,
    long_gap_seconds: float = 8.0,
    similarity_threshold: float = 0.92,
    gap_seconds_fallback: float = 2.5,
) -> Dict[Path, int]:
    metas: List[FrameMeta] = []
    no_exif: List[Path] = []
    for p in paths:
        m = read_frame_meta(p)
        if m is None:
            no_exif.append(p)
        else:
            metas.append(m)
    clusters = cluster_bursts(
        metas,
        embeddings=embeddings,
        short_gap_seconds=short_gap_seconds,
        long_gap_seconds=long_gap_seconds,
        similarity_threshold=similarity_threshold,
        gap_seconds_fallback=gap_seconds_fallback,
    )
    out: Dict[Path, int] = {}
    bid = 0
    for c in clusters:
        for m in c:
            out[m.path] = bid
        bid += 1
    for p in no_exif:
        out[p] = bid
        bid += 1
    return out


def summarize_bursts(burst_ids: Dict[Path, int]) -> Tuple[int, int, int]:
    from collections import Counter
    sizes = Counter(burst_ids.values())
    n_bursts = len(sizes)
    n_singleton = sum(1 for s in sizes.values() if s == 1)
    return n_bursts, n_bursts - n_singleton, n_singleton
