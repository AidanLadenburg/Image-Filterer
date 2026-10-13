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

Within a single camera id a frame joins the current burst ("moment") when it's
within short_gap_seconds of the previous frame, or within long_gap_seconds and
similar to the moment's recent frames and to its first frame (see
:func:`cluster_bursts`).

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


def _raw_exif(path: Path) -> Dict[str, object]:
    """EXIF for a RAW file, via pyexiv2.

    The embedded preview we decode RAW through does NOT carry
    DateTimeOriginal, so reading the container itself is the only way to place a
    RAW frame on the timeline. Without pyexiv2 we return nothing and the frame
    becomes its own single-frame burst rather than being lost.
    """
    if not _HAS_PYEXIV2:
        return {}
    try:
        img = pyexiv2.Image(str(path))
        e = img.read_exif()
        img.close()
    except Exception:
        return {}
    out: Dict[str, object] = {}
    for tag, name in (("Exif.Photo.DateTimeOriginal", "DateTimeOriginal"),
                      ("Exif.Photo.SubSecTimeOriginal", "SubsecTimeOriginal"),
                      ("Exif.Image.Make", "Make"),
                      ("Exif.Image.Model", "Model"),
                      ("Exif.Image.BodySerialNumber", "BodySerialNumber"),
                      ("Exif.Photo.BodySerialNumber", "BodySerialNumber"),
                      ("Exif.Photo.LensModel", "LensModel")):
        if tag in e and e[tag] and name not in out:
            out[name] = e[tag]
    return out


_NO_META = object()


def read_frame_meta(path: Path) -> Optional[FrameMeta]:
    """Capture time and camera from EXIF. Memoized in the process-wide cache
    by (path, size, mtime): every ingest of a live run re-reads all of them."""
    from .cache_io import MEMORY
    try:
        st = Path(path).stat()
        key = ("frame_meta", str(path), st.st_size, st.st_mtime_ns)
    except OSError:
        return _read_frame_meta(path)
    hit = MEMORY.get(key)
    if hit is not None:
        return None if hit is _NO_META else FrameMeta(path=path, camera_id=hit.camera_id, captured_at=hit.captured_at)
    meta = _read_frame_meta(path)
    MEMORY.put(key, _NO_META if meta is None else meta, 600)
    return meta


def _read_frame_meta(path: Path) -> Optional[FrameMeta]:
    try:
        from .imaging import is_raw
        if is_raw(path):
            named = _raw_exif(path)
        else:
            with Image.open(path) as im:
                exif = im._getexif() or {}
            named = {ExifTags.TAGS.get(k, str(k)): v for k, v in exif.items()}
        dt_str = named.get("DateTimeOriginal")
        if not dt_str:
            return None
        dt = datetime.strptime(str(dt_str), "%Y:%m:%d %H:%M:%S")
        subsec_raw = str(named.get("SubsecTimeOriginal", "0")).strip()
        if subsec_raw.isdigit():
            dt += timedelta(microseconds=int(subsec_raw[:6].ljust(6, "0")))
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
    long_gap_seconds: float = 30.0,
    similarity_threshold: float = 0.91,
    gap_seconds_fallback: float = 2.5,
    lookback: int = 3,
    max_span_seconds: float = 45.0,
    anchor_threshold: float = 0.89,
) -> List[List[FrameMeta]]:
    """Camera-strict, time + embedding-similarity aware burst ("moment") clustering.

    Within one camera, in time order, a frame joins the current moment when
      dt <= short_gap_seconds                                     (always), or
      dt <= long_gap_seconds and the moment is <= max_span_seconds long and
           the frame is similar (>= similarity_threshold) to any of the
           moment's last ``lookback`` frames and (>= anchor_threshold) to its
           first frame.
    If ``embeddings`` is None, falls back to time-only with ``gap_seconds_fallback``.
    """
    if not metas:
        return []
    ordered = sorted(metas, key=lambda m: (m.camera_id, m.captured_at))
    clusters: List[List[FrameMeta]] = []
    current: List[FrameMeta] = [ordered[0]]

    def emb(m):
        return embeddings.get(m.path) if embeddings is not None else None

    for m in ordered[1:]:
        prev = current[-1]
        if m.camera_id != prev.camera_id:
            clusters.append(current)
            current = [m]
            continue

        dt = (m.captured_at - prev.captured_at).total_seconds()
        span = (m.captured_at - current[0].captured_at).total_seconds()
        if embeddings is None:
            same_burst = dt <= gap_seconds_fallback
        elif dt <= short_gap_seconds:
            same_burst = True
        elif dt <= long_gap_seconds and span <= max_span_seconds:
            a = emb(m)
            recent = [emb(x) for x in current[-lookback:]]
            first = emb(current[0])
            if a is None or any(r is None for r in recent) or first is None:
                same_burst = False            # missing embedding: be conservative
            else:
                same_burst = (max(_cosine(a, r) for r in recent) >= similarity_threshold
                              and _cosine(a, first) >= anchor_threshold)
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
    long_gap_seconds: float = 30.0,
    similarity_threshold: float = 0.91,
    gap_seconds_fallback: float = 2.5,
    lookback: int = 3,
    max_span_seconds: float = 45.0,
    anchor_threshold: float = 0.89,
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
        lookback=lookback,
        max_span_seconds=max_span_seconds,
        anchor_threshold=anchor_threshold,
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
