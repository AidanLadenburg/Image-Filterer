"""Image-file hygiene for ingestion: open any supported file as a PIL image,
repair a known upload corruption, and detect files that can't be decoded so a
bad file never crashes a whole run.

RAW files are opened through their **embedded JPEG preview** rather than by
demosaicing the sensor data. The preview is the camera's own rendering — the
image the photographer judged on the back of the body — it is full resolution on
modern bodies, and it costs ~10 ms to pull instead of the 1-3 s a full demosaic
takes. Nothing downstream needs more: the encoders see a 384 px input and the
detectors work happily at preview scale.
"""

from __future__ import annotations

import io
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, TypeVar

from PIL import Image

T = TypeVar("T")
R = TypeVar("R")


def cpu_workers() -> int:
    """Threads for per-image CPU work (decode, face detection).

    Threads rather than processes: PIL's JPEG decoder, OpenCV and rawpy all
    release the GIL, so decoding genuinely runs in parallel — and threads can
    hand decoded images to the GPU stage without pickling 25 MP arrays.
    Override with IMAGE_FILTERER_CPU_WORKERS (1 = the old serial behaviour).
    """
    env = os.environ.get("IMAGE_FILTERER_CPU_WORKERS", "").strip()
    if env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, min(12, (os.cpu_count() or 2) - 2))


def ordered_map(fn: Callable[[T], R], items: Iterable[T], workers: Optional[int] = None,
                lookahead: Optional[int] = None) -> Iterator[R]:
    """``map(fn, items)`` on a thread pool: results in input order, with at most
    ``lookahead`` in flight so decoded full-resolution images can't pile up in
    memory faster than the consumer (usually the GPU) drains them."""
    workers = workers or cpu_workers()
    if workers <= 1:
        yield from map(fn, items)
        return
    lookahead = lookahead or workers * 2
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="img") as pool:
        pending: deque = deque()
        for item in items:
            pending.append(pool.submit(fn, item))
            if len(pending) >= lookahead:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()

# Extensions handled via the RAW path. Decoding needs `rawpy`; without it these
# files are reported undecodable and skipped, exactly as they were before.
RAW_EXTS = {".cr2", ".cr3", ".arw", ".nef", ".raf", ".orf", ".rw2", ".dng"}


def is_raw(path) -> bool:
    return Path(path).suffix.lower() in RAW_EXTS


def open_image(path) -> Image.Image:
    """Open any supported file as an RGB PIL image. Use this, not Image.open().

    Raises whatever the underlying reader raises, so callers can treat failure
    the same way they always have.
    """
    p = Path(path)
    if not is_raw(p):
        return Image.open(p)
    import rawpy  # imported lazily: only RAW shoots need it installed

    with rawpy.imread(str(p)) as raw:
        try:
            thumb = raw.extract_thumb()
        except Exception:  # noqa: BLE001 — no embedded preview; fall back below
            thumb = None
        if thumb is not None and thumb.format == rawpy.ThumbFormat.JPEG:
            return Image.open(io.BytesIO(thumb.data))
        if thumb is not None:                       # already a bitmap
            return Image.fromarray(thumb.data)
        # No usable preview — demosaic at half size. Slow, but it beats losing
        # the frame entirely.
        return Image.fromarray(raw.postprocess(half_size=True))

# Leading signatures of formats PIL can decode. Used to (a) detect and strip a
# few bytes of junk accidentally prepended to an uploaded file, and (b) sanity
# checks. NOT exhaustive — decodability is confirmed by actually opening.
IMAGE_MAGICS = (
    b"\xff\xd8\xff",            # JPEG
    b"\x89PNG\r\n\x1a\n",       # PNG
    b"GIF87a", b"GIF89a",        # GIF
    b"BM",                       # BMP
    b"II*\x00", b"MM\x00*",      # TIFF (little / big endian)
    b"RIFF",                     # WEBP (RIFF....WEBP)
)


def sanitize_image_file(path, max_lead: int = 8) -> bool:
    """Repair a file that has a few stray bytes before its real image header.

    Some upload paths (certain browser/proxy combinations) prepend a multipart
    separator — a leading ``\\r\\n`` — to the file body, so a valid JPEG arrives as
    ``\\r\\n\\xff\\xd8...`` and no longer decodes. If the true magic appears within
    the first ``max_lead`` bytes, strip the junk in place (restoring the original
    bytes, so the sha1 matches the cache again). Returns True if it repaired.

    Only ever called on *uploaded copies*, never on server-path originals.
    RAW containers are left strictly alone: they don't carry any of the magics
    below, so a chance byte match inside one could only ever truncate a good file.
    """
    p = Path(path)
    if is_raw(p):
        return False
    try:
        with open(p, "rb") as f:
            head = f.read(max_lead + 8)
    except OSError:
        return False
    if any(head.startswith(m) for m in IMAGE_MAGICS):
        return False  # already well-formed
    best = None
    for m in IMAGE_MAGICS:
        idx = head.find(m)
        if idx > 0 and (best is None or idx < best):
            best = idx
    if best is not None and best <= max_lead:
        data = p.read_bytes()
        p.write_bytes(data[best:])
        return True
    return False


def is_decodable(path) -> bool:
    """True only if the pixel data can actually be decoded.

    For RAW this pulls the embedded preview, which is also the honest test: a
    RAW whose preview can't be read is one we can't ingest.
    """
    try:
        with open_image(path) as im:
            im.load()
        return True
    except Exception:
        return False


def validated_images(paths, run_dir, prevalidated=None, stats=None):
    """Decode each unchanged source only once, including during live updates.
    ``prevalidated``: {path: [size, mtime_ns]} the caller already decoded (the
    hot-folder watcher checks every arrival) — not decoded a second time."""
    import json
    from .pipeline import atomic_write
    cache_path = run_dir / "validated.json"
    try:
        previous = json.loads(cache_path.read_text())
    except (OSError, ValueError):
        previous = {}
    def check(p):
        """(path, signature) if usable, else None. Decodes only new/changed files."""
        try:
            st = p.stat()
            signature = [st.st_size, st.st_mtime_ns]
            if previous.get(str(p)) != signature and (prevalidated or {}).get(str(p)) != signature:
                decoded.append(1)
                if not is_decodable(p):
                    return None
            after = p.stat()
            if signature != [after.st_size, after.st_mtime_ns]:
                return None
        except OSError:
            return None
        return p, signature

    valid, signatures, decoded = [], {}, []
    for res in ordered_map(check, paths):          # full decodes, in parallel
        if res is not None:
            valid.append(res[0])
            signatures[str(res[0])] = res[1]
    atomic_write(cache_path, lambda tmp: tmp.write_text(json.dumps(signatures)))
    if stats is not None:
        stats["decoded"] = len(decoded)
    return valid
