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
from pathlib import Path

from PIL import Image

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


def validated_images(paths, run_dir):
    """Decode each unchanged source only once, including during live updates."""
    import json
    from .pipeline import atomic_write
    cache_path = run_dir / "validated.json"
    try:
        previous = json.loads(cache_path.read_text())
    except (OSError, ValueError):
        previous = {}
    valid, signatures = [], {}
    for p in paths:
        try:
            st = p.stat()
            signature = [st.st_size, st.st_mtime_ns]
            if previous.get(str(p)) != signature and not is_decodable(p):
                continue
            after = p.stat()
            if signature != [after.st_size, after.st_mtime_ns]:
                continue
        except OSError:
            continue
        valid.append(p)
        signatures[str(p)] = signature
    atomic_write(cache_path, lambda tmp: tmp.write_text(json.dumps(signatures)))
    return valid
