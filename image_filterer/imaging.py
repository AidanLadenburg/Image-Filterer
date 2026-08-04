"""Image-file hygiene for ingestion: repair a known upload corruption and detect
files that can't be decoded so a bad file never crashes a whole run."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

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
    """
    p = Path(path)
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
    """True if PIL can open + parse the file's structure (cheap; no full decode)."""
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False
