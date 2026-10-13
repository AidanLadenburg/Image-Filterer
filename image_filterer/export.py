"""Server-side export: copy picked originals into a folder the server can write.

Built for a shared cloud folder (a LucidLink mount): anyone in the viewer —
phone, iPad, Safari — can send picks to the team's share, without the browser
needing the File System Access API or the files round-tripping through it.

Two properties matter more than speed:

- **Atomic appearance.** Each file is written as a hidden ``.<name>.partial``
  and renamed into place, so the editors' Lucid clients (and our own hot folder,
  if it watches the same share) never see half a photo.
- **Never ingested back.** The export root carries :data:`EXPORT_MARKER`; the
  hot folder skips any directory holding it, so picks written into the watched
  share don't come back as duplicate frames.
"""

from __future__ import annotations

import filecmp
import os
import shutil
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

EXPORT_MARKER = ".image-filterer-export"

ProgressCB = Callable[[int, int], None]  # (done, total)


def ensure_export_root(root: Path) -> Path:
    """Check the export root is there and drop the hot-folder marker in it.

    Never creates it: the folder is chosen in Settings, where it must already
    exist. If it's missing now, its share is unmounted (Lucid dropped) — and
    creating it would put "exports" on this machine's local disk, inside the
    bare mount point, where they'd also stop Lucid from mounting again. The
    callers report the error and auto-export retries on its next pass."""
    root = Path(root)
    if not root.is_dir():
        import errno
        raise FileNotFoundError(errno.ENOENT, "export folder not found (is its share mounted?)", str(root))
    marker = root / EXPORT_MARKER
    if not marker.exists():
        marker.write_text("Exports from Image Filterer. The hot folder never ingests this directory.\n")
    return root


def mark_export_dir(path: Path) -> Path:
    """Create ``path`` and mark it so no ingest ever reads it as a photo source.
    Every folder we write exports into gets one, not just the export root: a
    run can sit *below* the export root and never walk through it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    marker = path / EXPORT_MARKER
    if not marker.exists():
        marker.write_text("Exports from Image Filterer. Photo ingest never reads this directory.\n")
    return path


def source_root(paths: Sequence[str]) -> Optional[Path]:
    """The folder a run's photos came from (their deepest common folder)."""
    parents = [str(Path(p).parent) for p in paths]
    if not parents:
        return None
    try:
        return Path(os.path.commonpath(parents))
    except ValueError:
        return None


def export_overlaps_source(dest: Path, source: Optional[Path]) -> bool:
    """True if writing into ``dest`` would put files inside the run's own photo
    folder — they'd be ingested back as new photos and exported again."""
    if source is None:
        return False
    try:
        d, s = Path(dest).resolve(), Path(source).resolve()
    except OSError:
        return False
    return d == s or s in d.parents


def _destination(dest_dir: Path, src: Path, name: Optional[str] = None) -> Tuple[Path, bool]:
    """Where ``src`` should land, and whether an identical copy is already there.

    Frames from different cameras or subfolders sometimes share a filename, so a
    name clash with *different* content gets a numbered suffix instead of
    overwriting — the same rule the browser-side export uses.
    """
    base = Path(name or src.name)
    stem, suffix = base.stem, base.suffix
    candidate, n = dest_dir / base.name, 1
    while candidate.exists():
        if filecmp.cmp(src, candidate, shallow=False):
            return candidate, True
        n += 1
        candidate = dest_dir / f"{stem}-{n}{suffix}"
    return candidate, False


def copy_atomic(src: Path, dest_dir: Path, name: Optional[str] = None) -> str:
    """Copy ``src`` into ``dest_dir`` (as ``name``) so it appears all at once.
    Returns the final file name — an existing identical copy is reused."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dst, identical = _destination(dest_dir, src, name)
    if identical:
        return dst.name
    tmp = dest_dir / f".{dst.name}.partial"
    try:
        shutil.copyfile(src, tmp)
        shutil.copystat(src, tmp)          # keep capture mtime for editors' sort order
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return dst.name


def export_files(
    paths: Sequence[Path],
    dest_dir: Path,
    *,
    progress_cb: Optional[ProgressCB] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Dict[str, object]:
    """Copy ``paths`` into ``dest_dir``. One bad file never aborts the rest."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    written = skipped = failed = 0
    last_error = ""
    total = len(paths)
    for i, src in enumerate(paths):
        if should_stop and should_stop():
            break
        src = Path(src)
        try:
            if _destination(dest_dir, src)[1]:
                skipped += 1
            else:
                copy_atomic(src, dest_dir)
                written += 1
        except OSError as exc:
            failed += 1
            last_error = f"{src.name}: {exc.strerror or exc}"
        if progress_cb:
            progress_cb(i + 1, total)
    return {"written": written, "skipped": skipped, "failed": failed,
            "total": total, "last_error": last_error}
