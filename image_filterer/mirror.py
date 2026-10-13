"""Local mirror for photos that live on a network/cloud folder.

Ingest touches every new photo about five times — the watcher's decode check,
validation, the content hash, feature extraction, person detection. On a local
disk the page cache makes repeats free. On a LucidLink share reached through
SMB there's no client-side caching between opens, so each touch is another trip
through the (x86-emulated) Lucid VM: on the GTC test the app read 10.6 GB to
process 1.4 GB of photos, at ~1.5 MB/s effective.

So a photo on a network folder is copied **once** to a mirror on the local SSD,
and everything downstream — ingest, thumbnails, downloads, exports — reads the
local copy. Local folders are never mirrored.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

# Filesystems where a read is a network round trip (or worse).
NETWORK_FS = {"cifs", "smb3", "smbfs", "nfs", "nfs4", "9p", "afs", "ceph", "glusterfs",
              "davfs", "sshfs", "fuse.sshfs", "fuse.rclone", "fuse.lucid", "fuse.s3fs"}


def filesystem_type(path: Path) -> Optional[str]:
    """fstype of the mount holding ``path`` (Linux; None elsewhere)."""
    try:
        target = str(Path(path).resolve())
        best, fstype = "", None
        with open("/proc/mounts") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mnt = parts[1].replace("\\040", " ")
                if parts[2] != "autofs" and (target == mnt or target.startswith(mnt.rstrip("/") + "/")) \
                        and len(mnt) >= len(best):
                    best, fstype = mnt, parts[2]      # later lines are mounted on top (autofs → cifs)
        return fstype
    except OSError:
        return None


def is_network_path(path: Path) -> bool:
    fs = filesystem_type(path) or ""
    return fs in NETWORK_FS or (fs.startswith("fuse.") and fs != "fuse.portal")


def mirror_copy(src: Path, dst: Path) -> Path:
    """Copy ``src`` to ``dst`` atomically, keeping its mtime (capture-time
    fallback, and so an unchanged source is recognised next time). An existing
    copy with the same size and mtime is reused without reading the source."""
    st = src.stat()
    try:
        d = dst.stat()
        if d.st_size == st.st_size and d.st_mtime_ns == st.st_mtime_ns:
            return dst
    except OSError:
        pass
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.partial")
    try:
        shutil.copyfile(src, tmp)
        os.utime(tmp, ns=(st.st_atime_ns, st.st_mtime_ns))
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return dst


def _mount_table():
    try:
        with open("/proc/mounts") as fh:
            return [line.split() for line in fh if len(line.split()) >= 3]
    except OSError:
        return []


class ChannelPicker:
    """Spread reads from a network share over every mount of that same share.

    Samba serves each client connection from one process, which runs on one
    virtual CPU of the Lucid VM: one mount topped out at ~13 MB/s while three
    of the VM's four CPUs idled. Mounting the share several more times with
    separate connections (cifs ``nosharesock``) gives each mount its own smbd;
    this finds those mounts (same device string, same filesystem type) and
    hands them out round-robin. With a single mount it's a no-op.
    """

    def __init__(self, folder: Path):
        import itertools
        import threading
        self._lock = threading.Lock()
        self._next = itertools.count()
        self.mount, self.roots = None, []
        try:
            target = str(Path(folder).resolve())
        except OSError:
            return
        table = _mount_table()
        best = None
        for dev, mnt, fs, *_ in table:
            mnt = mnt.replace("\\040", " ")
            if fs == "autofs":
                continue                       # the automount placeholder, not the share
            if (target == mnt or target.startswith(mnt.rstrip("/") + "/")) and \
                    (best is None or len(mnt) >= len(best[1])):
                best = (dev, mnt, fs)
        if best is None or best[2] not in NETWORK_FS:
            return
        dev, mnt, fs = best
        self.mount = Path(mnt)
        self.roots = sorted({Path(m.replace("\\040", " ")) for d, m, f, *_ in table if d == dev and f == fs},
                            key=lambda p: (p != self.mount, str(p)))

    def source_for(self, path: Path) -> Path:
        """``path`` re-rooted on the next mount of the same share."""
        if len(self.roots) < 2 or self.mount is None:
            return path
        with self._lock:
            root = self.roots[next(self._next) % len(self.roots)]
        try:
            return root / Path(path).relative_to(self.mount)
        except ValueError:
            return path
