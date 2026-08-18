"""Watch a folder and fold new photos into an existing run as they arrive.

Built for a live event: the photo team offloads cards into a shared folder (a
LucidLink mount, a NAS, anything the server can read) and the shoot in the
viewer grows through the day without anyone pressing a button.

Why polling rather than filesystem events
-----------------------------------------
``inotify`` reports writes that pass through *this* machine's kernel. On a
network or cloud filesystem the photographer's bytes are written on *their*
machine; ours only learns about the file when the client syncs metadata, and no
local write ever happens — so no event fires. Polling is the only thing that
works for the case we actually care about.

Polling is also better on its own merits here: events fire on file *creation*,
not completion, so a stability check is needed regardless; and a dropped event
(inotify queues do overflow when 300 files land at once) means a photo silently
missing forever, whereas a poll simply notices it next time.

The two loops
-------------
Scanning and ingesting run at different rates on purpose. A scan is cheap
(``scandir`` plus a ``stat`` on files we haven't accounted for yet) so it runs
often and keeps an up-to-date set of files known to have finished copying.
Ingesting is expensive, so it runs only when there is something to do, the GPU
is free, and a cooldown has elapsed — which naturally batches a flood of 300
files into one pass instead of 300.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

from .config import IMAGE_EXTS


@dataclass
class _Pending:
    """A file seen but not yet confirmed finished."""
    size: int
    mtime_ns: int
    stable_ticks: int = 0


@dataclass
class HotFolderStatus:
    watching: bool = False
    folder: str = ""
    run_id: Optional[int] = None
    last_scan: float = 0.0
    scan_interval: float = 0.0
    n_ready: int = 0          # finished copying, waiting for the next batch
    n_pending: int = 0        # still arriving
    n_ingested: int = 0       # folded into the run so far, this session
    n_unusable: int = 0       # tried and can't be read (RAW without preview, junk)
    last_ingest: float = 0.0
    last_error: str = ""
    idle_seconds: float = 0.0

    def as_dict(self) -> Dict[str, object]:
        return dict(self.__dict__)


class HotFolderWatcher:
    """Polls one folder and calls ``ingest(paths)`` when a batch is ready.

    ``ingest`` is supplied by the caller and is responsible for the ingest lock,
    the run registry and publishing — this class only decides *when* and *what*.
    It must return True if the batch was ingested, False if it was declined (for
    example because a manual upload holds the lock), in which case the files stay
    queued for the next attempt.
    """

    def __init__(
        self,
        folder: Path,
        run_id: int,
        ingest: Callable[[List[Path]], bool],
        *,
        known: Optional[Set[str]] = None,
        scan_interval: float = 5.0,
        idle_scan_interval: float = 30.0,
        idle_after: float = 600.0,       # quiet this long → slow the scan down
        stable_ticks: int = 2,           # unchanged for this many scans → finished
        batch_cooldown: float = 20.0,    # min gap between ingests
        stop_after_idle: float = 6 * 3600.0,  # give up on a forgotten watch
        decodable: Optional[Callable[[Path], bool]] = None,
    ) -> None:
        self.folder = Path(folder)
        self.run_id = run_id
        self._ingest = ingest
        self.scan_interval = scan_interval
        self.idle_scan_interval = idle_scan_interval
        self.idle_after = idle_after
        self.stable_ticks = stable_ticks
        self.batch_cooldown = batch_cooldown
        self.stop_after_idle = stop_after_idle
        if decodable is None:
            from .imaging import is_decodable
            decodable = is_decodable
        self._decodable = decodable

        self._exts = {e.lower() for e in IMAGE_EXTS}
        self._known: Set[str] = set(known or ())   # already part of the run
        self._pending: Dict[str, _Pending] = {}
        self._ready: Set[str] = set()
        # Files we've tried and can't use. Remembered so a folder full of XMP
        # sidecars or unreadable RAW isn't re-examined every five seconds forever.
        self._unusable: Set[str] = set()

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._status = HotFolderStatus(watching=False, folder=str(folder), run_id=run_id)
        self._last_new = time.time()
        self._last_ingest = 0.0
        self._n_ingested = 0

    # ----------------------------------------------------------------- control

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="hotfolder", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=2.0)
        with self._lock:
            self._status.watching = False

    @property
    def alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive() and not self._stop.is_set())

    def status(self) -> Dict[str, object]:
        with self._lock:
            st = HotFolderStatus(**self._status.as_dict())
        st.watching = self.alive
        st.n_ready = len(self._ready)
        st.n_pending = len(self._pending)
        st.n_unusable = len(self._unusable)
        st.n_ingested = self._n_ingested
        st.last_ingest = self._last_ingest
        st.idle_seconds = time.time() - self._last_new
        st.scan_interval = self._current_interval()
        return st.as_dict()

    # ------------------------------------------------------------------- loop

    def _current_interval(self) -> float:
        quiet = time.time() - self._last_new
        return self.idle_scan_interval if quiet > self.idle_after else self.scan_interval

    def _loop(self) -> None:
        with self._lock:
            self._status.watching = True
        while not self._stop.is_set():
            try:
                self._scan()
                self._maybe_ingest()
            except Exception as exc:  # noqa: BLE001 — a watcher must not die on one bad pass
                with self._lock:
                    self._status.last_error = f"{type(exc).__name__}: {exc}"
            if time.time() - self._last_new > self.stop_after_idle:
                with self._lock:
                    self._status.last_error = "stopped: no new photos for a long time"
                break
            self._stop.wait(self._current_interval())
        with self._lock:
            self._status.watching = False

    def _scan(self) -> None:
        """Cheap pass: find new names, and decide which have finished copying."""
        try:
            entries = list(self.folder.iterdir())
        except OSError as exc:
            with self._lock:
                self._status.last_error = f"cannot read folder: {exc}"
            return

        seen_now: Set[str] = set()
        for e in entries:
            key = str(e)
            if e.suffix.lower() not in self._exts:
                continue                      # not an image type we handle
            if key in self._known or key in self._ready or key in self._unusable:
                continue                      # already accounted for
            seen_now.add(key)
            try:
                st = e.stat()
            except OSError:
                continue                      # vanished mid-scan
            prev = self._pending.get(key)
            if prev is None:
                self._pending[key] = _Pending(st.st_size, st.st_mtime_ns)
                # Wake up on FIRST SIGHT, not on readiness. Otherwise a watcher
                # that has backed off to 30 s also does its stability checks at
                # 30 s, so the first photo after a lull takes minutes to appear.
                self._last_new = time.time()
                continue
            if prev.size == st.st_size and prev.mtime_ns == st.st_mtime_ns:
                prev.stable_ticks += 1
            else:                              # still growing — restart the count
                prev.size, prev.mtime_ns, prev.stable_ticks = st.st_size, st.st_mtime_ns, 0
                continue
            if prev.stable_ticks >= self.stable_ticks:
                # Size settled AND it actually opens. The second test is what
                # catches a file that stopped growing because the copy failed.
                if self._decodable(Path(key)):
                    self._ready.add(key)
                    self._last_new = time.time()
                else:
                    self._unusable.add(key)
                self._pending.pop(key, None)

        # Forget pending entries whose file disappeared.
        for gone in [k for k in self._pending if k not in seen_now]:
            self._pending.pop(gone, None)
        with self._lock:
            self._status.last_scan = time.time()

    def _maybe_ingest(self) -> None:
        if not self._ready:
            return
        if time.time() - self._last_ingest < self.batch_cooldown:
            return
        batch = sorted(self._ready)
        accepted = self._ingest([Path(p) for p in batch])
        if not accepted:
            return          # GPU busy (a manual upload won); try again next tick
        self._last_ingest = time.time()
        self._n_ingested += len(batch)
        self._known.update(batch)
        self._ready.difference_update(batch)
