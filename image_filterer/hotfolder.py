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

import os
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
    decode_failures: int = 0
    first_decode_failure: float = 0.0
    retry_at: float = 0.0
    grew: bool = False          # seen changing: wait for an unchanged scan before checking


@dataclass
class HotFolderStatus:
    watching: bool = False
    folder: str = ""
    run_id: Optional[int] = None
    last_scan: float = 0.0
    scan_interval: float = 0.0
    n_ready: int = 0          # finished copying, waiting for the next batch
    n_pending: int = 0        # still arriving
    n_waiting_data: int = 0   # of n_pending: listed at full size but not decodable yet
    n_ingested: int = 0       # folded into the run so far, this session
    n_unusable: int = 0       # tried and can't be read (RAW without preview, junk)
    last_ingest: float = 0.0
    last_error: str = ""
    idle_seconds: float = 0.0
    # Where the time goes (for diagnosing a slow share):
    last_scan_s: float = 0.0          # listing + stat of the folder
    last_check_n: int = 0             # files copied/decode-checked in the last round
    last_check_s: float = 0.0
    last_check_mb_s: float = 0.0      # throughput of that round (bytes read from the share)
    last_ingest_s: float = 0.0        # duration of the last ingest batch
    copied_mb: float = 0.0            # total read from the share this session
    total_check_s: float = 0.0        # all copy + decode-check rounds this session
    total_ingest_s: float = 0.0       # all ingest batches this session
    n_ingests: int = 0

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
        stable_ticks: int = 0,           # unchanged scans before the decode check (0: on first sight)
        stable_ticks_raw: int = 1,       # RAW: a half-copied CR3 can pass (its preview comes first)
        batch_cooldown: float = 20.0,    # min gap between ingests
        stop_after_idle: float = 6 * 3600.0,  # give up on a forgotten watch
        decodable: Optional[Callable[[Path], bool]] = None,
        decode_retry_for: float = 1800.0,     # keep retrying an undecodable file this long
        decode_retry_max_interval: float = 60.0,
        decode_budget: float = 15.0,          # seconds of decode checks per scan before ingesting
        mirror_dir: Optional[Path] = None,    # network folder → copy each photo here once (see mirror.py)
        full_check_every: float = 60.0,       # re-stat already-handled files this often
        on_stopped: Optional[Callable[[], None]] = None,   # stopped on purpose (user, or idle) — not a crash
    ) -> None:
        self._on_stopped = on_stopped
        self.folder = Path(folder)
        self.run_id = run_id
        self._ingest = ingest
        self.scan_interval = scan_interval
        self.idle_scan_interval = idle_scan_interval
        self.idle_after = idle_after
        self.stable_ticks = stable_ticks
        self.stable_ticks_raw = stable_ticks_raw
        # {ingest path: [size, mtime_ns]} of every file that passed the decode
        # check, so ingest's own validation doesn't decode it a second time.
        self._validated: Dict[str, list] = {}
        self.batch_cooldown = batch_cooldown
        self.stop_after_idle = stop_after_idle
        if decodable is None:
            from .imaging import is_decodable
            decodable = is_decodable
        self._decodable = decodable
        self.decode_retry_for = decode_retry_for
        self.decode_retry_max_interval = decode_retry_max_interval
        self.decode_budget = decode_budget
        self.mirror_dir = Path(mirror_dir) if mirror_dir else None
        from .mirror import ChannelPicker
        self._channels = ChannelPicker(self.folder)       # no-op unless the share is mounted several times
        self.full_check_every = full_check_every
        self._last_full = 0.0
        # Scanning/copying and ingesting run on separate threads (copying the
        # next photos while a batch is processed); this guards the file sets.
        self._sets = threading.RLock()
        self._inflight: Optional[List[Path]] = None
        self._m_scan_s = self._m_check_s = self._m_check_mb_s = self._m_ingest_s = 0.0
        self._m_check_n = 0
        self._m_bytes = 0
        self._m_total_check_s = self._m_total_ingest_s = 0.0
        self._m_n_ingests = 0
        self._ingest_thread: Optional[threading.Thread] = None

        self._exts = {e.lower() for e in IMAGE_EXTS}
        self._known: Set[str] = set(known or ())   # already part of the run
        self._pending: Dict[str, _Pending] = {}
        self._ready: Set[str] = set()
        # Files we've tried and can't use. Remembered so a folder full of XMP
        # sidecars or unreadable RAW isn't re-examined every five seconds forever.
        self._unusable: Set[str] = set()
        self._observed: Dict[str, tuple] = {}

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._status = HotFolderStatus(watching=False, folder=str(folder), run_id=run_id)
        self._last_new = time.time()
        self._last_ingest = 0.0
        self._last_attempt = 0.0
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
        it = self._ingest_thread
        if it and it.is_alive() and it is not threading.current_thread():
            it.join(timeout=2.0)
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
        st.n_waiting_data = sum(1 for p in self._pending.values() if p.decode_failures)
        st.n_unusable = len(self._unusable)
        st.n_ingested = self._n_ingested
        st.last_ingest = self._last_ingest
        st.idle_seconds = time.time() - self._last_new
        st.scan_interval = self._current_interval()
        st.last_scan_s, st.last_check_n, st.last_check_s = self._m_scan_s, self._m_check_n, self._m_check_s
        st.last_check_mb_s, st.last_ingest_s = self._m_check_mb_s, self._m_ingest_s
        st.copied_mb = self._m_bytes / 1e6
        st.total_check_s, st.total_ingest_s, st.n_ingests = self._m_total_check_s, self._m_total_ingest_s, self._m_n_ingests
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
                self._start_ingest()
            except Exception as exc:  # noqa: BLE001 — a watcher must not die on one bad pass
                with self._lock:
                    self._status.last_error = f"{type(exc).__name__}: {exc}"
            if time.time() - self._last_new > self.stop_after_idle:
                with self._lock:
                    self._status.last_error = "stopped: no new photos for a long time"
                self._stop.set()                  # deliberate: not reported as a crashed watch
                break
            self._stop.wait(self._current_interval())
        with self._lock:
            self._status.watching = False
        if self._on_stopped is not None:
            try:
                self._on_stopped()
            except Exception:  # noqa: BLE001
                pass

    def _scan(self) -> None:
        """Cheap pass: find new names, and decide which have finished copying."""
        try:
            entries = self._list_files()
        except OSError as exc:
            with self._lock:
                self._status.last_error = f"cannot read folder: {exc}"
            return
        with self._lock:                       # readable again (e.g. Lucid back): clear the stale error
            if self._status.last_error.startswith("cannot read folder"):
                self._status.last_error = ""

        seen_now: Set[str] = set()
        due: List[str] = []                    # stable files whose decode check is due
        now = time.time()
        # Already-handled files are only re-stat'ed in a periodic full sweep: on
        # a cloud share every stat is a round trip, and stat-ing the whole shoot
        # every few seconds grew with it (1–5 s per scan at 600 photos on Lucid).
        full = now - self._last_full >= self.full_check_every
        if full:
            self._last_full = now
        self._sets.acquire()
        try:
            self._scan_entries(entries, seen_now, due, full)
        finally:
            self._sets.release()
        self._m_scan_s = time.time() - now
        self._check_decodable(sorted(due))
        with self._sets:
            self._forget_missing(seen_now)
        with self._lock:
            self._status.last_scan = time.time()

    def _scan_entries(self, entries, seen_now: Set[str], due: List[str], full: bool) -> None:
        from .imaging import is_raw
        for e in entries:
            key = str(e)
            if e.suffix.lower() not in self._exts:
                continue                      # not an image type we handle
            seen_now.add(key)
            if not full and (key in self._known or key in self._ready):   # unusable: rare, keep checking
                continue
            try:
                st = e.stat()
            except OSError:
                continue                      # vanished mid-scan
            signature = (st.st_size, st.st_mtime_ns)
            prior_signature = self._observed.get(key)
            self._observed[key] = signature
            if prior_signature is not None and prior_signature != signature:
                self._known.discard(key)
                self._ready.discard(key)
                self._unusable.discard(key)
            if key in self._known or key in self._ready or key in self._unusable:
                continue
            need = self.stable_ticks_raw if is_raw(e) else self.stable_ticks
            prev = self._pending.get(key)
            if prev is None:
                self._pending[key] = _Pending(st.st_size, st.st_mtime_ns)
                # Wake up on FIRST SIGHT, not on readiness. Otherwise a watcher
                # that has backed off to 30 s also does its stability checks at
                # 30 s, so the first photo after a lull takes minutes to appear.
                self._last_new = time.time()
                # No settle wait: the full decode check is what proves a file
                # is complete (on Lucid a file shows its final size while still
                # uploading, so a settled size proves nothing). A file seen
                # growing waits for one unchanged scan instead (no wasted reads).
                if need <= 0:
                    due.append(key)
                continue
            if prev.size == st.st_size and prev.mtime_ns == st.st_mtime_ns:
                prev.stable_ticks += 1
            else:                              # still growing — restart the count
                self._pending[key] = _Pending(st.st_size, st.st_mtime_ns, grew=True)
                continue
            if prev.stable_ticks >= max(need, 1 if prev.grew else 0) and time.time() >= prev.retry_at:
                due.append(key)

    def _forget_missing(self, seen_now: Set[str]) -> None:
        """Forget entries whose file disappeared."""
        for gone in [k for k in self._pending if k not in seen_now]:
            self._pending.pop(gone, None)
        for gone in set(self._observed) - seen_now:
            self._observed.pop(gone, None)
            self._known.discard(gone)
            self._ready.discard(gone)
            self._unusable.discard(gone)

    def _check_decodable(self, due: List[str]) -> None:
        """Size settled — does it actually open?

        On a cloud mount (LucidLink) an entry appears at its final size the
        moment the uploader's local copy finishes, while the bytes are still
        uploading — so size stability proves nothing there, and a failed decode
        means "not here yet" far more often than "corrupt". Retry with backoff
        and only give up after a long stretch of failures.

        Each check is a full download on a cloud mount, so they run in parallel
        (one at a time managed ~1 photo/s off Lucid) and within a time budget:
        whatever has passed goes to ingest now, the rest is checked next scan,
        rather than a 700-photo card dump showing nothing for ten minutes.
        """
        if not due:
            return
        from .imaging import ordered_map
        t0 = time.time()
        deadline = t0 + self.decode_budget
        checks = ordered_map(lambda k: (k, self._check_one(k)), due)
        n = nbytes = 0
        try:
            for key, ok in checks:
                now = time.time()
                with self._sets:
                    self._apply_check(key, ok, now)
                n += 1
                try:
                    nbytes += Path(key).stat().st_size if self.mirror_dir is None else self._local(key).stat().st_size
                except OSError:
                    pass
                if now >= deadline:
                    break
        finally:
            checks.close()
        dt = max(time.time() - t0, 1e-6)
        self._m_check_n, self._m_check_s, self._m_check_mb_s = n, dt, nbytes / 1e6 / dt
        self._m_bytes += nbytes
        self._m_total_check_s += dt

    def _apply_check(self, key: str, ok: bool, now: float) -> None:
        prev = self._pending.get(key)
        if prev is None:
            return                             # vanished or changed meanwhile
        if ok:
            self._ready.add(key)
            self._last_new = now
            self._pending.pop(key, None)
            return
        if not prev.decode_failures:
            prev.first_decode_failure = now
        prev.decode_failures += 1
        if now - prev.first_decode_failure >= self.decode_retry_for:
            self._unusable.add(key)
            self._pending.pop(key, None)
        else:
            delay = min(self.decode_retry_max_interval,
                        self.scan_interval * 2 ** (prev.decode_failures - 1))
            prev.retry_at = now + delay

    def _local(self, key: str) -> Path:
        """Where ingest reads ``key`` from: its mirror copy, or the file itself."""
        if self.mirror_dir is None:
            return Path(key)
        return self.mirror_dir / Path(key).relative_to(self.folder)

    def _decodable_recorded(self, path: Path) -> bool:
        """Decode check that remembers what it proved: the file's signature,
        if it didn't change while being decoded."""
        try:
            before = path.stat()
        except OSError:
            return False
        if not self._decodable(path):
            return False
        try:
            after = path.stat()
        except OSError:
            return False
        sig = [after.st_size, after.st_mtime_ns]
        if sig == [before.st_size, before.st_mtime_ns]:
            with self._lock:
                self._validated[str(path)] = sig
        return True

    def prevalidated(self) -> Dict[str, list]:
        with self._lock:
            return dict(self._validated)

    def _check_one(self, key: str) -> bool:
        """Decode check. With a mirror, this is also the one and only read of
        the photo from the network: copy it locally, then check the copy."""
        if self.mirror_dir is None:
            return self._decodable_recorded(Path(key))
        from .mirror import mirror_copy
        local = self._local(key)
        try:
            mirror_copy(self._channels.source_for(Path(key)), local)
        except OSError:
            try:
                mirror_copy(Path(key), local)          # that channel failed — use the main mount
            except OSError:
                return False
        if self._decodable_recorded(local):
            return True
        try:
            local.unlink()              # half-uploaded: fetch it again next time
        except OSError:
            pass
        return False

    @property
    def ingest_root(self) -> Path:
        return self.mirror_dir or self.folder

    def _list_files(self) -> List[Path]:
        """Every file under the folder, skipping hidden entries (in-flight
        ``.partial`` writes, ``.DS_Store``, Lucid/Finder metadata) and any
        directory holding the export marker, so exported picks written into the
        same share are never ingested back into the shoot."""
        from .export import EXPORT_MARKER
        out: List[Path] = []
        def onerror(exc: OSError) -> None:
            if Path(exc.filename or "") == self.folder:
                raise exc              # the watched folder itself is unreachable
        for root, dirs, files in os.walk(self.folder, onerror=onerror):
            if EXPORT_MARKER in files:
                dirs[:] = []
                continue
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            out.extend(Path(root) / f for f in files if not f.startswith("."))
        return out

    def ingest_paths(self) -> List[Path]:
        """Previously accepted files plus the stable arrivals in this batch
        (their local mirror copies, when mirroring)."""
        if self._inflight is not None:
            return list(self._inflight)
        with self._sets:
            return [self._local(p) for p in sorted(self._known | self._ready)]

    def _start_ingest(self) -> None:
        """Ingest on a background thread, so the scan loop keeps copying new
        photos off a slow share meanwhile instead of idling for the whole batch."""
        t = self._ingest_thread
        if t is not None and t.is_alive():
            return
        if not self._ready or time.time() - self._last_attempt < self.batch_cooldown:
            return
        def run():
            try:
                self._maybe_ingest()
            except Exception as exc:  # noqa: BLE001 — retried on a later tick
                with self._lock:
                    self._status.last_error = f"{type(exc).__name__}: {exc}"
        self._ingest_thread = threading.Thread(target=run, name="hotfolder-ingest", daemon=True)
        self._ingest_thread.start()

    def _maybe_ingest(self) -> None:
        with self._sets:
            if not self._ready:
                return
            if time.time() - self._last_attempt < self.batch_cooldown:
                return
            batch = sorted(self._ready)
            # A fixed snapshot: photos that arrive while this batch runs wait
            # for the next one instead of half-joining this one.
            self._inflight = [self._local(p) for p in sorted(self._known | set(batch))]
        self._last_attempt = time.time()
        try:
            accepted = self._ingest([Path(p) for p in batch])
        finally:
            self._inflight = None
            self._m_ingest_s = time.time() - self._last_attempt
            self._m_total_ingest_s += self._m_ingest_s
            self._m_n_ingests += 1
        if not accepted:
            return          # GPU busy (a manual upload won); try again next tick
        with self._lock:
            self._status.last_error = ""
        self._last_ingest = time.time()
        with self._sets:
            self._n_ingested += len(batch)
            self._known.update(batch)
            self._ready.difference_update(batch)
