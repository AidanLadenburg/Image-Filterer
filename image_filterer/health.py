"""Live health: is anything stopping photos from coming in or going out?

One list of current problems, shown to every viewer as a banner and raised as
an alert when it appears and again when it clears. A problem must persist for
``grace`` seconds first, so a one-scan blip on a cloud mount stays quiet.

Alerts go to the server log, a desktop notification on this machine (when
there is a desktop session), and ``IMAGE_FILTERER_ALERT_WEBHOOK`` if set — a
Slack/Teams-style incoming webhook that accepts ``{"text": ...}``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional

log = logging.getLogger("image_filterer.health")

LOW_MEMORY_GB = 8.0
LOW_DISK_GB = 20.0


def check(*, hotfolder=None, export_dir: Optional[Path] = None, auto_errors: Optional[Dict[str, str]] = None,
          ocr_status: Optional[dict] = None, data_root: Optional[Path] = None) -> Dict[str, str]:
    """{problem key: plain-language message} for everything wrong right now."""
    out: Dict[str, str] = {}
    hw = hotfolder
    if hw is not None and not hw._stop.is_set():
        folder = str(hw.folder)
        if not hw.alive:
            out["watch-dead"] = f"Folder watching stopped unexpectedly ({folder}). Start the watch again."
        elif not Path(folder).is_dir() or str(hw.status().get("last_error", "")).startswith("cannot read folder"):
            out["watch-unreadable"] = f"Can't read the watched folder {folder} — is Lucid mounted?"
    if export_dir is not None and not Path(export_dir).is_dir():
        out["export-missing"] = f"Export folder {export_dir} is missing — is Lucid mounted? Exports will resume when it's back."
    for name, err in (auto_errors or {}).items():
        if err:
            out[f"auto:{name}"] = f"Auto-export ({name}): {err}"
    if ocr_status and ocr_status.get("enabled") and ocr_status.get("error"):
        out["ocr"] = f"Text reading stopped: {ocr_status['error']}"
    try:
        import psutil
        avail = psutil.virtual_memory().available / 2**30
        if avail < LOW_MEMORY_GB:
            out["memory"] = f"Low memory: {avail:.1f} GB free. Restart the app between shoots."
    except Exception:  # noqa: BLE001
        pass
    if data_root is not None:
        try:
            free = shutil.disk_usage(data_root).free / 2**30
            if free < LOW_DISK_GB:
                out["disk"] = f"Low disk space: {free:.0f} GB free on {data_root}."
        except OSError:
            pass
    return out


class HealthMonitor:
    """Polls ``probe`` (→ check() result) and alerts on changes."""

    def __init__(self, probe: Callable[[], Dict[str, str]], *, interval: float = 10.0, grace: float = 30.0,
                 notify: Optional[Callable[[str], None]] = None) -> None:
        self._probe, self.interval, self.grace = probe, interval, grace
        self._notify = notify or send_alert
        self._first_seen: Dict[str, float] = {}
        self._active: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="health", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def problems(self) -> List[str]:
        with self._lock:
            return list(self._active.values())

    def tick(self, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        try:
            current = self._probe()
        except Exception as exc:  # noqa: BLE001 — a broken check must not kill the monitor
            current = {"health-check": f"Health check failed: {exc}"}
        raised, cleared = [], []
        with self._lock:
            for key in list(self._first_seen):
                if key not in current:
                    self._first_seen.pop(key)
            for key, msg in current.items():
                self._first_seen.setdefault(key, now)
                if now - self._first_seen[key] >= self.grace:
                    if key not in self._active:
                        raised.append(msg)
                    self._active[key] = msg
            for key in [k for k in self._active if k not in current]:
                cleared.append(self._active.pop(key))
        for msg in raised:
            self._notify(f"⚠️ Image Filterer: {msg}")
        for msg in cleared:
            self._notify(f"✅ Image Filterer: resolved — {msg}")

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.tick()


def send_alert(text: str) -> None:
    log.warning(text)
    if shutil.which("notify-send") and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
                                        or os.environ.get("DBUS_SESSION_BUS_ADDRESS")):
        try:
            subprocess.run(["notify-send", "-u", "critical", "Image Filterer", text], timeout=5,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:  # noqa: BLE001
            pass
    url = os.environ.get("IMAGE_FILTERER_ALERT_WEBHOOK", "").strip()
    if url:
        def post():
            try:
                req = urllib.request.Request(url, data=json.dumps({"text": text}).encode(),
                                             headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=10).close()
            except Exception as exc:  # noqa: BLE001
                log.warning("alert webhook failed: %s", exc)
        threading.Thread(target=post, daemon=True).start()
