"""Web service — the production browser UI.

Production viewer over unlabeled ingestion runs:

  python -m image_filterer.server --host 0.0.0.0 --port 8600

Flow: upload a folder → a new run is ingested by inference → browse / search /
filter it. Multiple viewers can read/search concurrently; ingestion is
single-flight (guarded by a lock).
"""

from __future__ import annotations

import argparse
import io
import json
import mimetypes
import re
import secrets
import shutil
import socket
import subprocess
import threading
import time
import zipfile
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime

try:
    import resource  # Unix only
except ImportError:  # pragma: no cover
    resource = None
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from flask import Flask, Response, abort, jsonify, request, send_file

from PIL import Image

from .config import Config, default_config
from .db import RunDB
from .ingest import ingest_folder
from .pipeline import read_version

PACKAGE_DIR = Path(__file__).resolve().parent


class _Cancelled(Exception):
    """Raised inside the progress callback when a user cancels the ingestion."""


def _safe_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-")
    return s[:60] or "run"


@dataclass
class LoadedRun:
    """One run's data, held in memory.

    Runs are cached per id rather than kept in a single global slot: run
    selection is per-viewer, so two people can browse different shoots at once
    without one of them yanking the other's view.

    ``version`` is the generation the data was read at. It is re-checked on every
    lookup, so a background re-ingest is picked up automatically, and clients
    paginating against an older generation can be told to reload.
    """

    run_id: int
    run_dir: Path
    ranked: "pd.DataFrame"
    bursts: "pd.DataFrame"
    version: int
    # path -> set of viewer ids that starred it. A star belongs to whoever made
    # it: several people can star the same frame, and un-starring only ever
    # removes your own. (Before this, clicking a colleague's star deleted it.)
    stars: Dict[str, Set[str]] = field(default_factory=dict)
    emb: Optional[np.ndarray] = None
    emb_context: Optional[str] = None
    shot_types: Optional[np.ndarray] = None
    captured: Optional[np.ndarray] = None
    scores_sorted: Optional[np.ndarray] = None   # for the quality percentile
    stars_lock: threading.Lock = field(default_factory=threading.Lock)


def create_app(cfg: Optional[Config] = None) -> Flask:
    cfg = cfg or default_config()
    cfg.ensure_dirs()
    db = RunDB(cfg.db_path)

    app = Flask(__name__, template_folder=str(PACKAGE_DIR / "templates"))
    # Per-run cache. Small: a run is a CSV plus a ~15 MB embedding matrix, and
    # holding a few lets several viewers browse different shoots concurrently.
    runs_cache: "OrderedDict[int, LoadedRun]" = OrderedDict()
    runs_lock = threading.Lock()
    RUN_CACHE_MAX = 3

    state: Dict[str, object] = {
        # Genuinely global: one GPU, one ingest at a time, one shared text tower.
        "allowed_roots": set(),          # union over every run loaded this session
        "text_encoder": None,
        "search_lock": threading.Lock(),
        "ingest_lock": threading.Lock(),
        "active_ingest": None,           # run_id currently ingesting (single-flight)
        # Live progress for whatever is processing, set by every ingest path.
        # Kept here rather than derived from the run's DB status: a hot-folder
        # batch deliberately leaves its run "ready" so viewers don't lose it
        # mid-event, which used to make those ingests invisible.
        "active_progress": None,         # {run_id, name, progress, message}
        "cancel_event": threading.Event(),
        "hotfolder": None,               # at most one HotFolderWatcher, ever
    }

    # ----------------------------- stars
    #
    # Stars mark individual frames (not bursts) — the whole job of the ranker is
    # to say *which* frame of a burst is the keeper, so that's the unit worth
    # saving. They live in the run's own folder, next to ranked.csv, which makes
    # them shared by every viewer and portable with the run.

    LEGACY_VIEWER = "legacy"   # stars made before ownership was tracked

    def _load_stars(run_dir: Path) -> Dict[str, Set[str]]:
        p = run_dir / "stars.json"
        if not p.exists():
            return {}
        try:
            blob = json.loads(p.read_text())
        except Exception:  # noqa: BLE001 — a corrupt stars file must not block the run
            return {}
        if isinstance(blob.get("stars"), dict):          # current format
            return {str(k): {str(v) for v in (vs or [])} for k, vs in blob["stars"].items()}
        # v1: a flat list with no owner. Attribute to nobody in particular.
        return {str(x): {LEGACY_VIEWER} for x in (blob.get("starred") or [])}

    def _save_stars(rn: LoadedRun) -> None:
        """Write via a temp file + rename so a crash mid-write can't truncate it."""
        payload = json.dumps({
            "version": 2,
            "stars": {k: sorted(v) for k, v in sorted(rn.stars.items()) if v},
        }, indent=1)
        tmp = rn.run_dir / "stars.json.tmp"
        tmp.write_text(payload)
        tmp.replace(rn.run_dir / "stars.json")

    def _viewer() -> str:
        """Who is asking. Browser-generated id — there are no accounts here."""
        v = request.args.get("viewer")
        if not v and request.method == "POST":
            v = (request.get_json(silent=True) or {}).get("viewer")
        return str(v or "anon")

    def _star_owners(rn: LoadedRun, path: str) -> Set[str]:
        return rn.stars.get(str(path), set())

    def _star_flags(rn: LoadedRun, path: str, me: str) -> Dict[str, bool]:
        owners = _star_owners(rn, path)
        return {"starred": bool(owners),
                "starred_mine": me in owners,
                "starred_others": bool(owners - {me})}

    def _paths_for_owner(rn: LoadedRun, me: str, owner: str) -> Set[str]:
        """Starred paths filtered to all / just mine / just other people's."""
        if owner == "mine":
            return {p for p, o in rn.stars.items() if me in o}
        if owner == "others":
            return {p for p, o in rn.stars.items() if o - {me}}
        return {p for p, o in rn.stars.items() if o}

    def _starred_counts(rn: LoadedRun) -> Dict[int, int]:
        """burst_id → how many of its frames are starred by anyone."""
        if not rn.stars:
            return {}
        hit = rn.ranked[rn.ranked["path"].astype(str).isin(set(rn.stars))]
        return {int(b): int(n) for b, n in hit.groupby("burst_id").size().items()}

    # ----------------------------- run loading

    def _read_run(run_id: int) -> Optional[LoadedRun]:
        """Read one run's outputs as a CONSISTENT snapshot.

        The version is read before and after the files. If it moved, an ingest
        committed mid-read and the files we just read may be a mixture of two
        generations, so we try again. That, plus writers bumping the version only
        after every output is in place, is what guarantees a viewer never sees a
        half-applied batch.
        """
        row = db.get(run_id)
        if not row or row["status"] != "ready":
            return None
        run_dir = cfg.runs_dir / row["folder"]
        if not (run_dir / "ranked.csv").exists():
            return None

        for _ in range(10):
            version = read_version(run_dir)
            if version % 2 == 1:
                # Odd = a write is in flight; the files may disagree. Wait it out.
                time.sleep(0.03)
                continue
            try:
                ranked = pd.read_csv(run_dir / "ranked.csv")
                bursts = pd.read_csv(run_dir / "bursts.csv")
            except Exception:  # noqa: BLE001 — mid-rename; retry
                continue
            rn = LoadedRun(run_id=run_id, run_dir=run_dir, ranked=ranked, bursts=bursts,
                           version=version, stars=_load_stars(run_dir))
            npy = run_dir / "search_index.npy"
            if npy.exists():
                try:
                    mat = np.load(npy)
                    if mat.ndim == 2 and mat.shape[0] == len(ranked):
                        rn.emb = mat.astype(np.float32)
                        meta = run_dir / "search_index.json"
                        rn.emb_context = (
                            json.loads(meta.read_text()).get("context_encoder",
                                                             cfg.features.context_encoder)
                            if meta.exists() else cfg.features.context_encoder)
                except Exception:  # noqa: BLE001
                    pass
            if "shot_type" in ranked.columns and ranked["shot_type"].astype(str).str.len().gt(0).any():
                rn.shot_types = ranked["shot_type"].astype(str).to_numpy()
            if read_version(run_dir) != version:
                continue                      # a batch landed mid-read — retry
            for pth in ranked["path"]:
                try:
                    state["allowed_roots"].add(str(Path(pth).resolve().parent))
                except Exception:  # noqa: BLE001
                    pass
            return rn
        return None

    def get_run(run_id: Optional[int]) -> Optional[LoadedRun]:
        """Cached run, reloaded automatically when its generation has moved on."""
        if run_id is None:
            return None
        with runs_lock:
            rn = runs_cache.get(int(run_id))
            if rn is not None and read_version(rn.run_dir) == rn.version:
                runs_cache.move_to_end(int(run_id))
                return rn
            previous = rn          # keep as a fallback while a write is in flight
        # Load outside the lock: reading a run takes ~a second and must not block
        # every other viewer's request.
        fresh = _read_run(int(run_id))
        if fresh is None:
            # Couldn't get a clean snapshot (a writer is mid-batch). Serving the
            # previous complete generation is strictly better than an empty view;
            # the client's poll will pick up the new one a moment later.
            return previous
        with runs_lock:
            runs_cache[int(run_id)] = fresh
            runs_cache.move_to_end(int(run_id))
            while len(runs_cache) > RUN_CACHE_MAX:
                runs_cache.popitem(last=False)
        return fresh

    def _default_run_id() -> Optional[int]:
        """Fallback for clients that send no run_id: the newest ready run."""
        for r in db.list(include_unready=False):
            return int(r["id"])
        return None

    def resolve_run() -> Optional[LoadedRun]:
        """The run THIS request is about — per viewer, never global state."""
        raw = request.args.get("run_id")
        if raw is None and request.method == "POST":
            body = request.get_json(silent=True) or {}
            raw = body.get("run_id", request.form.get("run_id"))
        try:
            run_id = int(raw) if raw not in (None, "") else None
        except (TypeError, ValueError):
            run_id = None
        return get_run(run_id if run_id is not None else _default_run_id())

    def _get_text_encoder(ctx: str):
        te = state["text_encoder"]
        if te is not None and getattr(te, "name", None) == ctx.strip().lower():
            return te
        from .encoders import build_text_encoder
        te = build_text_encoder(ctx)
        state["text_encoder"] = te
        return te

    def _ensure_shot_types(rn: LoadedRun) -> Optional[np.ndarray]:
        if rn.shot_types is not None:
            return rn.shot_types
        if rn.emb is None or len(rn.emb) != len(rn.ranked):
            return None
        ctx = rn.emb_context or cfg.features.context_encoder
        from .encoders import context_encoder_has_text_tower
        if not context_encoder_has_text_tower(ctx):
            return None
        from .shots import compute_shot_types
        with state["search_lock"]:
            te = _get_text_encoder(ctx)
            labels, _ = compute_shot_types(rn.emb, ctx, cfg.shots, text_encoder=te)
        if not labels:
            return None
        rn.shot_types = np.asarray(labels)
        return rn.shot_types

    # ----------------------------- capture times (chronological sort)

    def _capture_times(rn: LoadedRun) -> Optional[np.ndarray]:
        """Epoch seconds per ``ranked`` row, or None if unobtainable.

        Capture time isn't a column in older runs — EXIF is read during burst
        clustering and then discarded — so this backfills it and caches the
        result beside the run. Reading DateTimeOriginal costs ~0.3 ms/frame, so
        a full 3,000-frame backfill is about a second, once.
        """
        if rn.captured is not None:
            return rn.captured
        ranked = rn.ranked
        arr: Optional[np.ndarray] = None

        if "captured_at" in ranked.columns:      # written by newer ingests
            ts = pd.to_datetime(ranked["captured_at"], errors="coerce")
            if ts.notna().any():
                arr = ts.map(lambda x: x.timestamp() if pd.notna(x) else np.nan) \
                        .to_numpy(dtype=np.float64)

        side = rn.run_dir / "captured_at.json"
        if arr is None and side.exists():
            try:
                cached = json.loads(side.read_text())
                arr = np.array([float(cached[str(p)]) if cached.get(str(p)) is not None else np.nan
                                for p in ranked["path"]], dtype=np.float64)
            except Exception:  # noqa: BLE001 — a bad sidecar just forces a re-read
                arr = None

        if arr is None:
            from .bursts import read_frame_meta
            vals: List[float] = []
            for p in ranked["path"]:
                meta = read_frame_meta(Path(p))
                if meta is not None:
                    vals.append(meta.captured_at.timestamp())
                    continue
                try:    # no EXIF timestamp — mtime is a serviceable stand-in
                    vals.append(Path(p).stat().st_mtime)
                except OSError:
                    vals.append(float("nan"))
            arr = np.array(vals, dtype=np.float64)
            try:
                side.write_text(json.dumps({
                    str(p): (None if np.isnan(v) else v)
                    for p, v in zip(ranked["path"], arr)}))
            except OSError:
                pass
        rn.captured = arr
        return arr

    def _burst_min_times(rn: LoadedRun) -> Dict[int, float]:
        """burst_id → capture time of its earliest frame."""
        ct = _capture_times(rn)
        if ct is None:
            return {}
        s = pd.Series(ct, index=rn.ranked.index).groupby(rn.ranked["burst_id"]).min()
        return {int(b): float(v) for b, v in s.items()}

    def _iso(ts: Optional[float]) -> str:
        if ts is None or (isinstance(ts, float) and np.isnan(ts)):
            return ""
        return datetime.fromtimestamp(ts).isoformat(sep=" ", timespec="seconds")

    # ----------------------------- quality cutoff

    SCORE_COL = "score_s12_after_hard"

    def _quality_threshold(rn: LoadedRun, top_pct: float) -> Optional[float]:
        """Score at or above which a frame is in the top ``top_pct`` percent.

        The slider is expressed as a percentile rather than a raw score on
        purpose: the ranker's output is an ordering, not a measurement — the
        units are arbitrary and differ between runs — so "top 25%" is the only
        framing that means the same thing everywhere.
        """
        if top_pct >= 100 or top_pct <= 0:
            return None
        col = SCORE_COL if SCORE_COL in rn.ranked.columns else "score_s1"
        if col not in rn.ranked.columns or rn.ranked.empty:
            return None
        if rn.scores_sorted is None:
            rn.scores_sorted = np.sort(rn.ranked[col].to_numpy(dtype=np.float64))
        return float(np.quantile(rn.scores_sorted, 1.0 - top_pct / 100.0))

    def _top_pct() -> float:
        try:
            return max(1.0, min(100.0, float(request.args.get("top_pct", "100"))))
        except (TypeError, ValueError):
            return 100.0

    def _score_of(row) -> float:
        v = row.get(SCORE_COL, row.get("score_s1", 0.0))
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    def _burst_rep_shot_map(rn: LoadedRun) -> Dict[int, str]:
        st = _ensure_shot_types(rn)
        if st is None:
            return {}
        rep = rn.ranked.assign(_shot=st)
        rep = rep[rep["is_representative"].astype(bool)]
        return {int(b): str(s) for b, s in zip(rep["burst_id"], rep["_shot"])}

    # Warm the cache with the newest ready run so the first request is quick.
    get_run(_default_run_id())

    # ----------------------------- ingestion

    def _do_ingest(run_id: int, folder: Path) -> None:
        cancel = state["cancel_event"]

        row0 = db.get(run_id)
        run_name = row0["name"] if row0 else ""

        def cb(pct: float, msg: str) -> None:
            # Cooperative cancellation: this fires between feature chunks and at
            # each phase boundary, so Cancel takes effect within a chunk or two.
            if cancel.is_set():
                raise _Cancelled()
            state["active_progress"] = {"run_id": run_id, "name": run_name,
                                        "progress": float(pct), "message": msg}
            db.update(run_id, progress=float(pct), message=msg, status="ingesting")

        try:
            row = db.get(run_id)
            run_dir = cfg.runs_dir / row["folder"]
            run_dir.mkdir(parents=True, exist_ok=True)  # may not exist for server-path ingest
            (run_dir / "config.json").write_text(json.dumps({
                "context_encoder": cfg.features.context_encoder,
                "face_encoder": cfg.features.face_encoder,
            }, indent=2))
            res = ingest_folder(folder, cfg, run_dir, progress_cb=cb)
            db.update(run_id, status="ready", progress=100.0,
                      n_images=res["n_images"], n_bursts=res["n_bursts"],
                      message=f"Ready — {res['n_images']} images, {res['n_bursts']} bursts.")
        except _Cancelled:
            db.update(run_id, status="error", progress=0.0,
                      message="Cancelled by user.", error="cancelled")
        except Exception as exc:  # noqa: BLE001
            import traceback
            db.update(run_id, status="error", error=traceback.format_exc(),
                      message=f"Error: {exc}")
        finally:
            state["active_ingest"] = None
            state["active_progress"] = None
            state["ingest_lock"].release()

    def _upload_dir_for(run_id: int) -> Optional[Path]:
        row = db.get(run_id)
        if not row or row["status"] != "uploading":
            return None
        return cfg.runs_dir / row["folder"] / "uploads"

    def _save_within(upload_dir: Path, f) -> None:
        # Save f under upload_dir, preserving webkitRelativePath, blocking traversal.
        rel = Path(f.filename or "file")
        dest = (upload_dir / rel).resolve()
        if not str(dest).startswith(str(upload_dir.resolve())):
            return  # path-traversal attempt — skip
        dest.parent.mkdir(parents=True, exist_ok=True)
        f.save(str(dest))
        # Repair the occasional leading-\r\n corruption seen on some upload paths,
        # so the bytes (and sha1) match the original again.
        from .imaging import sanitize_image_file
        sanitize_image_file(dest)

    @app.route("/api/upload/start", methods=["POST"])
    def api_upload_start():
        """Begin a batched upload: reserve a run + its upload dir. Uploading does
        NOT hold the ingest lock, so multiple people can upload at once; only the
        finish step (which starts the GPU work) is single-flight."""
        raw_name = request.form.get("name", "").strip()
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = raw_name or f"Run {ts}"
        run_id = db.create(name, folder="")
        folder = f"run{run_id:04d}_{_safe_name(name)}"
        db.update(run_id, folder=folder, status="uploading", message="Uploading…")
        (cfg.runs_dir / folder / "uploads").mkdir(parents=True, exist_ok=True)
        return jsonify({"ok": True, "run_id": run_id})

    @app.route("/api/upload/chunk", methods=["POST"])
    def api_upload_chunk():
        """Receive one batch of files (a few hundred) for an in-progress upload.
        Bounding the batch size bounds how many tempfile FDs are open at once —
        the whole point, since werkzeug holds every part open for the request."""
        run_id = int(request.form.get("run_id", "0"))
        upload_dir = _upload_dir_for(run_id)
        if upload_dir is None:
            return jsonify({"error": "Unknown or non-uploading run."}), 400
        files = request.files.getlist("files")
        for f in files:
            _save_within(upload_dir, f)
        return jsonify({"ok": True, "saved": len(files)})

    @app.route("/api/upload/finish", methods=["POST"])
    def api_upload_finish():
        """All batches sent → kick off ingestion (single-flight)."""
        run_id = int(request.form.get("run_id", "0"))
        upload_dir = _upload_dir_for(run_id)
        if upload_dir is None:
            return jsonify({"error": "Unknown or non-uploading run."}), 400
        if not state["ingest_lock"].acquire(blocking=False):
            return jsonify({"error": "An ingestion is already running. Please wait."}), 409
        try:
            db.update(run_id, status="ingesting", message="Starting…")
            state["cancel_event"].clear()
            state["active_ingest"] = run_id
            threading.Thread(target=_do_ingest, args=(run_id, upload_dir), daemon=True).start()
            return jsonify({"ok": True, "run_id": run_id})
        except Exception:
            state["ingest_lock"].release()
            raise

    # ----------------------------- hot folder
    #
    # One watcher at a time, opt-in, and self-stopping. Watching is NOT something
    # runs acquire by existing: an ingested shoot is inert files on disk, and
    # nothing polls it. That keeps the cost proportional to what's actually live
    # rather than to how many shoots have ever been processed.

    def _hot_ingest(batch: List[Path]) -> bool:
        """Fold the watched folder into its run. Returns False if the GPU is busy.

        Re-ingests the WHOLE folder rather than just ``batch``: the content cache
        makes already-seen frames nearly free, and it keeps bursts, dedup and the
        ranking correct across the entire shoot instead of stapling new photos on
        the end.
        """
        hw = state["hotfolder"]
        if hw is None:
            return True
        # Never queue behind a manual upload — decline and retry next tick.
        if not state["ingest_lock"].acquire(blocking=False):
            return False
        run_id = hw.run_id
        try:
            state["active_ingest"] = run_id
            row = db.get(run_id)
            if not row:
                return True
            run_dir = cfg.runs_dir / row["folder"]
            run_dir.mkdir(parents=True, exist_ok=True)
            first_time = row["status"] != "ready"

            def cb(pct: float, msg: str) -> None:
                if state["cancel_event"].is_set():
                    raise _Cancelled()
                state["active_progress"] = {"run_id": run_id, "name": row["name"],
                                            "progress": float(pct), "message": msg}
                fields = {"progress": float(pct), "message": msg}
                # A run that's already live must stay selectable while it grows;
                # only the very first pass may show as "ingesting".
                if first_time:
                    fields["status"] = "ingesting"
                db.update(run_id, **fields)

            res = ingest_folder(Path(hw.folder), cfg, run_dir, progress_cb=cb)
            db.update(run_id, status="ready", progress=100.0,
                      n_images=res["n_images"], n_bursts=res["n_bursts"],
                      message=f"Watching — {res['n_images']} images, {res['n_bursts']} bursts.")
            return True
        except _Cancelled:
            db.update(run_id, message="Hot folder ingest cancelled.")
            return True
        except Exception as exc:  # noqa: BLE001
            # Consume the batch rather than retrying forever: the folder is
            # re-scanned in full next pass, so nothing is actually lost.
            db.update(run_id, message=f"Hot folder error: {exc}")
            return True
        finally:
            state["active_ingest"] = None
            state["active_progress"] = None
            state["ingest_lock"].release()

    @app.route("/api/hotfolder", methods=["GET"])
    def api_hotfolder():
        hw = state["hotfolder"]
        return jsonify(hw.status() if hw is not None else {"watching": False})

    @app.route("/api/hotfolder/start", methods=["POST"])
    def api_hotfolder_start():
        data = request.get_json(silent=True) or {}
        folder = str(data.get("path", "")).strip()
        if not folder:
            return jsonify({"error": "No path given"}), 400
        fp = Path(folder).expanduser()
        if not fp.is_dir():
            return jsonify({"error": f"Not a directory on the server: {fp}"}), 400

        old = state["hotfolder"]
        if old is not None and old.alive:
            old.stop()          # one at a time, so a forgotten watch can't linger

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = str(data.get("name", "")).strip() or fp.name or f"Watch {ts}"
        run_id = db.create(name, folder="")
        db.update(run_id, folder=f"run{run_id:04d}_{_safe_name(name)}",
                  status="queued", message="Watching folder…")
        state["cancel_event"].clear()

        from .hotfolder import HotFolderWatcher
        hw = HotFolderWatcher(fp, run_id, _hot_ingest,
                              scan_interval=float(data.get("scan_interval", 5.0)),
                              batch_cooldown=float(data.get("cooldown", 20.0)))
        state["hotfolder"] = hw
        hw.start()
        return jsonify({"ok": True, "run_id": run_id, "folder": str(fp)})

    @app.route("/api/hotfolder/stop", methods=["POST"])
    def api_hotfolder_stop():
        hw = state["hotfolder"]
        if hw is None or not hw.alive:
            return jsonify({"ok": False, "error": "Nothing is being watched."}), 400
        hw.stop()
        return jsonify({"ok": True, "run_id": hw.run_id})

    @app.route("/api/active")
    def api_active():
        """Global: is anything being processed right now? Polled by every client
        so the activity bar is visible to everyone, not just whoever started it.

        Reads the in-memory progress rather than the run's DB status, because a
        hot-folder batch keeps its run "ready" on purpose and would otherwise
        never show up here.
        """
        prog = state["active_progress"]
        if prog is None:
            return jsonify({"active": None})
        return jsonify({"active": dict(prog)})

    @app.route("/api/cancel_ingest", methods=["POST"])
    def api_cancel_ingest():
        if state["active_ingest"] is None:
            return jsonify({"ok": False, "error": "No ingestion is running."}), 400
        state["cancel_event"].set()
        return jsonify({"ok": True})

    @app.route("/api/ingest/status")
    def api_ingest_status():
        run_id = int(request.args.get("run_id", "0"))
        row = db.get(run_id)
        if not row:
            abort(404)
        return jsonify({k: row[k] for k in
                        ("id", "name", "status", "progress", "message", "error", "n_images", "n_bursts")})

    # ----------------------------- runs / selection

    def _runs_payload() -> List[Dict[str, object]]:
        out = []
        for r in db.list():
            out.append({
                "id": r["id"], "name": r["name"], "status": r["status"],
                "created_at": r["created_at"], "n_images": r["n_images"],
                "n_bursts": r["n_bursts"],
            })
        return out

    @app.route("/api/state")
    def api_state():
        """``current`` is only a *suggestion* for a client with no run yet — the
        server no longer has a single selected run."""
        cur = None
        rid = _default_run_id()
        if rid is not None:
            row = db.get(rid)
            if row:
                cur = {"id": row["id"], "name": row["name"],
                       "n_images": row["n_images"], "n_bursts": row["n_bursts"]}
        return jsonify({"current": cur, "runs": _runs_payload()})

    @app.route("/api/runs")
    def api_runs():
        return jsonify({"runs": _runs_payload(), "current_id": _default_run_id()})

    @app.route("/api/run/preview_delete")
    def api_preview_delete():
        """What deleting this run would actually destroy.

        The UI shows this before asking. The distinction that matters: an
        uploaded run keeps the ONLY copy of its photos under ``uploads/``, while
        a watched folder's originals live outside the run and are untouched.
        """
        try:
            run_id = int(request.args.get("run_id", "0"))
        except (TypeError, ValueError):
            return jsonify({"error": "Bad run_id"}), 400
        row = db.get(run_id)
        if not row:
            return jsonify({"error": "No such run."}), 404
        run_dir = cfg.runs_dir / row["folder"]
        uploads = run_dir / "uploads"
        n_originals, n_bytes = 0, 0
        if uploads.is_dir():
            for f in uploads.rglob("*"):
                if f.is_file():
                    n_originals += 1
                    try:
                        n_bytes += f.stat().st_size
                    except OSError:
                        pass
        total = 0
        if run_dir.is_dir():
            for f in run_dir.rglob("*"):
                if f.is_file():
                    try:
                        total += f.stat().st_size
                    except OSError:
                        pass
        return jsonify({
            "run_id": run_id, "name": row["name"], "status": row["status"],
            "n_images": row["n_images"], "exists": run_dir.is_dir(),
            "holds_originals": n_originals > 0,
            "n_originals": n_originals, "originals_bytes": n_bytes,
            "total_bytes": total,
            "busy": (state["active_ingest"] == run_id
                     or (state["hotfolder"] is not None and state["hotfolder"].alive
                         and state["hotfolder"].run_id == run_id)),
        })

    @app.route("/api/delete_run", methods=["POST"])
    def api_delete_run():
        """Remove a run: its folder and its registry row. Irreversible."""
        data = request.get_json(silent=True) or {}
        try:
            run_id = int(data.get("run_id", 0))
        except (TypeError, ValueError):
            return jsonify({"error": "Bad run_id"}), 400
        row = db.get(run_id)
        if not row:
            return jsonify({"error": "No such run."}), 404
        # Refuse while something is writing to it — deleting mid-ingest would
        # leave a half-written tree and a thread writing into nothing.
        if state["active_ingest"] == run_id:
            return jsonify({"error": "That run is being processed right now."}), 409
        hw = state["hotfolder"]
        if hw is not None and hw.alive and hw.run_id == run_id:
            return jsonify({"error": "That run is being watched. Stop watching first."}), 409

        run_dir = cfg.runs_dir / row["folder"]
        # Never delete outside the runs directory, whatever the registry says.
        try:
            run_dir.resolve().relative_to(cfg.runs_dir.resolve())
        except ValueError:
            return jsonify({"error": "Run folder is outside the runs directory."}), 400
        removed = False
        if run_dir.is_dir():
            shutil.rmtree(run_dir, ignore_errors=True)
            removed = not run_dir.exists()
        db.delete(run_id)
        with runs_lock:
            runs_cache.pop(run_id, None)
        return jsonify({"ok": True, "run_id": run_id, "name": row["name"],
                        "folder_removed": removed})

    @app.route("/api/select_run", methods=["POST"])
    def api_select_run():
        """Validate + warm a run. Selection itself lives in the client from here
        on, so this changes nothing for anyone else's session."""
        run_id = int((request.get_json(silent=True) or {}).get("run_id")
                     or request.form.get("run_id", 0))
        rn = get_run(run_id)
        if rn is None:
            return jsonify({"error": "Run not found or not ready."}), 404
        return jsonify({"ok": True, "run_id": run_id, "version": rn.version})

    @app.route("/api/version")
    def api_version():
        """Cheap poll: has this run's data moved on since the client loaded it?

        Drives the reload prompt. Returns the run's current generation plus its
        size, so the client can say how many photos arrived.
        """
        raw = request.args.get("run_id")
        try:
            run_id = int(raw) if raw not in (None, "") else _default_run_id()
        except (TypeError, ValueError):
            run_id = _default_run_id()
        row = db.get(run_id) if run_id is not None else None
        if not row:
            return jsonify({"version": 0, "n_frames": 0, "n_bursts": 0})
        # Deliberately does NOT load the run: this is polled every couple of
        # seconds by every open tab, and all it owes them is a number. Reading a
        # 15 MB matrix to answer "has anything changed?" would be absurd.
        run_dir = cfg.runs_dir / row["folder"]
        return jsonify({"run_id": run_id, "version": read_version(run_dir),
                        "n_frames": int(row["n_images"]), "n_bursts": int(row["n_bursts"])})

    # ----------------------------- browse / search

    @app.route("/")
    def index():
        return send_file(PACKAGE_DIR / "templates" / "index.html")

    def _run_stats(rn: LoadedRun, *, visible_bursts=None, thr: Optional[float] = None) -> Dict[str, object]:
        row = db.get(rn.run_id)
        # How much the quality cutoff actually removed, so "focus on the best"
        # never silently becomes "most of the shoot is gone".
        shown_frames = None
        if thr is not None:
            col = SCORE_COL if SCORE_COL in rn.ranked.columns else "score_s1"
            keep = rn.ranked[col] >= thr
            if visible_bursts is not None:
                keep = keep & rn.ranked["burst_id"].isin(list(visible_bursts))
            shown_frames = int(keep.sum())
        return {
            "n_frames": int(len(rn.ranked)),
            "n_bursts": int(rn.bursts["burst_id"].nunique()),
            "n_frames_shown": shown_frames,
            "n_multi": int((rn.bursts["burst_size"] > 1).sum()),
            "n_starred": int(len(rn.stars)),
            "run_name": row["name"] if row else "",
            "run_id": rn.run_id,
            "version": rn.version,
        }

    def _stale(rn: LoadedRun) -> bool:
        """True when the client is paginating against a superseded generation.

        Offset-based paging is only coherent within one ordering: if a batch
        landed since page 1, offset=50 into the NEW ordering would repeat some
        tiles and skip others. So we refuse and make the client reload.
        """
        seen = request.args.get("version")
        if seen in (None, ""):
            return False
        try:
            return int(seen) != rn.version
        except (TypeError, ValueError):
            return False

    @app.route("/api/bursts")
    def api_bursts():
        rn = resolve_run()
        if rn is None:
            return jsonify({"bursts": [], "stats": None,
                            "page": {"offset": 0, "limit": 50, "total_matching": 0}})
        if _stale(rn):
            return jsonify({"bursts": [], "stale": True, "version": rn.version,
                            "stats": _run_stats(rn),
                            "page": {"offset": 0, "limit": 0, "total_matching": 0}})
        bursts: pd.DataFrame = rn.bursts
        shot_filter = {s for s in request.args.get("shot", "").split(",") if s}
        subject_filter = request.args.get("subject", "").strip()   # "" | people | stage
        hero_only = request.args.get("hero") in ("1", "true", "yes")
        offset = int(request.args.get("offset", "0"))
        limit = int(request.args.get("limit", "50"))

        df = bursts.copy()
        # Quality cutoff drops whole moments whose best frame doesn't clear the
        # bar — a burst represented by a frame you'd never use isn't a moment
        # worth showing.
        thr = _quality_threshold(rn, _top_pct())
        if thr is not None:
            col = SCORE_COL if SCORE_COL in df.columns else "score_s1"
            df = df[df[col] >= thr]
        rep_shot = _burst_rep_shot_map(rn)
        if rep_shot:
            df = df.assign(_shot=df["burst_id"].map(lambda b: rep_shot.get(int(b), "")))
            if shot_filter:
                df = df[df["_shot"].isin(shot_filter)]
        if subject_filter and "subject_class" in df.columns:
            df = df[df["subject_class"].astype(str) == subject_filter]
        if hero_only and "is_hero" in df.columns:
            df = df[df["is_hero"].astype(bool)]

        df_all_ids = df["burst_id"].tolist()
        total = len(df)
        # sort=time orders bursts by when they were shot rather than by score.
        # Bursts stay collapsed either way — this only changes their order.
        tmin: Dict[int, float] = {}
        if request.args.get("sort") == "time":
            tmin = _burst_min_times(rn)
        if tmin:
            df = df.assign(_t=df["burst_id"].map(lambda b: tmin.get(int(b), float("nan"))))
            df = df.sort_values(["_t", "burst_rank"], na_position="last")
        else:
            df = df.sort_values("burst_rank")
        df = df.iloc[offset:offset + limit]
        star_counts = _starred_counts(rn)
        me = _viewer()
        out: List[Dict[str, object]] = []
        for _, r in df.iterrows():
            out.append({
                "burst_rank": int(r["burst_rank"]),
                "burst_id": int(r["burst_id"]),
                "burst_size": int(r["burst_size"]),
                "rep_filename": r["filename"],
                "rep_path": r["path"],
                "shot_type": str(r.get("_shot", "") or ""),
                "subject_class": str(r.get("subject_class", "") or ""),
                "is_hero": bool(r.get("is_hero", False)),
                "score_s1": float(r.get("score_s1", 0.0)),
                "score_s12_after_hard": float(r.get("score_s12_after_hard", 0.0)),
                **_star_flags(rn, str(r["path"]), me),
                "n_starred": int(star_counts.get(int(r["burst_id"]), 0)),
                "captured_at": _iso(tmin.get(int(r["burst_id"]))) if tmin else "",
            })
        return jsonify({"bursts": out,
                        "stats": _run_stats(rn, visible_bursts=set(df_all_ids), thr=thr),
                        "version": rn.version,
                        "page": {"offset": offset, "limit": limit, "total_matching": int(total)}})

    @app.route("/api/stars")
    def api_stars():
        """Every starred path in the loaded run — the client mirrors this set so
        tiles can render their star state without a request per tile."""
        rn = resolve_run()
        if rn is None:
            return jsonify({"mine": [], "others": [], "count": 0, "run_id": None})
        me = _viewer()
        mine = sorted(p for p, o in rn.stars.items() if me in o)
        others = sorted(p for p, o in rn.stars.items() if o - {me})
        return jsonify({"mine": mine, "others": others,
                        "count": len(rn.stars), "n_mine": len(mine),
                        "n_others": len(others), "run_id": rn.run_id})

    @app.route("/api/star", methods=["POST"])
    def api_star():
        """Star / unstar one frame. ``{path, starred}`` → the new count."""
        rn = resolve_run()
        if rn is None:
            return jsonify({"error": "No run selected."}), 400
        data = request.get_json(silent=True) or {}
        path = str(data.get("path", ""))
        want = bool(data.get("starred", True))
        # Only frames belonging to this run may be starred — same trust boundary
        # as image serving, so a stray path can't be written into stars.json.
        if path not in set(rn.ranked["path"].astype(str)):
            return jsonify({"error": "Unknown frame for this run."}), 404
        me = _viewer()
        with rn.stars_lock:
            owners = rn.stars.setdefault(path, set())
            if want:
                owners.add(me)
            else:
                # Only ever removes your own — a colleague's pick is not yours
                # to delete by clicking the same tile.
                owners.discard(me)
            if not owners:
                rn.stars.pop(path, None)
            _save_stars(rn)
            flags = _star_flags(rn, path, me)
            count = len(rn.stars)
        return jsonify({"ok": True, "path": path, "count": count, **flags})

    @app.route("/api/stars/clear", methods=["POST"])
    def api_stars_clear():
        """Remove YOUR stars from this run. Other people's are never touched.

        There is deliberately no "clear everyone's" — stars are a shared pick
        list with no undo, and one click that wipes a colleague's work during a
        live shoot is not a button worth having.
        """
        rn = resolve_run()
        if rn is None:
            return jsonify({"error": "No run selected."}), 400
        me = _viewer()
        with rn.stars_lock:
            cleared = 0
            for path in list(rn.stars):
                if me in rn.stars[path]:
                    rn.stars[path].discard(me)
                    cleared += 1
                    if not rn.stars[path]:
                        rn.stars.pop(path, None)
            _save_stars(rn)
            count = len(rn.stars)
        return jsonify({"ok": True, "cleared": cleared, "count": count})

    @app.route("/api/starred")
    def api_starred():
        """The saved-for-later list: one entry per starred FRAME, not per burst.

        Shaped like ``/api/bursts`` rows so the grid renders it unchanged. The
        shot / subject / hero filters still AND in, matching every other view.
        """
        offset = int(request.args.get("offset", "0"))
        limit = int(request.args.get("limit", "50"))
        page_empty = {"offset": offset, "limit": limit, "total_matching": 0}
        rn = resolve_run()
        if rn is None:
            return jsonify({"bursts": [], "stats": None, "page": page_empty})
        if _stale(rn):
            return jsonify({"bursts": [], "stale": True, "version": rn.version,
                            "stats": _run_stats(rn), "page": page_empty})
        ranked = rn.ranked

        me = _viewer()
        owner = request.args.get("owner", "all")     # all | mine | others
        # NB: the quality cutoff is deliberately NOT applied here. A star is an
        # explicit human pick; hiding one because the model scored it low would
        # be the tool overruling the person. Shot / subject / hero still apply.
        sub = ranked[ranked["path"].astype(str).isin(_paths_for_owner(rn, me, owner))].copy()
        st = _ensure_shot_types(rn)
        if st is not None and len(sub):
            sub["_shot"] = st[sub.index]
        shot_filter = {s for s in request.args.get("shot", "").split(",") if s}
        if shot_filter and "_shot" in sub.columns:
            sub = sub[sub["_shot"].isin(shot_filter)]
        subject_filter = request.args.get("subject", "").strip()
        if subject_filter and "subject_class" in sub.columns:
            sub = sub[sub["subject_class"].astype(str) == subject_filter]
        if request.args.get("hero") in ("1", "true", "yes") and "is_hero" in sub.columns:
            sub = sub[sub["is_hero"].astype(bool)]

        total = int(len(sub))
        sizes = ranked.groupby("burst_id").size()
        ct = _capture_times(rn) if request.args.get("sort") == "time" else None
        if ct is not None and len(sub):
            sub = sub.assign(_t=ct[sub.index]).sort_values("_t", na_position="last")
        else:
            sub = sub.sort_values(["burst_rank", "within_burst_rank"])
        sub = sub.iloc[offset:offset + limit]
        out: List[Dict[str, object]] = []
        for _, r in sub.iterrows():
            bid = int(r["burst_id"])
            out.append({
                "burst_rank": int(r["burst_rank"]),
                "burst_id": bid,
                "burst_size": int(sizes.get(bid, 1)),
                "rep_filename": r["filename"],
                "rep_path": r["path"],
                "shot_type": str(r.get("_shot", "") or ""),
                "subject_class": str(r.get("subject_class", "") or ""),
                "is_hero": bool(r.get("is_hero", False)),
                "score_s1": float(r.get("score_s1", 0.0)),
                "score_s12_after_hard": float(r.get("score_s12_after_hard", 0.0)),
                **_star_flags(rn, str(r["path"]), me),
                "n_starred": 1,
                "captured_at": _iso(float(r["_t"])) if "_t" in sub.columns else "",
                "match_within_burst_rank": int(r.get("within_burst_rank", 1)),
            })
        return jsonify({"bursts": out, "stats": _run_stats(rn), "version": rn.version,
                        "page": {"offset": offset, "limit": limit, "total_matching": total}})

    @app.route("/api/burst/<int:burst_id>")
    def api_burst(burst_id):
        rn = resolve_run()
        if rn is None:
            abort(404)
        ranked = rn.ranked
        sub = ranked[ranked["burst_id"] == burst_id].copy()
        if sub.empty:
            abort(404)
        n_all = len(sub)
        thr = _quality_threshold(rn, _top_pct())
        if thr is not None:
            col = SCORE_COL if SCORE_COL in sub.columns else "score_s1"
            kept = sub[sub[col] >= thr]
            if not kept.empty:          # never leave a burst with nothing in it
                sub = kept
        # Frames within a burst follow the same ordering as the grid by default:
        # in chronological mode, walking a burst best-first is jarring, because
        # the strip no longer reads as the sequence the photographer shot.
        ct = _capture_times(rn)
        if request.args.get("sort") == "time" and ct is not None:
            sub = sub.assign(_t=ct[sub.index]).sort_values("_t", na_position="last")
        else:
            sub = sub.sort_values("within_burst_rank")
        me = _viewer()
        st = _ensure_shot_types(rn)
        shot_by_idx = dict(zip(sub.index, st[sub.index])) if st is not None else {}
        # captured_at travels with every frame so the viewer can re-sort a burst
        # without another round trip.
        frames = []
        for idx, r in sub.iterrows():
            frames.append({
                "captured_at": _iso(float(ct[idx])) if ct is not None else "",
                "within_burst_rank": int(r["within_burst_rank"]),
                "is_representative": bool(r["is_representative"]),
                "filename": r["filename"],
                "path": r["path"],
                "shot_type": str(shot_by_idx.get(idx, "")),
                "score_s1": float(r["score_s1"]),
                "score_s12_after_hard": float(r["score_s12_after_hard"]),
                **_star_flags(rn, str(r["path"]), me),
                "tech_hard_reject": bool(r.get("tech_hard_reject", False)),
                "tech_hard_reject_reason": str(r.get("tech_hard_reject_reason", "")),
            })
        return jsonify({"burst_id": burst_id, "frames": frames,
                        "n_frames_total": n_all, "n_frames_hidden": n_all - len(frames)})

    @app.route("/api/search")
    def api_search():
        q = request.args.get("q", "").strip()
        offset = int(request.args.get("offset", "0"))
        limit = int(request.args.get("limit", "50"))
        empty_page = {"offset": offset, "limit": limit, "total_matching": 0}
        rn = resolve_run()
        if rn is None:
            return jsonify({"bursts": [], "page": empty_page, "query": q, "error": "No run selected."})
        if not q:
            return jsonify({"bursts": [], "page": empty_page, "query": q})
        ranked = rn.ranked
        if rn.emb is None or len(rn.emb) != len(ranked):
            return jsonify({"bursts": [], "page": empty_page, "query": q,
                            "error": "Search index unavailable for this run."})
        ctx = rn.emb_context or cfg.features.context_encoder
        try:
            with state["search_lock"]:
                te = _get_text_encoder(ctx)
                qvec = te.embed_texts([q])[0].astype(np.float32)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"bursts": [], "page": empty_page, "query": q,
                            "error": f"text encoder unavailable: {exc}"})

        work = ranked.copy()
        work["_sim"] = rn.emb @ qvec
        st = _ensure_shot_types(rn)
        if st is not None:
            work["_shot"] = st
        best = work.loc[work.groupby("burst_id")["_sim"].idxmax()].sort_values("_sim", ascending=False)

        shot_filter = {s for s in request.args.get("shot", "").split(",") if s}
        if shot_filter and "_shot" in best.columns:
            best = best[best["_shot"].isin(shot_filter)]
        subject_filter = request.args.get("subject", "").strip()
        if subject_filter and "subject_class" in best.columns:
            best = best[best["subject_class"].astype(str) == subject_filter]
        if request.args.get("hero") in ("1", "true", "yes") and "is_hero" in best.columns:
            best = best[best["is_hero"].astype(bool)]

        thr = _quality_threshold(rn, _top_pct())
        if thr is not None:
            col = SCORE_COL if SCORE_COL in best.columns else "score_s1"
            best = best[best[col] >= thr]
        star_counts = _starred_counts(rn)
        me = _viewer()
        # Search collapses each burst to its best-matching frame, which may not be
        # the starred one — so "starred" here means "this burst holds a star".
        if request.args.get("starred") in ("1", "true", "yes"):
            best = best[best["burst_id"].astype(int).isin(star_counts)]

        sizes = ranked.groupby("burst_id").size()
        total = int(len(best))
        page = best.iloc[offset:offset + limit]
        out: List[Dict[str, object]] = []
        for _, r in page.iterrows():
            bid = int(r["burst_id"])
            out.append({
                "burst_rank": int(r["burst_rank"]),
                "burst_id": bid,
                "burst_size": int(sizes.get(bid, 1)),
                "rep_filename": r["filename"],
                "rep_path": r["path"],
                "shot_type": str(r.get("_shot", "") or ""),
                "subject_class": str(r.get("subject_class", "") or ""),
                "is_hero": bool(r.get("is_hero", False)),
                "score_s1": float(r.get("score_s1", 0.0)),
                "score_s12_after_hard": float(r.get("score_s12_after_hard", 0.0)),
                "relevance": float(r["_sim"]),
                "match_within_burst_rank": int(r.get("within_burst_rank", 1)),
                **_star_flags(rn, str(r["path"]), me),
                "n_starred": int(star_counts.get(bid, 0)),
            })
        return jsonify({"bursts": out, "stats": _run_stats(rn), "version": rn.version, "query": q,
                        "page": {"offset": offset, "limit": limit, "total_matching": total}})

    # ----------------------------- image serving

    def _check_allowed(p: str) -> Path:
        try:
            real = Path(p).resolve()
        except Exception:
            abort(400)
        if str(real.parent) not in state["allowed_roots"]:
            abort(403)
        if not real.is_file():
            abort(404)
        return real

    # Uploaded files never change under a given path, so let browsers cache
    # thumbnails/previews aggressively — re-scrolling and re-opening become
    # instant instead of re-fetching over the (possibly slow) network.
    _CACHE_HDR = "public, max-age=604800, immutable"

    @app.route("/img")
    def img():
        real = _check_allowed(request.args.get("path", ""))
        from .imaging import is_raw, open_image
        if is_raw(real):
            # A browser can't render a .CR3, so serve the embedded preview — which
            # on modern bodies is full resolution anyway. /download still hands
            # back the untouched original.
            buf = io.BytesIO()
            with open_image(real) as im:
                im.convert("RGB").save(buf, format="JPEG", quality=92)
            resp = Response(buf.getvalue(), mimetype="image/jpeg")
            resp.headers["Cache-Control"] = _CACHE_HDR
            return resp
        mime, _ = mimetypes.guess_type(str(real))
        resp = send_file(str(real), mimetype=mime or "image/jpeg", conditional=True, max_age=604800)
        resp.headers["Cache-Control"] = _CACHE_HDR
        return resp

    # ----------------------------- downloads
    #
    # Every download hands back the ORIGINAL file, not a preview — the point of
    # the tool is producing deliverables. Bulk selections are zipped and streamed
    # rather than buffered: "download all starred" on a real shoot is routinely
    # several GB, which would blow up the process if assembled in memory.

    _dl_tokens: Dict[str, Dict[str, object]] = {}
    _dl_lock = threading.Lock()
    _DL_TTL = 3600.0          # seconds a prepared selection stays valid

    def _zip_arcnames(paths: Sequence[str]) -> List[str]:
        """Flat, collision-free names inside the archive.

        Frames from different subfolders can share a basename, and a zip with
        duplicate entries silently loses files on extraction.
        """
        used: Set[str] = set()
        out: List[str] = []
        for p in paths:
            name = Path(p).name
            if name in used:
                stem, dot, ext = name.rpartition(".")
                base, suffix = (stem, f".{ext}") if dot else (name, "")
                i = 2
                while f"{base}_{i}{suffix}" in used:
                    i += 1
                name = f"{base}_{i}{suffix}"
            used.add(name)
            out.append(name)
        return out

    def _prune_tokens() -> None:
        now = time.time()
        for t in [t for t, v in _dl_tokens.items() if float(v["expires"]) < now]:
            _dl_tokens.pop(t, None)

    @app.route("/download")
    def download_one():
        """One original file, as an attachment. Same allowlist as /img."""
        real = _check_allowed(request.args.get("path", ""))
        return send_file(str(real), as_attachment=True, download_name=real.name,
                         mimetype=mimetypes.guess_type(str(real))[0] or "image/jpeg")

    @app.route("/api/download/prepare", methods=["POST"])
    def api_download_prepare():
        """Validate a selection and reserve a token for it.

        Two steps rather than one because the browser must *navigate* to the zip
        for a native streaming download — it can't stream the response of a POST
        without buffering the whole archive in JS memory first.

        Body: ``{"scope": "starred"}`` or ``{"paths": [...]}``.
        """
        rn = resolve_run()
        if rn is None:
            return jsonify({"error": "No run selected."}), 400
        data = request.get_json(silent=True) or {}
        if str(data.get("scope", "")) == "starred":
            wanted = sorted(_paths_for_owner(rn, _viewer(), str(data.get("owner", "all"))))
        else:
            wanted = [str(p) for p in (data.get("paths") or [])]

        known = set(rn.ranked["path"].astype(str))
        seen: Set[str] = set()
        paths: List[str] = []
        missing = 0
        for p in wanted:
            if p in seen or p not in known:      # same trust boundary as /img
                continue
            seen.add(p)
            if Path(p).is_file():
                paths.append(p)
            else:
                missing += 1
        if not paths:
            return jsonify({"error": "Nothing available to download."}), 400

        total = sum(Path(p).stat().st_size for p in paths)
        row = db.get(rn.run_id)
        label = _safe_name(row["name"]) if row else "run"
        kind = "starred" if str(data.get("scope", "")) == "starred" else "selection"
        token = secrets.token_urlsafe(18)
        with _dl_lock:
            _prune_tokens()
            _dl_tokens[token] = {"paths": paths, "filename": f"{label}_{kind}.zip",
                                 "expires": time.time() + _DL_TTL}
        return jsonify({"ok": True, "token": token, "count": len(paths),
                        "bytes": int(total), "missing": missing,
                        "filename": f"{label}_{kind}.zip"})

    @app.route("/api/download/zip")
    def api_download_zip():
        """Stream a prepared selection as a zip.

        Entries are STORED, not deflated: JPEGs are already compressed, so
        deflate costs real CPU per file for roughly nothing, and stored entries
        let the archive stream out at disk speed.
        """
        token = request.args.get("token", "")
        with _dl_lock:
            _prune_tokens()
            entry = _dl_tokens.get(token)
        if entry is None:
            abort(404)
        paths: List[str] = list(entry["paths"])  # type: ignore[arg-type]
        names = _zip_arcnames(paths)

        class _Funnel:
            """Collects zipfile's writes so the view can yield them onward."""
            def __init__(self) -> None:
                self.buf = bytearray()

            def write(self, b: bytes) -> int:
                self.buf += b
                return len(b)

            def flush(self) -> None:
                pass

            def drain(self) -> bytes:
                out = bytes(self.buf)
                del self.buf[:]
                return out

        def generate():
            funnel = _Funnel()
            # zipfile writes data descriptors when the stream isn't seekable,
            # which is exactly what lets this be a generator.
            with zipfile.ZipFile(funnel, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
                for path, arc in zip(paths, names):
                    try:
                        with open(path, "rb") as src, zf.open(arc, "w") as dst:
                            while True:
                                chunk = src.read(1 << 20)
                                if not chunk:
                                    break
                                dst.write(chunk)
                                out = funnel.drain()
                                if out:
                                    yield out
                    except OSError:
                        continue  # a file vanished mid-download; skip it
                    out = funnel.drain()
                    if out:
                        yield out
            yield funnel.drain()  # central directory

        resp = Response(generate(), mimetype="application/zip")
        resp.headers["Content-Disposition"] = f'attachment; filename="{entry["filename"]}"'
        resp.headers["Cache-Control"] = "no-store"
        return resp

    _thumb_cache: Dict[Tuple[str, int], bytes] = {}
    _thumb_order: List[Tuple[str, int]] = []
    _thumb_lock = threading.Lock()
    _THUMB_MAX = 4096                    # entries
    _THUMB_MAX_BYTES = 256 * 1024 * 1024  # 256 MB
    _thumb_bytes = 0

    @app.route("/thumb")
    def thumb():
        """Downscaled JPEG. The grid requests one of a few fixed widths depending
        on its zoom level (320 → 960 px); the modal previews at w≈1600
        (~150–300 KB) instead of the multi-MB original — the big remote-viewing win.

        The LRU is bounded by *bytes* as well as entries: a 320 px thumb is ~7 KB
        but a 960 px one is ~200 KB, so an entry cap alone would let a zoomed-in
        grid hold most of a gigabyte of JPEG in the process.
        """
        nonlocal _thumb_bytes
        real = _check_allowed(request.args.get("path", ""))
        try:
            w = max(64, min(2048, int(request.args.get("w", "320"))))
        except ValueError:
            w = 320
        key = (str(real), w)
        hit = _thumb_cache.get(key)
        if hit is None:
            from .imaging import open_image
            with open_image(real) as im:  # context-managed so the FD is closed
                try:
                    im.draft("RGB", (w * 2, w * 2))  # fast partial JPEG decode
                except Exception:
                    pass
                im.thumbnail((w, w * 10))
                buf = io.BytesIO()
                im.convert("RGB").save(buf, format="JPEG", quality=80)
            hit = buf.getvalue()
            with _thumb_lock:
                if key not in _thumb_cache:
                    _thumb_cache[key] = hit
                    _thumb_order.append(key)
                    _thumb_bytes += len(hit)
                while _thumb_order and (
                    len(_thumb_order) > _THUMB_MAX or _thumb_bytes > _THUMB_MAX_BYTES
                ):
                    evicted = _thumb_cache.pop(_thumb_order.pop(0), None)
                    if evicted is not None:
                        _thumb_bytes -= len(evicted)
        return Response(hit, mimetype="image/jpeg",
                        headers={"Cache-Control": _CACHE_HDR})

    return app


def _raise_fd_limit() -> Optional[int]:
    """Raise the open-file soft limit toward the hard limit. Uploads and image
    serving open many descriptors at once; the default (often 1024) is easy to hit."""
    if resource is None:
        return None
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = hard if hard != resource.RLIM_INFINITY else 1_048_576
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        return resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    except Exception:
        return None


def _primary_ip() -> str:
    """Best-effort outward-facing IP so we can print a URL remote hosts can use."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # no packets sent; just picks the routing iface
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"


def _open_firewall(port: int) -> None:
    """Best-effort `ufw allow <port>/tcp`. Uses non-interactive sudo so it never
    hangs on a password prompt; on failure it just prints the manual command."""
    ufw = shutil.which("ufw")
    if not ufw:
        print(f"[firewall] ufw not found — if remote hosts can't connect, open port "
              f"{port}/tcp in whatever firewall this host uses.")
        return
    try:
        r = subprocess.run(["sudo", "-n", ufw, "allow", f"{port}/tcp"],
                           capture_output=True, text=True, timeout=15)
        if r.returncode == 0:
            print(f"[firewall] ufw allow {port}/tcp → {r.stdout.strip() or 'ok'}")
        else:
            print(f"[firewall] could not open the port automatically "
                  f"({(r.stderr or r.stdout).strip() or 'passwordless sudo unavailable'}).")
            print(f"[firewall] run once, manually:  sudo ufw allow {port}/tcp")
    except Exception as e:  # noqa: BLE001
        print(f"[firewall] auto-open failed ({e}); run once:  sudo ufw allow {port}/tcp")


def main() -> None:
    ap = argparse.ArgumentParser("image_filterer.server")
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--open-firewall", action="store_true",
                    help="Best-effort `ufw allow <port>/tcp` on launch (needs passwordless sudo).")
    args = ap.parse_args()
    cfg = default_config()
    if not cfg.model_path.exists():
        raise SystemExit(f"No production model at {cfg.model_path}. Run `python -m image_filterer.train` first.")

    fd = _raise_fd_limit()
    if fd:
        print(f"[fd] open-file limit: {fd}")

    listens_all = args.host in ("0.0.0.0", "::")
    if args.open_firewall and listens_all:
        _open_firewall(args.port)

    app = create_app(cfg)
    ip = _primary_ip() if listens_all else args.host
    print(f"Image Filterer (model={cfg.model_path.name})")
    print(f"  data:   {cfg.data_root}   (set IMAGE_FILTERER_DATA_ROOT to change)")
    print(f"  local:  http://127.0.0.1:{args.port}/")
    if listens_all:
        print(f"  remote: http://{ip}:{args.port}/")
        print(f"  if a remote host can't connect, open the port once:  sudo ufw allow {args.port}/tcp")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
