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

PACKAGE_DIR = Path(__file__).resolve().parent


class _Cancelled(Exception):
    """Raised inside the progress callback when a user cancels the ingestion."""


def _safe_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-")
    return s[:60] or "run"


def create_app(cfg: Optional[Config] = None) -> Flask:
    cfg = cfg or default_config()
    cfg.ensure_dirs()
    db = RunDB(cfg.db_path)

    app = Flask(__name__, template_folder=str(PACKAGE_DIR / "templates"))
    state: Dict[str, object] = {
        "run_id": None,
        "run_dir": None,
        "ranked": None,
        "bursts": None,
        "allowed_roots": set(),
        "emb": None,
        "emb_context": None,
        "text_encoder": None,
        "shot_types": None,
        "captured": None,                # epoch seconds per ranked row (chronological sort)
        "stars": set(),                  # starred frame paths for the loaded run
        "stars_lock": threading.Lock(),
        "search_lock": threading.Lock(),
        "ingest_lock": threading.Lock(),
        "active_ingest": None,           # run_id currently ingesting (global, single-flight)
        "cancel_event": threading.Event(),
    }

    # ----------------------------- stars
    #
    # Stars mark individual frames (not bursts) — the whole job of the ranker is
    # to say *which* frame of a burst is the keeper, so that's the unit worth
    # saving. They live in the run's own folder, next to ranked.csv, which makes
    # them shared by every viewer and portable with the run.

    def _load_stars(run_dir: Path) -> Set[str]:
        p = run_dir / "stars.json"
        if not p.exists():
            return set()
        try:
            return {str(x) for x in (json.loads(p.read_text()).get("starred") or [])}
        except Exception:  # noqa: BLE001 — a corrupt stars file must not block the run
            return set()

    def _save_stars() -> None:
        """Write via a temp file + rename so a crash mid-write can't truncate it."""
        run_dir = state["run_dir"]
        if run_dir is None:
            return
        payload = json.dumps({"starred": sorted(state["stars"])}, indent=1)
        tmp = Path(run_dir) / "stars.json.tmp"
        tmp.write_text(payload)
        tmp.replace(Path(run_dir) / "stars.json")

    def _starred_counts() -> Dict[int, int]:
        """burst_id → how many of its frames are starred."""
        ranked = state["ranked"]
        if ranked is None or not state["stars"]:
            return {}
        hit = ranked[ranked["path"].astype(str).isin(state["stars"])]
        return {int(b): int(n) for b, n in hit.groupby("burst_id").size().items()}

    # ----------------------------- run loading

    def load_run(run_id: int) -> bool:
        row = db.get(run_id)
        if not row or row["status"] != "ready":
            return False
        run_dir = cfg.runs_dir / row["folder"]
        if not (run_dir / "ranked.csv").exists():
            return False
        ranked = pd.read_csv(run_dir / "ranked.csv")
        bursts = pd.read_csv(run_dir / "bursts.csv")
        roots: Set[str] = set()
        for p in ranked["path"]:
            try:
                roots.add(str(Path(p).resolve().parent))
            except Exception:
                pass
        state.update({
            "run_id": run_id, "run_dir": run_dir, "ranked": ranked, "bursts": bursts,
            "allowed_roots": roots, "emb": None, "emb_context": None,
            "shot_types": None, "captured": None, "stars": _load_stars(run_dir),
        })
        # search index (row-aligned) if present
        npy = run_dir / "search_index.npy"
        if npy.exists():
            try:
                mat = np.load(npy)
                if mat.ndim == 2 and mat.shape[0] == len(ranked):
                    state["emb"] = mat.astype(np.float32)
                    meta = run_dir / "search_index.json"
                    ctx = json.loads(meta.read_text()).get("context_encoder", cfg.features.context_encoder) \
                        if meta.exists() else cfg.features.context_encoder
                    state["emb_context"] = ctx
            except Exception:
                pass
        if "shot_type" in ranked.columns and ranked["shot_type"].astype(str).str.len().gt(0).any():
            state["shot_types"] = ranked["shot_type"].astype(str).to_numpy()
        return True

    def _get_text_encoder(ctx: str):
        te = state["text_encoder"]
        if te is not None and getattr(te, "name", None) == ctx.strip().lower():
            return te
        from .encoders import build_text_encoder
        te = build_text_encoder(ctx)
        state["text_encoder"] = te
        return te

    def _ensure_shot_types() -> Optional[np.ndarray]:
        st = state["shot_types"]
        if st is not None:
            return st
        ranked = state["ranked"]
        emb = state["emb"]
        if ranked is None or emb is None or len(emb) != len(ranked):
            return None
        ctx = state["emb_context"] or cfg.features.context_encoder
        from .encoders import context_encoder_has_text_tower
        if not context_encoder_has_text_tower(ctx):
            return None
        from .shots import compute_shot_types
        with state["search_lock"]:
            te = _get_text_encoder(ctx)
            labels, _ = compute_shot_types(emb, ctx, cfg.shots, text_encoder=te)
        if not labels:
            return None
        st = np.asarray(labels)
        state["shot_types"] = st
        return st

    # ----------------------------- capture times (chronological sort)

    def _capture_times() -> Optional[np.ndarray]:
        """Epoch seconds per ``ranked`` row, or None if unobtainable.

        Capture time isn't a column in older runs — EXIF is read during burst
        clustering and then discarded — so this backfills it and caches the
        result beside the run. Reading DateTimeOriginal costs ~0.3 ms/frame, so
        a full 3,000-frame backfill is about a second, once.
        """
        if state["captured"] is not None:
            return state["captured"]
        ranked = state["ranked"]
        if ranked is None:
            return None
        run_dir = state["run_dir"]
        arr: Optional[np.ndarray] = None

        if "captured_at" in ranked.columns:      # written by newer ingests
            ts = pd.to_datetime(ranked["captured_at"], errors="coerce")
            if ts.notna().any():
                arr = ts.map(lambda x: x.timestamp() if pd.notna(x) else np.nan) \
                        .to_numpy(dtype=np.float64)

        side = (Path(run_dir) / "captured_at.json") if run_dir else None
        if arr is None and side is not None and side.exists():
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
            if side is not None:
                try:
                    side.write_text(json.dumps({
                        str(p): (None if np.isnan(v) else v)
                        for p, v in zip(ranked["path"], arr)}))
                except OSError:
                    pass
        state["captured"] = arr
        return arr

    def _burst_min_times() -> Dict[int, float]:
        """burst_id → capture time of its earliest frame."""
        ct = _capture_times()
        ranked = state["ranked"]
        if ct is None or ranked is None:
            return {}
        s = pd.Series(ct, index=ranked.index).groupby(ranked["burst_id"]).min()
        return {int(b): float(v) for b, v in s.items()}

    def _iso(ts: float) -> str:
        if ts is None or (isinstance(ts, float) and np.isnan(ts)):
            return ""
        return datetime.fromtimestamp(ts).isoformat(sep=" ", timespec="seconds")

    def _burst_rep_shot_map() -> Dict[int, str]:
        st = _ensure_shot_types()
        if st is None:
            return {}
        ranked = state["ranked"]
        rep = ranked.assign(_shot=st)
        rep = rep[rep["is_representative"].astype(bool)]
        return {int(b): str(s) for b, s in zip(rep["burst_id"], rep["_shot"])}

    # Load most-recent ready run on startup.
    for r in db.list(include_unready=False):
        if load_run(r["id"]):
            break

    # ----------------------------- ingestion

    def _do_ingest(run_id: int, folder: Path) -> None:
        cancel = state["cancel_event"]

        def cb(pct: float, msg: str) -> None:
            # Cooperative cancellation: this fires between feature chunks and at
            # each phase boundary, so Cancel takes effect within a chunk or two.
            if cancel.is_set():
                raise _Cancelled()
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

    @app.route("/api/ingest_path", methods=["POST"])
    def api_ingest_path():
        """Ingest a folder that already lives on THIS machine (a local path or a
        mounted share) — no upload. The fast path when the data is on the compute
        box: images are read/served in place, nothing is copied over the network."""
        data = request.get_json(silent=True) or {}
        folder = str(data.get("path", "")).strip()
        if not folder:
            return jsonify({"error": "No path given"}), 400
        fp = Path(folder).expanduser()
        if not fp.is_dir():
            return jsonify({"error": f"Not a directory on the server: {fp}"}), 400
        if not state["ingest_lock"].acquire(blocking=False):
            return jsonify({"error": "An ingestion is already running. Please wait."}), 409
        try:
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            name = str(data.get("name", "")).strip() or fp.name or f"Run {ts}"
            run_id = db.create(name, folder="")
            db.update(run_id, folder=f"run{run_id:04d}_{_safe_name(name)}")
            db.update(run_id, status="ingesting", message="Starting…")
            state["cancel_event"].clear()
            state["active_ingest"] = run_id
            # Ingest the folder in place (do NOT copy into the run dir).
            threading.Thread(target=_do_ingest, args=(run_id, fp), daemon=True).start()
            return jsonify({"ok": True, "run_id": run_id})
        except Exception:
            state["ingest_lock"].release()
            raise

    @app.route("/api/active")
    def api_active():
        """Global: is anyone ingesting right now? Polled by every client so the
        in-progress bar is visible to all users, not just whoever started it."""
        aid = state["active_ingest"]
        if aid is None:
            return jsonify({"active": None})
        row = db.get(int(aid))
        if not row or row["status"] != "ingesting":
            return jsonify({"active": None})
        return jsonify({"active": {
            "run_id": row["id"], "name": row["name"],
            "progress": row["progress"], "message": row["message"],
        }})

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
        cur = None
        if state["run_id"] is not None:
            row = db.get(int(state["run_id"]))
            if row:
                cur = {"id": row["id"], "name": row["name"],
                       "n_images": row["n_images"], "n_bursts": row["n_bursts"]}
        return jsonify({"current": cur, "runs": _runs_payload()})

    @app.route("/api/runs")
    def api_runs():
        return jsonify({"runs": _runs_payload(),
                        "current_id": state["run_id"]})

    @app.route("/api/select_run", methods=["POST"])
    def api_select_run():
        run_id = int((request.get_json(silent=True) or {}).get("run_id")
                     or request.form.get("run_id", 0))
        if not load_run(run_id):
            return jsonify({"error": "Run not found or not ready."}), 404
        return jsonify({"ok": True, "run_id": run_id})

    # ----------------------------- browse / search

    @app.route("/")
    def index():
        return send_file(PACKAGE_DIR / "templates" / "index.html")

    def _run_stats() -> Dict[str, object]:
        ranked = state["ranked"]
        bursts = state["bursts"]
        row = db.get(int(state["run_id"])) if state["run_id"] is not None else None
        return {
            "n_frames": int(len(ranked)),
            "n_bursts": int(bursts["burst_id"].nunique()),
            "n_multi": int((bursts["burst_size"] > 1).sum()),
            "n_starred": int(len(state["stars"])),
            "run_name": row["name"] if row else "",
            "run_id": state["run_id"],
        }

    @app.route("/api/bursts")
    def api_bursts():
        if state["ranked"] is None:
            return jsonify({"bursts": [], "stats": None,
                            "page": {"offset": 0, "limit": 50, "total_matching": 0}})
        bursts: pd.DataFrame = state["bursts"]  # type: ignore
        shot_filter = {s for s in request.args.get("shot", "").split(",") if s}
        subject_filter = request.args.get("subject", "").strip()   # "" | people | stage
        hero_only = request.args.get("hero") in ("1", "true", "yes")
        offset = int(request.args.get("offset", "0"))
        limit = int(request.args.get("limit", "50"))

        df = bursts.copy()
        rep_shot = _burst_rep_shot_map()
        if rep_shot:
            df = df.assign(_shot=df["burst_id"].map(lambda b: rep_shot.get(int(b), "")))
            if shot_filter:
                df = df[df["_shot"].isin(shot_filter)]
        if subject_filter and "subject_class" in df.columns:
            df = df[df["subject_class"].astype(str) == subject_filter]
        if hero_only and "is_hero" in df.columns:
            df = df[df["is_hero"].astype(bool)]

        total = len(df)
        # sort=time orders bursts by when they were shot rather than by score.
        # Bursts stay collapsed either way — this only changes their order.
        tmin: Dict[int, float] = {}
        if request.args.get("sort") == "time":
            tmin = _burst_min_times()
        if tmin:
            df = df.assign(_t=df["burst_id"].map(lambda b: tmin.get(int(b), float("nan"))))
            df = df.sort_values(["_t", "burst_rank"], na_position="last")
        else:
            df = df.sort_values("burst_rank")
        df = df.iloc[offset:offset + limit]
        star_counts = _starred_counts()
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
                "starred": str(r["path"]) in state["stars"],
                "n_starred": int(star_counts.get(int(r["burst_id"]), 0)),
                "captured_at": _iso(tmin.get(int(r["burst_id"]))) if tmin else "",
            })
        return jsonify({"bursts": out, "stats": _run_stats(),
                        "page": {"offset": offset, "limit": limit, "total_matching": int(total)}})

    @app.route("/api/stars")
    def api_stars():
        """Every starred path in the loaded run — the client mirrors this set so
        tiles can render their star state without a request per tile."""
        return jsonify({"starred": sorted(state["stars"]), "count": len(state["stars"])})

    @app.route("/api/star", methods=["POST"])
    def api_star():
        """Star / unstar one frame. ``{path, starred}`` → the new count."""
        if state["ranked"] is None:
            return jsonify({"error": "No run selected."}), 400
        data = request.get_json(silent=True) or {}
        path = str(data.get("path", ""))
        want = bool(data.get("starred", True))
        # Only frames belonging to this run may be starred — same trust boundary
        # as image serving, so a stray path can't be written into stars.json.
        if path not in set(state["ranked"]["path"].astype(str)):
            return jsonify({"error": "Unknown frame for this run."}), 404
        with state["stars_lock"]:
            if want:
                state["stars"].add(path)
            else:
                state["stars"].discard(path)
            _save_stars()
            count = len(state["stars"])
        return jsonify({"ok": True, "path": path, "starred": want, "count": count})

    @app.route("/api/stars/clear", methods=["POST"])
    def api_stars_clear():
        """Unstar everything in the loaded run. Stars are shared, so the UI
        confirms first — this can discard a colleague's picks, not just yours."""
        if state["ranked"] is None:
            return jsonify({"error": "No run selected."}), 400
        with state["stars_lock"]:
            cleared = len(state["stars"])
            state["stars"].clear()
            _save_stars()
        return jsonify({"ok": True, "cleared": cleared, "count": 0})

    @app.route("/api/starred")
    def api_starred():
        """The saved-for-later list: one entry per starred FRAME, not per burst.

        Shaped like ``/api/bursts`` rows so the grid renders it unchanged. The
        shot / subject / hero filters still AND in, matching every other view.
        """
        offset = int(request.args.get("offset", "0"))
        limit = int(request.args.get("limit", "50"))
        page_empty = {"offset": offset, "limit": limit, "total_matching": 0}
        ranked = state["ranked"]
        if ranked is None:
            return jsonify({"bursts": [], "stats": None, "page": page_empty})

        sub = ranked[ranked["path"].astype(str).isin(state["stars"])].copy()
        st = _ensure_shot_types()
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
        ct = _capture_times() if request.args.get("sort") == "time" else None
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
                "starred": True,
                "n_starred": 1,
                "captured_at": _iso(float(r["_t"])) if "_t" in sub.columns else "",
                "match_within_burst_rank": int(r.get("within_burst_rank", 1)),
            })
        return jsonify({"bursts": out, "stats": _run_stats(),
                        "page": {"offset": offset, "limit": limit, "total_matching": total}})

    @app.route("/api/burst/<int:burst_id>")
    def api_burst(burst_id):
        ranked: pd.DataFrame = state["ranked"]  # type: ignore
        if ranked is None:
            abort(404)
        sub = ranked[ranked["burst_id"] == burst_id].copy()
        if sub.empty:
            abort(404)
        sub = sub.sort_values("within_burst_rank")
        st = _ensure_shot_types()
        shot_by_idx = dict(zip(sub.index, st[sub.index])) if st is not None else {}
        frames = []
        for idx, r in sub.iterrows():
            frames.append({
                "within_burst_rank": int(r["within_burst_rank"]),
                "is_representative": bool(r["is_representative"]),
                "filename": r["filename"],
                "path": r["path"],
                "shot_type": str(shot_by_idx.get(idx, "")),
                "score_s1": float(r["score_s1"]),
                "score_s12_after_hard": float(r["score_s12_after_hard"]),
                "starred": str(r["path"]) in state["stars"],
                "tech_hard_reject": bool(r.get("tech_hard_reject", False)),
                "tech_hard_reject_reason": str(r.get("tech_hard_reject_reason", "")),
            })
        return jsonify({"burst_id": burst_id, "frames": frames})

    @app.route("/api/search")
    def api_search():
        q = request.args.get("q", "").strip()
        offset = int(request.args.get("offset", "0"))
        limit = int(request.args.get("limit", "50"))
        empty_page = {"offset": offset, "limit": limit, "total_matching": 0}
        if state["ranked"] is None:
            return jsonify({"bursts": [], "page": empty_page, "query": q, "error": "No run selected."})
        if not q:
            return jsonify({"bursts": [], "page": empty_page, "query": q})
        ranked: pd.DataFrame = state["ranked"]  # type: ignore
        if state["emb"] is None or len(state["emb"]) != len(ranked):
            return jsonify({"bursts": [], "page": empty_page, "query": q,
                            "error": "Search index unavailable for this run."})
        ctx = state["emb_context"] or cfg.features.context_encoder
        try:
            with state["search_lock"]:
                te = _get_text_encoder(ctx)
                qvec = te.embed_texts([q])[0].astype(np.float32)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"bursts": [], "page": empty_page, "query": q,
                            "error": f"text encoder unavailable: {exc}"})

        emb: np.ndarray = state["emb"]  # type: ignore
        work = ranked.copy()
        work["_sim"] = emb @ qvec
        st = _ensure_shot_types()
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

        star_counts = _starred_counts()
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
                "starred": str(r["path"]) in state["stars"],
                "n_starred": int(star_counts.get(bid, 0)),
            })
        return jsonify({"bursts": out, "stats": _run_stats(), "query": q,
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
        if state["ranked"] is None:
            return jsonify({"error": "No run selected."}), 400
        data = request.get_json(silent=True) or {}
        if str(data.get("scope", "")) == "starred":
            wanted = sorted(state["stars"])
        else:
            wanted = [str(p) for p in (data.get("paths") or [])]

        known = set(state["ranked"]["path"].astype(str))
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
        row = db.get(int(state["run_id"])) if state["run_id"] is not None else None
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
            with Image.open(str(real)) as im:  # context-managed so the FD is closed
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
