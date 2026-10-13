"""Inference-only ingestion of an unlabeled image folder.

Produces, in ``run_dir``:
  ranked.csv         one row per unique frame (scores, burst columns, shot_type)
  bursts.csv         one row per burst (representative), ranked
  search_index.npy   full-frame embeddings for semantic search
  config.json        the config used

No labels, no training — the shipped production model scores every frame. A
``progress_cb(pct, message)`` is invoked throughout so a caller can surface a
progress bar.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .ranker import feature_blocks, load_any, predict, stack_features
from .pipeline import (atomic_write, output_dir, publish_outputs, attach_burst_columns, scan_images,
                       write_bursts_csv, write_search_index)
from .technical import combine_stage_scores, reports_to_dicts, technical_scores

from .cache_io import MEMORY
from .common import load_or_extract
from .config import Config

# progress_cb(percent 0..100, message)
StatusCB = Callable[[float, str], None]


def _noop(pct: float, msg: str) -> None:  # pragma: no cover
    pass


def _near_identical_pairs(embs: np.ndarray, threshold: float, device=None) -> Dict[int, List[int]]:
    """{i: [earlier j with cosine >= threshold]} — on the GPU when there is
    one (10k photos: ~0.1 s instead of ~3 s of CPU block compares)."""
    dev = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    e = torch.from_numpy(embs).to(dev)
    out: Dict[int, List[int]] = {}
    step = 2048
    with torch.no_grad():
        for start in range(0, len(embs), step):
            stop = min(start + step, len(embs))
            sim = e[start:stop] @ e[:stop].T                   # rows vs. everything before them
            rows = torch.arange(start, stop, device=dev)[:, None]
            hit = (sim >= threshold) & (torch.arange(stop, device=dev)[None, :] < rows)
            for i, j in torch.nonzero(hit).cpu().tolist():
                out.setdefault(start + i, []).append(j)
    del e, sim, hit
    if dev.type == "cuda":
        # Buffer sizes grow with the run, so cached blocks from the last
        # ingest never fit the next one: ~1 GB held by 10k photos otherwise.
        torch.cuda.empty_cache()
    return out


def _dedup(bundles, paths, threshold: float = 0.9995, device=None) -> Tuple[list, List[Path]]:
    """Drop byte-identical (sha1) then pixel-identical (embedding) duplicates."""
    seen: Dict[str, int] = {}
    keep: List[int] = []
    for i, b in enumerate(bundles):
        if b.sha1 not in seen:
            seen[b.sha1] = i
            keep.append(i)
    bundles = [bundles[i] for i in keep]
    paths = [paths[i] for i in keep]
    if not bundles:
        return bundles, paths

    embs = np.stack([b.emb_full for b in bundles]).astype(np.float32)
    embs /= np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-12)
    # Greedy representative rule: a frame is dropped when an earlier *kept*
    # frame is pixel-identical to it.
    kept = np.ones(len(bundles), dtype=bool)
    for i, earlier in sorted(_near_identical_pairs(embs, threshold, device).items()):
        if kept[earlier].any():
            kept[i] = False
    idx = np.flatnonzero(kept)
    return [bundles[i] for i in idx], [paths[i] for i in idx]


def analyze_scenes(bundles, cfg: Config, device, analyzer=None, progress=None, stats=None):
    """SceneData for every bundle (cached by content). Decoding happens on
    worker threads so the GPU (pose) never waits on JPEG decoding; cache hits
    skip the file entirely."""
    from .cache_io import FeatureCache
    from .imaging import open_image, ordered_map
    from .scene import SceneAnalyzer
    owns = analyzer is None
    an = analyzer or SceneAnalyzer(cfg.scene, FeatureCache(cfg.cache_dir),
                                   device="0" if device.type == "cuda" else None)

    def load(b):
        if an.cached(b.sha1):
            return b, None
        with open_image(b.path) as src:
            return b, an.prepare(src)

    out = []
    computed = sum(1 for b in bundles if not an.cached(b.sha1))
    if stats is not None:
        stats["scene_computed"] = computed
    try:
        n = len(bundles)
        for i, (b, prep) in enumerate(ordered_map(load, bundles)):
            out.append(an.analyze(b.sha1, prep if prep is not None else (lambda b=b: open_image(b.path))))
            if progress and (i + 1) % 50 == 0:
                progress(i + 1, n)
    finally:
        if owns:
            an.close()
    return out


def ingest_folder(
    folder: Path,
    cfg: Config,
    run_dir: Path,
    *,
    progress_cb: Optional[StatusCB] = None,
    image_paths: Optional[List[Path]] = None,
    feature_extractor=None,
    scene_analyzer=None,
    prevalidated: Optional[Dict[str, list]] = None,
) -> Dict[str, int]:
    user_cb = progress_cb or _noop
    run_dir.mkdir(parents=True, exist_ok=True)
    # Per-phase timing, taken from the progress messages: a phase runs from its
    # first message until the next phase's. Returned as result["timings"].
    import re
    import time as _time
    timings: Dict[str, float] = {}
    t_start = _time.perf_counter()
    phase = {"name": None, "t": t_start}

    def cb(pct: float, msg: str) -> None:
        name = re.sub(r"\s+", " ", re.sub(r"\d+(/\d+)?", "", msg.split("…")[0])).strip()
        name = "Extracting features" if name.startswith("Found") else name
        now = _time.perf_counter()
        if name != phase["name"]:
            if phase["name"]:
                timings[phase["name"]] = timings.get(phase["name"], 0.0) + now - phase["t"]
            phase["name"], phase["t"] = name, now
        user_cb(pct, msg)

    cb(1, "Scanning folder…")
    paths = scan_images(folder, candidates=image_paths)
    # RAW+JPEG pairs: the JPEG is ranked; its RAW is what exports/downloads hand out.
    from .pipeline import list_photo_files, raw_twins
    twins = raw_twins(list_photo_files(folder, candidates=image_paths))
    if not paths:
        raise RuntimeError("No images found in the uploaded folder.")
    n_found = len(paths)

    # Drop files PIL can't decode (corrupt / truncated / unsupported) so one bad
    # file doesn't abort the whole run. Report how many were skipped.
    cb(2, f"Validating {n_found} images…")
    from .imaging import validated_images
    vstats: Dict[str, float] = {}
    good = validated_images(paths, run_dir, prevalidated, vstats)
    n_bad = n_found - len(good)
    if not good:
        raise RuntimeError(
            f"None of the {n_found} files could be decoded as images "
            "(corrupt upload, or an unsupported format such as RAW/CR3)."
        )
    paths = good
    skip_note = f" ({n_bad} unreadable skipped)" if n_bad else ""
    cb(3, f"Found {len(paths)} images{skip_note} — extracting features…")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kind, model = load_any(cfg.model_path, device)
    meta = model.meta if kind == "v2" else model[2]
    for key in ("context_encoder", "face_encoder"):
        if meta.get(key) and meta[key] != getattr(cfg.features, key):
            raise ValueError(f"Ranker requires {key}={meta[key]!r}; runtime encoder differs.")

    # Features: 5% → 60% of the bar, driven by extraction chunks.
    def feat_progress(done: int, total: int) -> None:
        cb(5 + 55 * done / max(1, total), f"Extracting features… {done}/{total}")

    fx_before = dict(feature_extractor.stats) if feature_extractor is not None else None
    share_scene = feature_extractor is not None and scene_analyzer is not None and cfg.scene.enabled
    if share_scene:
        feature_extractor.scene_analyzer = scene_analyzer
    try:
        bundles = load_or_extract(cfg, paths, desc="ingest", progress_cb=feat_progress,
                                  extractor=feature_extractor)
        if share_scene:
            feature_extractor.wait_scenes()
    finally:
        if share_scene:
            feature_extractor.scene_analyzer = None
            try:
                feature_extractor.wait_scenes()    # never leave work running into the next ingest
            except Exception:  # noqa: BLE001 — already failing; the first error is the one to report
                pass
    detail: Dict[str, float] = {"validate_decoded": vstats.get("decoded", 0)}
    if fx_before is not None:
        for k, v in feature_extractor.stats.items():
            detail[f"features_{k}"] = round(v - fx_before[k], 3)

    cb(61, "De-duplicating…")
    bundles, paths = _dedup(bundles, paths, cfg.bursts.pixel_identical_cosine, device)

    # People, pose and the speaker's expression (scene.py). Before scoring:
    # the ranker uses the expression.
    scenes = None
    if cfg.scene.enabled:
        cb(62, "Finding the speaker…")
        scenes = analyze_scenes(bundles, cfg, device, scene_analyzer,
                                lambda i, n: cb(62 + 16 * i / max(1, n), f"Finding the speaker… {i}/{n}"),
                                stats=detail)

    cb(79, "Scoring images…")
    if kind == "v2":
        s1 = model.score(feature_blocks(bundles, scenes), device)
    else:
        # Stack features with the SAME feature-group flags the model was trained on,
        # so the vector layout matches the weights regardless of runtime config.
        net, saved_cfg, _ = model
        from .config import RankerConfig
        s1 = predict(net, stack_features(bundles, RankerConfig(**saved_cfg)), device).astype(np.float32)

    tech_reports, s2_raw = technical_scores(bundles, cfg.technical)
    tech_dicts = reports_to_dicts(tech_reports)
    s12 = combine_stage_scores(s1, s2_raw, cfg.technical.alpha)
    hard_reject = np.array([r.hard_reject for r in tech_reports], dtype=bool)
    s12_after_hard = s12.copy()
    s12_after_hard[hard_reject] = -1e9

    cb(82, "Tagging shot scale…")
    shot_types = None
    shot_closeness = None
    if cfg.shots.enabled and scenes is not None:
        from .shots import shot_from_pose
        shot_types = [shot_from_pose(sc, cfg.shots)[0] for sc in scenes]
    elif cfg.shots.enabled:
        # No pose data: the older SigLIP zero-shot axis.
        from .encoders import context_encoder_has_text_tower
        from .cache_io import FeatureCache
        from .shots import compute_shot_types
        if context_encoder_has_text_tower(cfg.features.context_encoder):
            emb_all = np.stack([b.emb_full for b in bundles]).astype(np.float32)
            labels, closeness = compute_shot_types(
                emb_all, cfg.features.context_encoder, cfg.shots,
                cache=FeatureCache(cfg.cache_dir))
            if labels:
                shot_types, shot_closeness = labels, closeness

    # Subject (people/stage) + Hero (the speaker alone against a dark background).
    scene = None
    if scenes is not None:
        from .scene import classify_scene
        scene = []
        for sc in scenes:
            subj, hero = classify_scene(sc, cfg.scene)
            b = sc.subject_box()
            scene.append({"subject_class": subj, "is_hero": hero,
                          "person_area": float((b[2] - b[0]) * (b[3] - b[1])) if b is not None else 0.0,
                          "bg_bright": float(sc.bg_lit)})

    # Capture time, so the viewer can offer chronological order without having to
    # re-read EXIF for the whole run later. Falls back to mtime when a frame has
    # no DateTimeOriginal. ~0.3 ms/frame — worth reusing when re-ingesting a
    # folder we've already seen, which is the hot-folder case.
    from .bursts import read_frame_meta
    # Camera id rides along: auto-export needs it to tell when a burst is finished.
    prior_times: Dict[str, Tuple[str, str, str]] = {}
    prev_csv = output_dir(run_dir) / "ranked.csv"
    if prev_csv.exists():
        try:
            prev = pd.read_csv(prev_csv)
            if {"captured_at", "sha1", "camera_id"}.issubset(prev.columns):
                prior_times = {str(p): (str(sha), str(t), "" if pd.isna(c) else str(c)) for p, sha, t, c in
                               zip(prev["path"], prev["sha1"], prev["captured_at"], prev["camera_id"])
                               if pd.notna(t) and str(t)}
        except Exception:  # noqa: BLE001 — a stale CSV just means we re-read EXIF
            prior_times = {}
    captured: List[str] = []
    cameras: List[str] = []
    for p, bundle in zip(paths, bundles):
        known = prior_times.get(str(p))
        if known and known[0] == bundle.sha1:
            captured.append(known[1])
            cameras.append(known[2])
            continue
        meta = read_frame_meta(p)
        if meta is not None:
            captured.append(meta.captured_at.isoformat(sep=" "))
            cameras.append(meta.camera_id)
        else:
            cameras.append("")
            try:
                from datetime import datetime as _dt
                captured.append(_dt.fromtimestamp(p.stat().st_mtime).isoformat(sep=" "))
            except OSError:
                captured.append("")

    rows = []
    for i in range(len(bundles)):
        rows.append({
            "filename": paths[i].name,
            "path": str(paths[i]),
            "raw_path": twins.get(str(paths[i]), ""),
            "sha1": bundles[i].sha1,
            "captured_at": captured[i],
            "camera_id": cameras[i],
            "score_s1": float(s1[i]),
            "score_s12": float(s12[i]),
            "score_s12_after_hard": float(s12_after_hard[i]),
            "score_s3_final": float(s12_after_hard[i]),
            "shot_type": shot_types[i] if shot_types is not None else "",
            "shot_closeness": float(shot_closeness[i]) if shot_closeness is not None else 0.0,
            "subject_class": scene[i]["subject_class"] if scene is not None else "",
            "is_hero": scene[i]["is_hero"] if scene is not None else False,
            "person_area": scene[i]["person_area"] if scene is not None else 0.0,
            "bg_bright": scene[i]["bg_bright"] if scene is not None else 0.0,
            **tech_dicts[i],
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("score_s3_final", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    cb(88, "Clustering bursts…")
    embeddings_map = {Path(str(bundles[i].path)): bundles[i].emb_full for i in range(len(bundles))}
    df = attach_burst_columns(df, paths=df["path"].tolist(), cfg=cfg, embeddings=embeddings_map)

    cb(94, "Writing outputs…")
    n_bursts = int(df["burst_id"].nunique())
    def write_outputs(stage):
        atomic_write(stage / "ranked.csv", lambda t: df.to_csv(t, index=False))
        write_bursts_csv(df, stage, cfg)
        write_search_index(stage, df, embeddings_map, cfg)
        (stage / "config.json").write_text(json.dumps(asdict(cfg), default=str, indent=2))

    publish_outputs(run_dir, write_outputs, n_images=int(len(df)), n_bursts=n_bursts)
    skip_note = f"; {n_bad} unreadable skipped" if n_bad else ""
    cb(100, f"Done — {len(df)} images in {n_bursts} bursts{skip_note}.")
    timings.pop("Done", None)
    result = {"n_images": int(len(df)), "n_bursts": n_bursts, "n_skipped": int(n_bad),
              "timings": {k: round(v, 3) for k, v in timings.items()}, "detail": detail,
              "memory_cache": MEMORY.stats()}
    # One line per ingest in the run's folder: where the time went, readable
    # without the web app (a watched shoot re-ingests on every arrival).
    try:
        import datetime as _dt
        with open(run_dir / "ingest_log.jsonl", "a") as f:
            f.write(json.dumps({"at": _dt.datetime.now().isoformat(timespec="seconds"),
                                "seconds": round(_time.perf_counter() - t_start, 2), **result}) + "\n")
    except OSError:
        pass
    return result
