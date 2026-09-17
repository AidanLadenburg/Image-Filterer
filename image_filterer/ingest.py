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

from .ranker import load_model, predict, stack_features
from .pipeline import (atomic_write, output_dir, publish_outputs, attach_burst_columns, scan_images,
                       write_bursts_csv, write_search_index)
from .technical import combine_stage_scores, reports_to_dicts, technical_scores

from .common import load_or_extract
from .config import Config

# progress_cb(percent 0..100, message)
StatusCB = Callable[[float, str], None]


def _noop(pct: float, msg: str) -> None:  # pragma: no cover
    pass


def _dedup(bundles, paths, threshold: float = 0.9995) -> Tuple[list, List[Path]]:
    """Drop byte-identical (sha1) then pixel-identical (embedding) duplicates."""
    seen: Dict[str, int] = {}
    keep: List[int] = []
    for i, b in enumerate(bundles):
        if b.sha1 not in seen:
            seen[b.sha1] = i
            keep.append(i)
    bundles = [bundles[i] for i in keep]
    paths = [paths[i] for i in keep]

    embs = np.stack([b.emb_full for b in bundles]).astype(np.float32)
    embs /= np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-12)
    # Compare bounded blocks against the frames already retained. Preserve the
    # greedy representative rule without allocating an N x N matrix.
    keep2: List[int] = []
    block_size = 256
    for start in range(0, len(bundles), block_size):
        stop = min(start + block_size, len(bundles))
        duplicate = np.zeros(stop - start, dtype=bool)
        for k in range(0, len(keep2), block_size):
            prior = keep2[k:k + block_size]
            duplicate |= (embs[start:stop] @ embs[prior].T >= threshold).any(axis=1)
        local = embs[start:stop] @ embs[start:stop].T
        chosen = []
        for j in range(stop - start):
            if not duplicate[j] and not any(local[j, k] >= threshold for k in chosen):
                chosen.append(j)
        keep2.extend(start + j for j in chosen)
    return [bundles[i] for i in keep2], [paths[i] for i in keep2]


def ingest_folder(
    folder: Path,
    cfg: Config,
    run_dir: Path,
    *,
    progress_cb: Optional[StatusCB] = None,
    image_paths: Optional[List[Path]] = None,
) -> Dict[str, int]:
    cb = progress_cb or _noop
    run_dir.mkdir(parents=True, exist_ok=True)

    cb(1, "Scanning folder…")
    paths = scan_images(folder, candidates=image_paths)
    if not paths:
        raise RuntimeError("No images found in the uploaded folder.")
    n_found = len(paths)

    # Drop files PIL can't decode (corrupt / truncated / unsupported) so one bad
    # file doesn't abort the whole run. Report how many were skipped.
    cb(2, f"Validating {n_found} images…")
    from .imaging import validated_images
    good = validated_images(paths, run_dir)
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
    model, saved_cfg, meta = load_model(cfg.model_path, device)
    for key in ("context_encoder", "face_encoder"):
        if meta.get(key) and meta[key] != getattr(cfg.features, key):
            raise ValueError(f"Ranker requires {key}={meta[key]!r}; runtime encoder differs.")

    # Features: 5% → 65% of the bar, driven by extraction chunks.
    def feat_progress(done: int, total: int) -> None:
        cb(5 + 60 * done / max(1, total), f"Extracting features… {done}/{total}")

    bundles = load_or_extract(cfg, paths, desc="ingest", progress_cb=feat_progress)

    cb(66, "De-duplicating…")
    bundles, paths = _dedup(bundles, paths, cfg.bursts.pixel_identical_cosine)

    cb(70, "Scoring images…")
    # Stack features with the SAME feature-group flags the model was trained on,
    # so the vector layout matches the weights regardless of runtime config.
    from .config import RankerConfig
    rc = RankerConfig(**saved_cfg)
    X = stack_features(bundles, rc)
    s1 = predict(model, X, device).astype(np.float32)

    tech_reports, s2_raw = technical_scores(bundles, cfg.technical)
    tech_dicts = reports_to_dicts(tech_reports)
    s12 = combine_stage_scores(s1, s2_raw, cfg.technical.alpha)
    hard_reject = np.array([r.hard_reject for r in tech_reports], dtype=bool)
    s12_after_hard = s12.copy()
    s12_after_hard[hard_reject] = -1e9

    cb(80, "Tagging shot scale…")
    shot_types = None
    shot_closeness = None
    if cfg.shots.enabled:
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

    # Scene tagging: Subject (people/stage) + Hero (prominent solo, clean bg).
    scene = None
    if cfg.scene.enabled:
        cb(84, "Detecting people / scene…")
        from .cache_io import FeatureCache
        from .scene import SceneAnalyzer, classify_scene
        from .imaging import open_image
        an = SceneAnalyzer(cfg.scene, FeatureCache(cfg.cache_dir),
                           device="0" if device.type == "cuda" else None)
        scene = []
        n = len(bundles)
        try:
            for i, b in enumerate(bundles):
                vec = an.raw(b.sha1, lambda b=b: open_image(b.path))
                subj, hero = classify_scene(vec, cfg.scene)
                scene.append({"subject_class": subj, "is_hero": hero,
                              "person_area": float(vec[0]), "bg_bright": float(vec[3])})
                if (i + 1) % 100 == 0:
                    cb(84 + 6 * (i + 1) / n, f"Detecting people… {i+1}/{n}")
        finally:
            an.close()

    # Capture time, so the viewer can offer chronological order without having to
    # re-read EXIF for the whole run later. Falls back to mtime when a frame has
    # no DateTimeOriginal. ~0.3 ms/frame — worth reusing when re-ingesting a
    # folder we've already seen, which is the hot-folder case.
    from .bursts import read_frame_meta
    prior_times: Dict[str, Tuple[str, str]] = {}
    prev_csv = output_dir(run_dir) / "ranked.csv"
    if prev_csv.exists():
        try:
            prev = pd.read_csv(prev_csv)
            if {"captured_at", "sha1"}.issubset(prev.columns):
                prior_times = {str(p): (str(sha), str(t)) for p, sha, t in
                               zip(prev["path"], prev["sha1"], prev["captured_at"])
                               if pd.notna(t) and str(t)}
        except Exception:  # noqa: BLE001 — a stale CSV just means we re-read EXIF
            prior_times = {}
    captured: List[str] = []
    for p, bundle in zip(paths, bundles):
        known = prior_times.get(str(p))
        if known and known[0] == bundle.sha1:
            captured.append(known[1])
            continue
        meta = read_frame_meta(p)
        if meta is not None:
            captured.append(meta.captured_at.isoformat(sep=" "))
        else:
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
            "sha1": bundles[i].sha1,
            "captured_at": captured[i],
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
    return {"n_images": int(len(df)), "n_bursts": n_bursts, "n_skipped": int(n_bad)}
