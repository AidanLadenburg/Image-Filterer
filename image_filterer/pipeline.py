"""Shared pipeline steps used by both training and ingestion.

These sit between the per-frame scorers (``ranker``, ``technical``) and the run
outputs: burst assignment and the search index. Both training and ingest call
them, so they live here rather than in either.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .bursts import assign_burst_ids
from .config import IMAGE_EXTS, Config


def atomic_write(path: Path, write: "Callable[[Path], None]") -> Path:
    """Write via a temp file in the same directory, then rename into place.

    Run outputs are read by the server while it is serving. A direct write
    leaves a window where ``ranked.csv`` is half-written or disagrees with
    ``search_index.npy``; ``os.replace`` is atomic on POSIX, so a reader sees
    either the old file or the new one and never a partial one.
    """
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return path


def output_dir(run_dir: Path) -> Path:
    """Resolve one immutable published snapshot; support pre-generation runs."""
    current = Path(run_dir) / "current"
    return current.resolve(strict=True) if current.is_symlink() else Path(run_dir)


def read_version(run_dir: Path) -> int:
    path = output_dir(run_dir) / "version.json"
    if not path.exists():
        return 0
    return int(json.loads(path.read_text()).get("version", 0))


def publish_outputs(run_dir: Path, write: Callable[[Path], None], **counts) -> int:
    """Build an entire generation before atomically switching the current link.

    A crash before the switch leaves the old generation available. A crash
    after it leaves a complete new generation. Readers resolve the link once.
    Top-level aliases preserve the familiar CSV paths for external tools.
    """
    run_dir = Path(run_dir).resolve()
    generations = run_dir / ".generations"
    generations.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="generation-", dir=generations))
    previous = output_dir(run_dir)
    version = (read_version(run_dir) | 1) + 1
    switched = False
    link = run_dir / f".current-{uuid.uuid4().hex}"
    try:
        write(stage)
        (stage / "version.json").write_text(json.dumps({"version": version, **counts}))
        link.symlink_to(stage.relative_to(run_dir), target_is_directory=True)
        os.replace(link, run_dir / "current")
        switched = True
        for name in ("ranked.csv", "bursts.csv", "search_index.npy", "search_index.json",
                     "version.json", "config.json"):
            alias = run_dir / f".{name}-{uuid.uuid4().hex}"
            try:
                alias.symlink_to(Path("current") / name)
                os.replace(alias, run_dir / name)
            finally:
                alias.unlink(missing_ok=True)
    finally:
        link.unlink(missing_ok=True)
        if not switched:
            shutil.rmtree(stage)
    # Keep the previous generation for readers that already resolved its path.
    # Older readers can retry against current if they race cleanup.
    for old in generations.iterdir():
        if old.is_dir() and old not in (stage, previous):
            shutil.rmtree(old, ignore_errors=True)
    return version


def list_images(root: Path) -> List[Path]:
    """Images directly inside ``root`` (non-recursive). Empty list if absent."""
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file())


def scan_images(root: Path, *, candidates=None) -> List[Path]:
    """Images anywhere under ``root`` (recursive).

    When a shoot contains both ``X.CR3`` and ``X.JPG``, the JPEG wins and the RAW
    is dropped. Content dedup would *probably* catch the pair — the RAW's
    embedded preview and the camera JPEG are near-identical renderings — but
    "probably" would mean occasionally showing every frame twice, and a filename
    rule is free and certain.
    """
    from .imaging import is_raw

    entries = root.rglob("*") if candidates is None else candidates
    found = sorted(p for p in entries if p.suffix.lower() in IMAGE_EXTS and p.is_file())
    non_raw_stems = {(p.parent, p.stem) for p in found if not is_raw(p)}
    return [p for p in found if not (is_raw(p) and (p.parent, p.stem) in non_raw_stems)]


def attach_burst_columns(
    df: pd.DataFrame,
    paths: Sequence[Path],
    cfg: Config,
    *,
    embeddings: Optional[Dict[Path, np.ndarray]] = None,
) -> pd.DataFrame:
    """Add ``burst_id``, ``within_burst_rank``, ``is_representative``, ``burst_rank``.

    Clustering uses EXIF plus (optionally) full-frame embedding similarity; when
    ``embeddings`` is supplied the short/long/similarity three-rule applies (see
    ``bursts.cluster_bursts``).
    """
    burst_map = assign_burst_ids(
        [Path(p) for p in paths],
        embeddings=embeddings,
        short_gap_seconds=cfg.bursts.short_gap_seconds,
        long_gap_seconds=cfg.bursts.long_gap_seconds,
        similarity_threshold=cfg.bursts.similarity_threshold,
        gap_seconds_fallback=cfg.bursts.gap_seconds,
    )
    df = df.copy()
    df["burst_id"] = [burst_map[Path(p)] for p in df["path"]]

    score_col = cfg.bursts.representative_score_col
    if score_col not in df.columns:
        score_col = "score_s1"

    df["within_burst_rank"] = (
        df.groupby("burst_id")[score_col].rank(ascending=False, method="first").astype(int)
    )

    # Representative per burst: highest score, preferring non-hard-rejected frames.
    df["is_representative"] = False
    if "tech_hard_reject" in df.columns:
        priority = (~df["tech_hard_reject"].astype(bool)).astype(int)  # 1 = preferred
    else:
        priority = pd.Series([1] * len(df), index=df.index)
    df["_rep_key"] = list(zip(priority.values, df[score_col].values))
    idx_rep = df.groupby("burst_id")["_rep_key"].idxmax()
    df.loc[idx_rep, "is_representative"] = True
    df.drop(columns=["_rep_key"], inplace=True)

    # Burst rank = rank of the burst's representative.
    rep_scores = df[df["is_representative"]][["burst_id", score_col]].rename(
        columns={score_col: "_rep_score"}
    )
    rep_scores["burst_rank"] = (
        rep_scores["_rep_score"].rank(ascending=False, method="min").astype(int)
    )
    df = df.merge(rep_scores[["burst_id", "burst_rank"]], on="burst_id", how="left")
    return df


def write_search_index(
    run_dir: Path,
    df: pd.DataFrame,
    embeddings_map: Dict[Path, np.ndarray],
    cfg: Config,
) -> Optional[Path]:
    """Persist the full-frame embedding matrix for text→image search.

    Row order matches ``ranked.csv`` exactly, so the server aligns rows by index.
    The matrix holds the context encoder's ``emb_full`` (SigLIP2), which shares a
    joint space with the text tower — see ``encoders.build_text_encoder``. Only
    written when the context encoder *has* a text tower; otherwise search is
    unavailable and the file is skipped.
    """
    from .encoders import context_encoder_has_text_tower

    if not context_encoder_has_text_tower(cfg.features.context_encoder):
        return None
    try:
        mat = np.stack([embeddings_map[Path(p)] for p in df["path"]]).astype(np.float32)
    except KeyError:
        return None
    def _save_npy(tmp: Path) -> None:
        # Pass a file object: np.save() appends ".npy" to a *path* that lacks it,
        # which would silently write alongside the temp file instead of to it.
        with open(tmp, "wb") as fh:
            np.save(fh, mat)

    atomic_write(run_dir / "search_index.npy", _save_npy)
    (run_dir / "search_index.json").write_text(json.dumps({
        "context_encoder": cfg.features.context_encoder,
        "kind": "emb_full",
        "dim": int(mat.shape[1]),
        "n": int(mat.shape[0]),
    }))
    return run_dir / "search_index.npy"


def write_bursts_csv(df: pd.DataFrame, run_dir: Path, cfg: Config) -> Path:
    """One row per burst (its representative + size), ranked."""
    reps = df[df["is_representative"]].copy()
    sizes = df.groupby("burst_id").size().rename("burst_size").reset_index()
    bursts = reps.merge(sizes, on="burst_id").sort_values("burst_rank").reset_index(drop=True)
    keep = [
        "burst_rank", "burst_id", "burst_size", "filename", "path",
        "shot_type", "subject_class", "is_hero",
        "score_s1", "score_s12_after_hard", "score_s3_final",
    ]
    out = bursts[[c for c in keep if c in bursts.columns]]
    path = run_dir / "bursts.csv"
    atomic_write(path, lambda t: out.to_csv(t, index=False))
    return path
