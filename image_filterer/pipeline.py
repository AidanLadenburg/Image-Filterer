"""Shared pipeline steps used by both training and ingestion.

These sit between the per-frame scorers (``ranker``, ``technical``) and the run
outputs: burst assignment and the search index. Both training and ingest call
them, so they live here rather than in either.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .bursts import assign_burst_ids
from .config import IMAGE_EXTS, Config


def list_images(root: Path) -> List[Path]:
    """Images directly inside ``root`` (non-recursive). Empty list if absent."""
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.suffix in IMAGE_EXTS and p.is_file())


def scan_images(root: Path) -> List[Path]:
    """Images anywhere under ``root`` (recursive)."""
    return sorted(p for p in root.rglob("*") if p.suffix in IMAGE_EXTS and p.is_file())


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
    np.save(run_dir / "search_index.npy", mat)
    (run_dir / "search_index.json").write_text(json.dumps({
        "context_encoder": cfg.features.context_encoder,
        "kind": "emb_full",
        "dim": int(mat.shape[1]),
        "n": int(mat.shape[0]),
    }))
    return run_dir / "search_index.npy"


def write_bursts_csv(df: pd.DataFrame, run_dir: Path, cfg: Config) -> Path:
    """One row per burst (its representative + size), ranked."""
    score_col = (
        cfg.bursts.representative_score_col
        if cfg.bursts.representative_score_col in df.columns
        else "score_s1"
    )
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
    out.to_csv(path, index=False)
    return path
