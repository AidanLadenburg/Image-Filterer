"""Train the production ranker on ALL labeled data.

This ships one fixed model:

* Positives  = Good ∪ Top-5to10 (the must-find gold keepers, no longer held aside
  for evaluation now that we're in production).
* Negatives  = Bad.
* Content-deduped by sha1 (a positive label wins any collision).
* The saved model is trained on 100% of these frames. A holdout split is still
  carved out, but ONLY to report an honest AUC — the shipped weights come from
  ``cross_validate_and_fit``'s final all-data fit.

Run once (features are cached, so this is fast after the first extraction):

    python -m image_filterer.train                      # default dataset + output path
    python -m image_filterer.train --train-root PATH    # override labeled dataset
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from .bursts import assign_burst_ids
from .cache_io import file_sha1
from .config import MUST_FIND_FOLDERS
from .ranker import cross_validate_and_fit, save_model, serialize_eval_summary, stack_features
from .pipeline import list_images

from .common import load_or_extract
from .config import Config, default_config


def gather_training(cfg: Config) -> Tuple[List[Path], np.ndarray]:
    """Return (paths, y) with y=1 for Good/must-find, 0 for Bad, sha1-deduped."""
    items: List[Tuple[Path, int]] = []
    items += [(p, 1) for p in list_images(cfg.train_root / "Good")]
    items += [(p, 0) for p in list_images(cfg.train_root / "Bad")]
    for folder in MUST_FIND_FOLDERS:
        items += [(p, 1) for p in list_images(cfg.train_root / folder)]

    by_sha: dict[str, Tuple[Path, int]] = {}
    for p, lab in items:
        s = file_sha1(p)
        if s in by_sha:
            kept_p, kept_lab = by_sha[s]
            by_sha[s] = (kept_p, max(kept_lab, lab))  # any positive wins
        else:
            by_sha[s] = (p, lab)
    paths = [v[0] for v in by_sha.values()]
    y = np.array([v[1] for v in by_sha.values()], dtype=np.int64)
    return paths, y


def train_production_model(cfg: Config, *, holdout_fraction: float = 0.2, verbose: bool = True):
    cfg.ensure_dirs()
    paths, y = gather_training(cfg)
    if verbose:
        print(f"[train] {len(paths)} unique labeled frames "
              f"(pos={int((y == 1).sum())}, neg={int((y == 0).sum())})")

    bundles = load_or_extract(cfg, paths, desc="train-features")
    X = stack_features(bundles, cfg.ranker)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"[train] features {X.shape} on {device}")

    burst_ids = None
    if cfg.ranker.within_burst_pair_fraction > 0.0:
        emb_map = {Path(str(b.path)): b.emb_full for b in bundles}
        bm = assign_burst_ids(
            [Path(p) for p in paths],
            embeddings=emb_map,
            short_gap_seconds=cfg.bursts.short_gap_seconds,
            long_gap_seconds=cfg.bursts.long_gap_seconds,
            similarity_threshold=cfg.bursts.similarity_threshold,
            gap_seconds_fallback=cfg.bursts.gap_seconds,
        )
        burst_ids = np.array([bm[Path(p)] for p in paths], dtype=np.int64)

    cfg.ranker.holdout_fraction = holdout_fraction
    res = cross_validate_and_fit(X, y, cfg.ranker, device=device, burst_ids=burst_ids)

    meta = {
        "context_encoder": cfg.features.context_encoder,
        "face_encoder": cfg.features.face_encoder,
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "trained_on": "all labeled frames (Good + Top-5to10 vs Bad), no eval holdout",
        "within_burst_pair_fraction": cfg.ranker.within_burst_pair_fraction,
    }
    save_model(cfg.model_path, res["model"], cfg.ranker, X.shape[1], meta)
    (cfg.model_path.parent / "training_summary.json").write_text(serialize_eval_summary(res))
    if verbose:
        print(f"[train] saved {cfg.model_path}")
        print(f"[train] holdout AUC={res['auc_holdout']:.4f}  "
              f"mean_cv_auc={res['mean_cv_auc']:.4f}  (eval-only; model uses all data)")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("image_filterer.train", description="Train the production ranker on all data.")
    ap.add_argument("--train-root", type=str, default=None)
    ap.add_argument("--holdout-fraction", type=float, default=0.2, help="Eval-only; model still uses all data.")
    args = ap.parse_args(argv)
    cfg = default_config()
    if args.train_root:
        cfg.train_root = Path(args.train_root)
    train_production_model(cfg, holdout_fraction=args.holdout_fraction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
