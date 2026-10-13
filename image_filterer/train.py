"""Train the production ranker.

    python -m image_filterer.train                          # default dataset
    python -m image_filterer.train --train-root A --train-root B   # several labeled events
    python -m image_filterer.train --no-eval                # skip the held-out tests

Each training root holds one labeled event::

    Good/  Bad/  Top-5to10/  (Questionable/ is ignored)

What it does, and why (details in docs/ARCHITECTURE.md):

* Features come from the same code as ingest (features.py + scene.py), so the
  model always sees exactly what the app computes.
* **Photographer-days with only one kind of label are left out.** Some
  photographers hand over only their picks; their photos are 100% "Good", so a
  model learns their camera, position and framing style — mostly wide,
  faceless stage shots — as "good". On a new shoot that pushed empty stages
  and slides to the top. (Likewise a day where everything is "Bad".)
* The model is ranker.RankerV2: compressed embeddings + face measurements +
  the speaker's expression, pairwise loss with half the pairs drawn inside a
  burst, three seeds averaged.
* Held-out tests, reported in model/training_summary.json — never on photos the
  model trained on, and never splitting a burst between train and test:
    - leave one photographer out: train on everyone else, rank only that
      photographer's photos (their camera and position can't help);
    - five contiguous slices of each event's timeline.
  Metrics are per moment (= burst; a moment counts as a keeper if any frame is
  Good): average precision, the share of keepers in the top 10%, and how often
  the best frame of a mixed burst is a Good one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .bursts import assign_burst_ids, read_frame_meta
from .cache_io import Sha1Memo
from .common import load_or_extract
from .config import MUST_FIND_FOLDERS, Config, default_config
from .ingest import analyze_scenes
from .pipeline import list_images
from .ranker import feature_blocks, fit_v2, save_v2


def gather_training(roots: Sequence[Path], cache_dir: Path) -> pd.DataFrame:
    """One row per unique photo (content-deduped; a positive label wins)."""
    rows = []
    memo = Sha1Memo(cache_dir / "sha1_memo.db")
    for ev, root in enumerate(roots):
        items = [(p, 1, False) for p in list_images(root / "Good")]
        items += [(p, 0, False) for p in list_images(root / "Bad")]
        for folder in MUST_FIND_FOLDERS:
            items += [(p, 1, True) for p in list_images(root / folder)]
        sha = memo.sha1_many([p for p, _, _ in items])
        for (p, y, top), s in zip(items, sha):
            rows.append({"path": str(p), "y": y, "must_find": top, "sha1": s, "event": ev})
    df = (pd.DataFrame(rows).sort_values(["y", "must_find"], ascending=False)
          .groupby("sha1", as_index=False).first())
    metas = [read_frame_meta(Path(p)) for p in df.path]
    df["camera_id"] = [m.camera_id if m else "" for m in metas]
    df["t"] = [m.captured_at if m else pd.NaT for m in metas]
    df["day"] = df.t.dt.date.astype(str)
    return df.sort_values("t").reset_index(drop=True)


def usable(df: pd.DataFrame) -> np.ndarray:
    """Photographer-days that contributed both keepers and rejects."""
    g = df.groupby(["event", "camera_id", "day"]).y.transform(lambda s: s.nunique() == 2)
    return g.to_numpy(bool)


# ------------------------------------------------------------------ held-out tests

def _average_precision(y, s) -> float:
    order = np.argsort(-np.asarray(s), kind="stable")
    hits = np.asarray(y)[order] == 1
    if not hits.any():
        return float("nan")
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(precision[hits].mean())


def _moment_metrics(y, burst, s, must):
    d = pd.DataFrame({"b": burst, "y": y, "s": s, "m": must})
    m = d.groupby("b").agg(y=("y", "max"), s=("s", "max"), m=("m", "max"))
    k = max(1, int(round(len(m) * 0.10)))
    top = m.sort_values("s", ascending=False)
    picks = [x.y.iloc[int(np.argmax(x.s.values))] == 1 for _, x in d.groupby("b") if x.y.nunique() == 2]
    return {"ap": _average_precision(m.y.to_numpy(), m.s.to_numpy()),
            "keepers_in_top10pct": float(top.y.head(k).mean()),
            "best_frame_picked": float(np.mean(picks)) if picks else float("nan"),
            "n_mixed_bursts": len(picks),
            "must_find_in_top10pct": int(top.m.head(k).sum()), "must_find_total": int(m.m.sum())}


def held_out_tests(blocks, df, burst, cfg: Config, device) -> Dict[str, object]:
    y = df.y.to_numpy(); cams = df.camera_id.to_numpy(); must = df.must_find.to_numpy()
    take = lambda idx: {k: v[idx] for k, v in blocks.items()}

    def fit_score(tr, te):
        m = fit_v2(take(tr), y[tr], burst[tr], cfg.ranker, device=device)
        return m.score(take(te), device)

    out: Dict[str, object] = {}
    per = {}
    for c, g in df.groupby("camera_id"):
        if g.y.nunique() == 2 and g.y.sum() >= 8:
            te = np.where(cams == c)[0]; tr = np.where(cams != c)[0]
            per[c] = _moment_metrics(y[te], burst[te], fit_score(tr, te), must[te])
    if per:
        P = pd.DataFrame(per).T
        out["leave_one_photographer_out"] = {
            "photographers": len(P), "ap": float(P.ap.mean()),
            "keepers_in_top10pct": float(P.keepers_in_top10pct.mean()),
            "best_frame_picked": float(np.average(P.best_frame_picked.fillna(0), weights=P.n_mixed_bursts))}
    s = np.zeros(len(df))
    for ev in df.event.unique():
        ev_idx = np.where(df.event.to_numpy() == ev)[0]
        start = pd.Series(df.t.values[ev_idx]).groupby(burst[ev_idx]).min().sort_values()
        fold_of = {b: min(4, int(i * 5 / len(start))) for i, b in enumerate(start.index)}
        fold = np.array([fold_of[b] for b in burst[ev_idx]])
        for f in range(5):
            te = ev_idx[fold == f]
            tr = np.setdiff1d(np.arange(len(df)), te)
            v = fit_score(tr, te); s[te] = (v - np.median(v)) / (v.std() + 1e-9)
    out["timeline_slices"] = _moment_metrics(y, burst, s, must)
    return out


# ------------------------------------------------------------------ training

def train_production_model(cfg: Config, roots: Sequence[Path], *, evaluate: bool = True, verbose: bool = True):
    cfg.ensure_dirs()
    log = print if verbose else (lambda *a, **k: None)
    df = gather_training(roots, cfg.cache_dir)
    keep = usable(df)
    log(f"[train] {len(df)} unique labeled photos from {len(roots)} event(s); "
        f"{int((~keep).sum())} left out (photographer-days with only one kind of label)")
    df = df[keep].reset_index(drop=True)
    log(f"[train] training on {len(df)}: {int(df.y.sum())} keepers, {int((df.y == 0).sum())} rejects")

    paths = [Path(p) for p in df.path]
    bundles = load_or_extract(cfg, paths, desc="train-features")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("[train] finding the speaker / measuring expressions…")
    scenes = analyze_scenes(bundles, cfg, device)
    blocks = feature_blocks(bundles, scenes)
    bm = assign_burst_ids(paths, embeddings={Path(str(b.path)): b.emb_full for b in bundles},
                          short_gap_seconds=cfg.bursts.short_gap_seconds,
                          long_gap_seconds=cfg.bursts.long_gap_seconds,
                          similarity_threshold=cfg.bursts.similarity_threshold,
                          gap_seconds_fallback=cfg.bursts.gap_seconds, lookback=cfg.bursts.lookback,
                          max_span_seconds=cfg.bursts.max_span_seconds, anchor_threshold=cfg.bursts.anchor_threshold)
    burst = np.array([bm[p] for p in paths])

    summary: Dict[str, object] = {
        "events": [str(r) for r in roots], "n_train": int(len(df)),
        "n_keepers": int(df.y.sum()), "n_rejects": int((df.y == 0).sum()),
        "left_out_one_label_photographer_days": int((~keep).sum()),
    }
    if evaluate:
        log("[train] held-out tests (a few minutes)…")
        summary["held_out"] = held_out_tests(blocks, df, burst, cfg, device)
        log(json.dumps(summary["held_out"], indent=2))

    model = fit_v2(blocks, df.y.to_numpy(), burst, cfg.ranker, device=device, meta={
        "context_encoder": cfg.features.context_encoder,
        "face_encoder": cfg.features.face_encoder,
        "events": summary["events"], "n_keepers": summary["n_keepers"], "n_rejects": summary["n_rejects"],
    })
    save_v2(cfg.model_path, model, cfg.ranker)
    (cfg.model_path.parent / "training_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"[train] saved {cfg.model_path}")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("image_filterer.train", description="Train the production ranker.")
    ap.add_argument("--train-root", action="append", default=None,
                    help="A labeled event folder (Good/ Bad/ Top-5to10/). Repeat for several events.")
    ap.add_argument("--no-eval", action="store_true", help="Skip the held-out tests.")
    args = ap.parse_args(argv)
    cfg = default_config()
    roots = [Path(r) for r in args.train_root] if args.train_root else [cfg.train_root]
    train_production_model(cfg, roots, evaluate=not args.no_eval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
