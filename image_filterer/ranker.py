"""Pairwise RankNet probe for hero-shot scoring.

Why pairwise ranking instead of binary logistic regression (v2):

* With G Good and B Bad images you get G*B pairs of training signal — for v2's
  79 Good × 90 Bad that's 7,110 pairs from 169 examples. The probe learns from a
  much denser signal in the same data regime.
* The loss matches the metric we actually care about (rank order, not calibrated
  probability), and is robust to class imbalance.
* Generalizes much better than logistic regression in the small-label, high-D
  feature regime we're in.

The ranker is a small MLP on top of the concatenated feature vector. We score one
image at a time at inference (model(x_i)), but during training we sample pairs
and minimize -log sigmoid(s_pos - s_neg).

Includes:
- stratified train / holdout split
- inner k-fold cross-validation on the train split
- ROC AUC (Mann-Whitney) for honest evaluation
- ablation hooks (toggle feature groups via RankerConfig)
- model save/load to .pt for use in stage-1 scoring later
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RankerConfig
from .features import FeatureBundle


# ----------------------------------------------------------------------- model


class RankNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# -------------------------------------------------------------- pair sampling


def _sample_good_bad_pairs(
    pos_idx: np.ndarray,
    neg_idx: np.ndarray,
    n_pairs: int,
    rng: np.random.Generator,
    *,
    burst_ids: Optional[np.ndarray],
    within_burst_fraction: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """The original Good-vs-Bad sampler, optionally burst-aware."""
    if n_pairs <= 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    if burst_ids is None or within_burst_fraction <= 0.0:
        p = rng.integers(0, len(pos_idx), size=n_pairs)
        n = rng.integers(0, len(neg_idx), size=n_pairs)
        return pos_idx[p], neg_idx[n]
    from collections import defaultdict
    pos_by_burst: Dict[int, List[int]] = defaultdict(list)
    neg_by_burst: Dict[int, List[int]] = defaultdict(list)
    for i in pos_idx:
        pos_by_burst[int(burst_ids[i])].append(int(i))
    for i in neg_idx:
        neg_by_burst[int(burst_ids[i])].append(int(i))
    mixed = [b for b in pos_by_burst if b in neg_by_burst]
    n_within = int(round(n_pairs * within_burst_fraction)) if mixed else 0
    n_cross = n_pairs - n_within
    p_cross = pos_idx[rng.integers(0, len(pos_idx), size=n_cross)] if n_cross else np.zeros(0, dtype=np.int64)
    n_cross_a = neg_idx[rng.integers(0, len(neg_idx), size=n_cross)] if n_cross else np.zeros(0, dtype=np.int64)
    if n_within:
        burst_arr = np.asarray(mixed, dtype=np.int64)
        chosen = burst_arr[rng.integers(0, len(burst_arr), size=n_within)]
        p_w = np.array([pos_by_burst[b][int(rng.integers(0, len(pos_by_burst[b])))] for b in chosen], dtype=np.int64)
        n_w = np.array([neg_by_burst[b][int(rng.integers(0, len(neg_by_burst[b])))] for b in chosen], dtype=np.int64)
        return np.concatenate([p_cross, p_w]), np.concatenate([n_cross_a, n_w])
    return p_cross, n_cross_a


def sample_pair_indices(
    pos_idx: np.ndarray,
    neg_idx: np.ndarray,
    n_pairs: int,
    rng: np.random.Generator,
    *,
    burst_ids: Optional[np.ndarray] = None,
    within_burst_fraction: float = 0.0,
    top_idx: Optional[np.ndarray] = None,
    must_find_pair_fraction: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample n_pairs (positive, negative) index pairs for the ranking loss.

    Three-way supervision:
      - ``(Good, Bad)`` is the baseline pair type, with optional within-burst
        oversampling as before.
      - ``(Top, Bad)`` pairs explicitly teach the ranker that must-find content
        is above ordinary Bad. Sampled at ``must_find_pair_fraction`` of the
        total budget. Only fires when ``top_idx`` is non-empty.

    Top here means "Good frame that is byte- or pixel-equivalent to a must-find
    image from the held-aside Top-5to10 set" (detected in cmd_train).
    """
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("Need at least one positive and one negative for pairwise training.")

    n_top = 0
    if top_idx is not None and len(top_idx) > 0 and must_find_pair_fraction > 0.0:
        n_top = int(round(n_pairs * must_find_pair_fraction))
    n_gb = n_pairs - n_top

    p_gb, n_gb_arr = _sample_good_bad_pairs(
        pos_idx, neg_idx, n_gb, rng,
        burst_ids=burst_ids, within_burst_fraction=within_burst_fraction,
    )

    if n_top:
        p_t = top_idx[rng.integers(0, len(top_idx), size=n_top)]
        n_t = neg_idx[rng.integers(0, len(neg_idx), size=n_top)]
        return np.concatenate([p_gb, p_t]), np.concatenate([n_gb_arr, n_t])
    return p_gb, n_gb_arr


# ---------------------------------------------------------------------- splits


def stratified_holdout(
    y: np.ndarray, fraction: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    if fraction <= 0 or fraction >= 1:
        return np.arange(len(y)), np.array([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    train, hold = [], []
    for c in (0, 1):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_hold = max(1, int(round(len(idx) * fraction))) if len(idx) >= 2 else 0
        n_hold = min(n_hold, max(0, len(idx) - 1))
        hold.append(idx[:n_hold])
        train.append(idx[n_hold:])
    return np.concatenate(train), np.concatenate(hold)


def stratified_kfold(y: np.ndarray, k: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for fi in range(k):
        val, train = [], []
        for c in (0, 1):
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            parts = np.array_split(idx, k)
            val.append(parts[fi])
            train.append(np.concatenate([parts[j] for j in range(k) if j != fi]))
        folds.append((np.concatenate(train), np.concatenate(val)))
    return folds


# ------------------------------------------------------------------ AUC helper


def auc_mannwhitney(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg])))
    r_pos = ranks[: len(pos)]
    n1, n2 = len(pos), len(neg)
    u = r_pos.sum() - n1 * (n1 - 1) / 2
    return float(u / (n1 * n2))


# --------------------------------------------------------------- feature stack


def stack_features(bundles: Sequence[FeatureBundle], cfg: RankerConfig) -> np.ndarray:
    return np.stack(
        [
            b.concat(
                use_full=cfg.use_full,
                use_face=cfg.use_face,
                use_body=cfg.use_body,
                use_va=cfg.use_va,
                use_quality=cfg.use_quality,
            )
            for b in bundles
        ],
        axis=0,
    )


# ----------------------------------------------------------------------- train


def _train_one(
    X: np.ndarray,
    y: np.ndarray,
    cfg: RankerConfig,
    *,
    device: torch.device,
    val_X: Optional[np.ndarray] = None,
    val_y: Optional[np.ndarray] = None,
    burst_ids: Optional[np.ndarray] = None,
    top_mask: Optional[np.ndarray] = None,
) -> Tuple[RankNet, Dict[str, List[float]]]:
    """Train a single RankNet on (X, y). Optionally track val AUC each epoch."""
    in_dim = X.shape[1]
    model = RankNet(in_dim, cfg.hidden, cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    top_idx = np.where(top_mask)[0] if top_mask is not None else None
    rng = np.random.default_rng(cfg.seed)
    Xt = torch.from_numpy(X.astype(np.float32)).to(device)
    if val_X is not None and val_y is not None:
        vXt = torch.from_numpy(val_X.astype(np.float32)).to(device)
    history: Dict[str, List[float]] = {"loss": [], "val_auc": []}

    for epoch in range(cfg.epochs):
        model.train()
        p_idx, n_idx = sample_pair_indices(
            pos_idx, neg_idx, cfg.pairs_per_epoch, rng,
            burst_ids=burst_ids,
            within_burst_fraction=cfg.within_burst_pair_fraction,
            top_idx=top_idx,
            must_find_pair_fraction=cfg.must_find_pair_fraction,
        )
        s_pos = model(Xt[p_idx])
        s_neg = model(Xt[n_idx])
        diff = s_pos - s_neg
        if cfg.margin > 0:
            loss = F.relu(cfg.margin - diff).mean()
        else:
            loss = F.softplus(-diff).mean()  # equivalent to -log sigmoid(diff)
        opt.zero_grad()
        loss.backward()
        opt.step()
        history["loss"].append(float(loss.item()))
        if val_X is not None and val_y is not None:
            model.eval()
            with torch.no_grad():
                s_val = model(vXt).cpu().numpy()
            history["val_auc"].append(auc_mannwhitney(val_y, s_val))
    return model, history


def predict(model: RankNet, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X.astype(np.float32)).to(device)).cpu().numpy()


# --------------------------------------------------------------------- driver


def cross_validate_and_fit(
    X: np.ndarray,
    y: np.ndarray,
    cfg: RankerConfig,
    *,
    device: torch.device,
    burst_ids: Optional[np.ndarray] = None,
    top_mask: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """
    1. Stratified holdout split.
    2. Inner k-fold on train split for honest CV AUC.
    3. Holdout AUC from a model trained only on the train split.
    4. Final model trained on **all** labeled data (returned).
    """
    tr_idx, ho_idx = stratified_holdout(y, cfg.holdout_fraction, cfg.seed)
    inner_aucs: List[float] = []
    if len(ho_idx) > 0 and cfg.cv_folds >= 2:
        min_class = int(min((y[tr_idx] == 0).sum(), (y[tr_idx] == 1).sum()))
        k = min(cfg.cv_folds, max(2, min_class))
        folds = stratified_kfold(y[tr_idx], k=k, seed=cfg.seed + 1)
        for fi, (tr_l, va_l) in enumerate(folds):
            tr_g = tr_idx[tr_l]
            va_g = tr_idx[va_l]
            bids_fold = burst_ids[tr_g] if burst_ids is not None else None
            tmask_fold = top_mask[tr_g] if top_mask is not None else None
            m, _hist = _train_one(X[tr_g], y[tr_g], cfg, device=device,
                                  burst_ids=bids_fold, top_mask=tmask_fold)
            s = predict(m, X[va_g], device)
            inner_aucs.append(auc_mannwhitney(y[va_g], s))
    mean_cv = float(np.nanmean(inner_aucs)) if inner_aucs else float("nan")

    auc_holdout = float("nan")
    auc_in_train = float("nan")
    if len(ho_idx) > 0:
        bids_tr = burst_ids[tr_idx] if burst_ids is not None else None
        tmask_tr = top_mask[tr_idx] if top_mask is not None else None
        m_ho, _hist = _train_one(X[tr_idx], y[tr_idx], cfg, device=device,
                                 burst_ids=bids_tr, top_mask=tmask_tr)
        s_h = predict(m_ho, X[ho_idx], device)
        s_t = predict(m_ho, X[tr_idx], device)
        auc_holdout = auc_mannwhitney(y[ho_idx], s_h)
        auc_in_train = auc_mannwhitney(y[tr_idx], s_t)

    final_model, hist = _train_one(X, y, cfg, device=device,
                                   burst_ids=burst_ids, top_mask=top_mask)
    s_all = predict(final_model, X, device)
    auc_all_in_sample = auc_mannwhitney(y, s_all)

    return {
        "model": final_model,
        "tr_idx": tr_idx,
        "ho_idx": ho_idx,
        "inner_auc_per_fold": inner_aucs,
        "mean_cv_auc": mean_cv,
        "auc_holdout": auc_holdout,
        "auc_in_train": auc_in_train,
        "auc_all_in_sample": auc_all_in_sample,
        "history": hist,
    }


def save_model(path: Path, model: RankNet, cfg: RankerConfig, in_dim: int, meta: Dict[str, object]) -> None:
    blob = {
        "state_dict": model.state_dict(),
        "config": asdict(cfg),
        "in_dim": int(in_dim),
        "meta": meta,
    }
    torch.save(blob, path)


def load_model(path: Path, device: torch.device) -> Tuple[RankNet, Dict, Dict]:
    # The checkpoint stores non-tensor fields (config, meta), so it needs the
    # full unpickler; torch>=2.6 defaults weights_only=True and would refuse it.
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = blob["config"]
    in_dim = int(blob["in_dim"])
    cfg = RankerConfig(**cfg_dict)
    model = RankNet(in_dim, cfg.hidden, cfg.dropout).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, cfg_dict, blob.get("meta", {})


def serialize_eval_summary(result: Dict[str, object]) -> str:
    return json.dumps(
        {
            "mean_cv_auc": result["mean_cv_auc"],
            "auc_holdout": result["auc_holdout"],
            "auc_in_train": result["auc_in_train"],
            "auc_all_in_sample": result["auc_all_in_sample"],
            "inner_auc_per_fold": result["inner_auc_per_fold"],
            "n_train": int(len(result["tr_idx"])),
            "n_holdout": int(len(result["ho_idx"])),
        },
        indent=2,
    )
