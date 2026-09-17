"""Stage-2 technical quality: soft penalty + minimal hard-reject set.

The point of this stage is *not* to make subjective judgements — the learned ranker does that. It only handles things that are unambiguously bad:
  - Eyes definitely closed in a portrait (large face area, low EAR)
  - Severely blurry face (Laplacian variance under floor)
  - Face is mostly blown highlights or crushed shadows

Everything else becomes a continuous penalty added to the stage-1 score so that
two otherwise-equal frames are ranked by technical quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import TechnicalConfig
from .features import FeatureBundle


@dataclass
class TechReport:
    sharpness_norm: float        # 0..1, robust normalized log-sharpness
    exposure_penalty: float      # 0..1
    eye_open_penalty: float      # 0..1
    composition_bonus: float     # 0..1 (face size + centeredness)
    hard_reject: bool
    hard_reject_reason: str
    score: float                 # combined stage-2 contribution; positive = better


def _robust_unit(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return float(min(1.0, max(0.0, (x - lo) / (hi - lo))))


def technical_scores(
    bundles: Sequence[FeatureBundle],
    cfg: TechnicalConfig,
) -> Tuple[List[TechReport], np.ndarray]:
    """Compute a per-image technical score and the hard-reject mask."""
    sharps = np.array([b.quality.get("sharpness", 0.0) for b in bundles], dtype=np.float32)
    log_sharps = np.log1p(sharps)
    if log_sharps.size:
        s_lo = float(np.quantile(log_sharps, 0.05))
        s_hi = float(np.quantile(log_sharps, 0.95))
    else:
        s_lo, s_hi = 0.0, 1.0

    reports: List[TechReport] = []
    scores = np.zeros(len(bundles), dtype=np.float32)
    for i, b in enumerate(bundles):
        q = b.quality
        face_frac = float(q.get("face_frac", 0.0))
        ear = float(q.get("ear", 1.0))
        sharp = float(q.get("sharpness", 0.0))
        clipped = float(q.get("exposure_clipped_frac", 0.0))
        center = float(q.get("face_centeredness", 0.5))

        sharpness_norm = _robust_unit(float(np.log1p(sharp)), s_lo, s_hi)
        exposure_penalty = _robust_unit(clipped, 0.0, 0.30)
        # Eyes: only meaningfully penalize if the face is reasonably large.
        eye_open_penalty = (1.0 - _robust_unit(ear, 0.10, 0.22)) * _robust_unit(face_frac, 0.005, 0.05)
        composition_bonus = 0.5 * _robust_unit(face_frac, 0.0, 0.20) + 0.5 * center

        face_detected = bool(q.get("face_detected", 0.0))
        hard = False
        reason = ""
        if not getattr(cfg, "hard_reject_enabled", True):
            pass  # hard-reject disabled — MLP handles quality via features
        elif face_frac > 0.05 and ear < cfg.hard_reject_eyes_closed_ear:
            hard, reason = True, "eyes_closed"
        elif (
            face_detected
            and face_frac > cfg.hard_reject_min_face_frac
            and sharp < cfg.hard_reject_severe_blur_lapvar
        ):
            # Only flag blur when there *is* a face to measure. Otherwise sharpness was
            # computed on a center-crop of the full frame (often flat backdrop) and the
            # number means nothing.
            hard, reason = True, "severe_blur"
        elif face_detected and clipped > cfg.hard_reject_clipped_face_frac:
            hard, reason = True, "exposure_clipped"

        # Combined contribution: positive = better. Range roughly [-1, +1].
        s = (
            0.4 * sharpness_norm
            - 0.4 * exposure_penalty
            - 0.5 * eye_open_penalty
            + 0.2 * composition_bonus
        )
        scores[i] = s
        reports.append(
            TechReport(
                sharpness_norm=sharpness_norm,
                exposure_penalty=exposure_penalty,
                eye_open_penalty=eye_open_penalty,
                composition_bonus=composition_bonus,
                hard_reject=hard,
                hard_reject_reason=reason,
                score=float(s),
            )
        )
    return reports, scores


def combine_stage_scores(
    s1: np.ndarray, s2: np.ndarray, alpha: float
) -> np.ndarray:
    """Stage-1 + alpha * stage-2 (z-scored on the live distribution)."""
    s1z = _zscore(s1)
    s2z = _zscore(s2)
    return s1z + alpha * s2z


def _zscore(x: np.ndarray) -> np.ndarray:
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < 1e-8:
        return np.zeros_like(x)
    return (x - mu) / sd


def reports_to_dicts(reports: Sequence[TechReport]) -> List[Dict[str, object]]:
    return [
        {
            "tech_sharpness_norm": r.sharpness_norm,
            "tech_exposure_penalty": r.exposure_penalty,
            "tech_eye_open_penalty": r.eye_open_penalty,
            "tech_composition_bonus": r.composition_bonus,
            "tech_score": r.score,
            "tech_hard_reject": r.hard_reject,
            "tech_hard_reject_reason": r.hard_reject_reason,
        }
        for r in reports
    ]
