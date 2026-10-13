"""Shot-scale tagging: wide / medium / close.

Primary: :func:`shot_from_pose` — where the frame cuts the speaker's body,
from the body keypoints in scene.py. See ``config.ShotConfig`` for the rules.

Fallback (runs without pose data): every frame placed on a continuous
wide↔close axis in the SigLIP2 joint image-text space,

    closeness(image) = cos(emb_full, close_prototype) - cos(emb_full, wide_prototype)

split by two thresholds. It works on cached embeddings only, but it's fooled
by darkness and content, which is why it's no longer the default.
"""

from __future__ import annotations

import hashlib
import json
from typing import List, Optional, Sequence, Tuple

from .cache_io import FeatureCache

import numpy as np

from .config import ShotConfig
from .encoders import TextEncoder, build_text_encoder, context_encoder_has_text_tower

SHOT_CLASSES: Tuple[str, str, str] = ("wide", "medium", "close")

# Prompt ensembles defining the axis endpoints. Kept deliberately generic (stage
# / speaker / person) so the axis reflects framing scale, not scene content.
WIDE_PROMPTS: Sequence[str] = (
    "a wide establishing shot of a stage and a large audience",
    "a wide-angle photo of a full auditorium",
    "a long shot of a person full length on a big stage",
    "a distant wide shot of a crowd of people",
)
CLOSE_PROMPTS: Sequence[str] = (
    "a close-up portrait of a person's face",
    "a tight headshot of one person",
    "a close-up photo of a face",
)


def build_shot_axis(text_encoder: TextEncoder) -> np.ndarray:
    """Return a (2, D) array [wide_prototype, close_prototype], L2-normalized.

    ``closeness`` = image·close - image·wide = image · (row1 - row0).
    """
    def _proto(prompts: Sequence[str]) -> np.ndarray:
        z = text_encoder.embed_texts(list(prompts))  # already L2-normalized rows
        mean = z.mean(axis=0)
        return (mean / max(float(np.linalg.norm(mean)), 1e-12)).astype(np.float32)

    return np.stack([_proto(WIDE_PROMPTS), _proto(CLOSE_PROMPTS)])


def closeness_scores(emb_full: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """emb_full: (N, D) L2-normalized image embeddings. axis: (2, D). → (N,)."""
    return (emb_full @ (axis[1] - axis[0])).astype(np.float32)


def classify_closeness(closeness: np.ndarray, cfg: ShotConfig) -> List[str]:
    labels: List[str] = []
    for c in closeness:
        if c < cfg.wide_max:
            labels.append("wide")
        elif c >= cfg.close_min:
            labels.append("close")
        else:
            labels.append("medium")
    return labels


def compute_shot_types(
    emb_full: np.ndarray,
    context_encoder_name: str,
    cfg: ShotConfig,
    *,
    text_encoder: Optional[TextEncoder] = None,
    cache: Optional[FeatureCache] = None,
) -> Tuple[List[str], np.ndarray]:
    """Tag a matrix of full-frame embeddings with (labels, closeness).

    Returns ([], empty) when the context encoder has no text tower (DINOv2 /
    C-RADIO) — shot tagging is unavailable there, same as semantic search.
    """
    if not context_encoder_has_text_tower(context_encoder_name):
        return [], np.zeros((0,), dtype=np.float32)
    key = hashlib.sha1(json.dumps([context_encoder_name, WIDE_PROMPTS, CLOSE_PROMPTS]).encode()).hexdigest()
    if cache is not None and cache.has(key, "shot_axis", context_encoder_name):
        axis = cache.load(key, "shot_axis", context_encoder_name)["z"]
    else:
        if text_encoder is None:
            text_encoder = build_text_encoder(context_encoder_name)
        axis = build_shot_axis(text_encoder)
        if cache is not None:
            cache.save(key, "shot_axis", {"z": axis}, context_encoder_name)
    closeness = closeness_scores(emb_full.astype(np.float32), axis)
    return classify_closeness(closeness, cfg), closeness


def shot_from_pose(scene, cfg: ShotConfig) -> Tuple[str, str]:
    """(label, reason) from where the frame cuts the speaker's body."""
    from .scene import KP_CONF
    b = scene.subject_box()
    if b is None:
        return "wide", "no prominent person"
    k = scene.kps[scene.subject]; c = k[:, 2]
    h = float(b[3] - b[1])
    if c[15:17].max() > KP_CONF:
        return "wide", "whole body in frame"
    if c[11:13].max() > KP_CONF:
        return ("medium", "cut above the feet") if h >= cfg.medium_min_h else ("wide", "small figure")
    if not (c[:5].max() > KP_CONF or c[5:7].max() > KP_CONF):
        return "wide", "partly seen"
    sh = [k[j, 1] for j in (5, 6) if c[j] > KP_CONF]
    eyes = [k[j, 1] for j in (1, 2) if c[j] > KP_CONF] or [k[j, 1] for j in (0,) if c[j] > KP_CONF]
    if sh and eyes and np.mean(sh) - np.mean(eyes) > 1e-3:
        head = float(np.mean(sh) - np.mean(eyes))               # eyes → shoulders, share of frame height
        below = (min(float(b[3]), 1.0) - float(np.mean(sh))) / head
        if below < cfg.chest_cut and head >= cfg.close_head and h >= cfg.close_span:
            return "close", "chest-up"
    return ("medium", "cut above the waist") if h >= cfg.medium_min_h else ("wide", "small figure")
