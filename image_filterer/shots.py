"""Shot-scale tagging: wide / medium / close.

Every frame is placed on a *continuous* wide↔close axis in the SigLIP2 joint
image-text space (the same space the semantic-search text tower and the cached
``emb_full`` image vectors share), then split into three classes by two
thresholds.

    closeness(image) = cos(emb_full, close_prototype) - cos(emb_full, wide_prototype)

Each prototype is the L2-normalized mean of a small prompt ensemble. Higher
closeness = tighter framing (face-dominant); lower = wider (subject small in a
large scene). See ``config.ShotConfig`` for why face-geometry was rejected and
how the thresholds were calibrated.

Nothing here re-processes pixels: it operates on the already-cached full-frame
embeddings, so shot tagging is essentially free during the ranking pass and can
be backfilled for any run from its search index / feature cache.
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
