"""Scene tagging: prominent-person detection → Subject (people/stage) + Hero.

Two viewer filters beyond shot scale:
  • Subject — is there a prominent foreground person? "people" vs "stage"
    (empty stage / distant crowd / slide, i.e. no main focus).
  • Hero — a single dominant person against a clean/dark background (no other
    people, no background graphics/screens). "The speaker alone, nothing behind
    them."

Signals (per frame), computed once and cached raw so thresholds stay tunable
without re-processing:
  • YOLOv8-pose person boxes → largest-person area, #prominent persons, 2nd area
  • background brightness → fraction of non-person pixels above a luma floor
    (a clean dark stage ≈ 0; a lit screen / rack / chart lights it up)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from .cache_io import FeatureCache

SCENE_CACHE_ID = "yolov8s"  # cache namespace for the raw scene vector


@dataclass
class SceneConfig:
    enabled: bool = True
    yolo_model: str = "yolov8s-pose.pt"
    prominent_area: float = 0.03        # a person box this big counts as "prominent"
    person_min_area: float = 0.04       # Subject: people vs stage
    hero_min_area: float = 0.06         # Hero: how much of the frame he fills
    hero_max_second_area: float = 0.02  # Hero: no other prominent person
    hero_bg_bright_max: float = 0.05    # Hero: max fraction of bright background
    bg_luma: int = 55                   # luma (0-255) above which a bg pixel is "lit"
    bg_small: int = 200                 # downscaled long side for the bg measure


def _bg_bright_frac(gray_small: np.ndarray, box_norm, cfg: SceneConfig) -> float:
    """Fraction of non-person pixels brighter than the luma floor."""
    h, w = gray_small.shape
    mask = np.ones((h, w), dtype=bool)
    if box_norm is not None and box_norm[2] > box_norm[0]:
        x1, y1, x2, y2 = box_norm
        mask[max(0, int(y1 * h)):int(y2 * h), max(0, int(x1 * w)):int(x2 * w)] = False
    bg = gray_small[mask]
    return float((bg > cfg.bg_luma).mean()) if bg.size else 1.0


class SceneAnalyzer:
    """Caches a raw scene vector [area, n_prominent, second_area, bg_bright]."""

    def __init__(self, cfg: SceneConfig, cache: FeatureCache, device: Optional[str] = None) -> None:
        self.cfg = cfg
        self.cache = cache
        self.device = device
        self._model = None

    def _yolo(self):
        if self._model is None:
            from ultralytics import YOLO

            from .config import asset_dir
            # Prefer a pre-fetched local copy; otherwise pass the bare name and
            # let ultralytics download it (see scripts/fetch_assets.py).
            local = asset_dir() / self.cfg.yolo_model
            self._model = YOLO(str(local) if local.is_file() else self.cfg.yolo_model)
        return self._model

    def raw(self, sha1: str, pil) -> np.ndarray:
        """``pil`` may be an image or a zero-arg callable returning one.

        Pass a callable when re-running over a mostly-cached folder: opening the
        file is only worth paying for on a miss, and for RAW a needless open
        means pulling a 30 MB container off disk (or off a network share).
        """
        if self.cache.has(sha1, "scene", SCENE_CACHE_ID):
            return self.cache.load(sha1, "scene", SCENE_CACHE_ID)["z"]
        img = (pil() if callable(pil) else pil).convert("RGB")
        W, H = img.size
        res = self._yolo().predict(img, verbose=False, device=self.device)[0]
        box_norm = None
        area = nprom = second = 0.0
        if res.boxes is not None and len(res.boxes) > 0:
            b = res.boxes.xyxy.cpu().numpy()
            a = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])) / (W * H)
            idx = np.argsort(-a)
            area = float(a[idx[0]])
            nprom = float((a >= self.cfg.prominent_area).sum())
            second = float(a[idx[1]]) if len(a) > 1 else 0.0
            big = b[idx[0]]
            box_norm = (big[0] / W, big[1] / H, big[2] / W, big[3] / H)
        sc = self.cfg.bg_small / max(W, H)
        gray = np.asarray(img.convert("L").resize((max(1, int(W * sc)), max(1, int(H * sc)))))
        bg = _bg_bright_frac(gray, box_norm, self.cfg)
        vec = np.array([area, nprom, second, bg], dtype=np.float32)
        self.cache.save(sha1, "scene", {"z": vec}, SCENE_CACHE_ID)
        return vec

    def close(self) -> None:
        self._model = None


def classify_scene(vec: np.ndarray, cfg: SceneConfig) -> Tuple[str, bool]:
    """(subject_class, is_hero) from the raw [area, nprom, second, bg] vector."""
    area, nprom, second, bg = float(vec[0]), float(vec[1]), float(vec[2]), float(vec[3])
    subject = "people" if area >= cfg.person_min_area else "stage"
    hero = (
        area >= cfg.hero_min_area
        and nprom <= 1
        and second < cfg.hero_max_second_area
        and bg <= cfg.hero_bg_bright_max
    )
    return subject, bool(hero)
