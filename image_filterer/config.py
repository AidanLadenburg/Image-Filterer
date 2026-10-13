"""Central configuration.

Every tunable lives here as a dataclass. :func:`default_config` returns the
**production** configuration — the one the shipped model was trained with and the
one the server runs. Where a default deviates from the obvious choice, the
comment says which experiment settled it; don't change those without re-running
the corresponding evaluation.

Filesystem layout is resolved from environment variables so the same code runs
from a checkout, a container, or an install:

===============================  ==============================================
``IMAGE_FILTERER_DATA_ROOT``     run storage + the SQLite registry
                                 (default ``~/.local/share/image-filterer``)
``IMAGE_FILTERER_CACHE_DIR``     content-addressed feature cache
                                 (default ``$IMAGE_FILTERER_DATA_ROOT/cache``)
``IMAGE_FILTERER_MODEL_PATH``    the production ranker checkpoint
                                 (default ``image_filterer/model/ranker.pt``)
``IMAGE_FILTERER_ASSET_DIR``     third-party weights: FaRL, YuNet, YOLO
                                 (default ``$IMAGE_FILTERER_DATA_ROOT/assets``)
``IMAGE_FILTERER_PASSWORD``      shared password gating the server
                                 (default unset — auth disabled)
``IMAGE_FILTERER_EXPORT_DIR``    server-side export folder, e.g. a LucidLink
                                 share (default unset — option hidden)
===============================  ==============================================

The feature cache is keyed by ``(sha1, kind, encoder_id)``, so it is safe to
share between machines and across versions — point ``IMAGE_FILTERER_CACHE_DIR``
at an existing cache and nothing is recomputed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

PACKAGE_DIR = Path(__file__).resolve().parent

def _data_root() -> Path:
    """Resolved per call, not at import, so the environment can be set late."""
    return Path(
        os.environ.get("IMAGE_FILTERER_DATA_ROOT", str(Path.home() / ".local" / "share" / "image-filterer"))
    )


def asset_dir() -> Path:
    """Where third-party weights (FaRL / YuNet / YOLO) are looked up."""
    return Path(os.environ.get("IMAGE_FILTERER_ASSET_DIR", str(_data_root() / "assets")))


# ------------------------------------------------------------------- features


@dataclass
class FeatureConfig:
    context_encoder: str = "siglip2_so400m_14"
    face_encoder: str = "farl_base"
    use_valence_arousal: bool = False
    face_expand: float = 1.55
    batch_size: int = 8
    cradio_input_side: Optional[int] = 1024
    skip_face_detect: bool = False

    # Face-detection backend. MediaPipe's FaceLandmarker is tuned for
    # selfie-distance frontal faces and misses small/distant subjects — on stage
    # shots it detects a face only ~2% of the time, leaving the face embedding,
    # EAR (eyes-closed) and expression signals dead. "yunet" runs OpenCV's YuNet
    # detector (fast, small-face-capable) to find + crop the face, then MediaPipe
    # on that crop for the fine eyelid/mouth landmarks (~97% detection).
    # Namespaced separately in the feature cache, so switching backends never
    # returns stale crops.
    face_detector: str = "yunet"             # "yunet" | "mediapipe"
    yunet_score_threshold: float = 0.6
    yunet_det_size: int = 1024               # long-side px the detector runs at

    # SigLIP body-crop embedding. The pose ablation showed the body channel
    # doesn't earn its cost, so production skips it — one fewer context-encoder
    # forward pass per image at ingest.
    extract_body: bool = False


# --------------------------------------------------------------------- ranker


@dataclass
class RankerConfig:
    hidden: int = 256
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 60
    pairs_per_epoch: int = 4096
    margin: float = 0.0
    seed: int = 42
    holdout_fraction: float = 0.2
    cv_folds: int = 5

    # Feature groups. use_body/use_va are off to match FeatureConfig above;
    # together they take the input vector from 2825 → 1671 dims.
    use_full: bool = True
    use_face: bool = True
    use_body: bool = False
    use_va: bool = False
    use_quality: bool = True

    # Fraction of each epoch's pairs sampled within the same burst (same camera +
    # near-identical timestamp). 0.0 = uniform random Good-vs-Bad pairs. Higher
    # values fight the cross-burst pair-dilution shortcut and force the ranker to
    # learn within-burst-distinguishing features (eye direction, mouth shape,
    # gesture phase) rather than scene-level cues.
    #
    # Sweep: 0.0 → 0.3 → 0.5 → 0.7 → 1.0 moved best-Good@1-in-burst 59% → 84%
    # with holdout AUC flat through 0.5, then degrading. 1.0 collapses global AUC
    # because most bursts aren't mixed.
    within_burst_pair_fraction: float = 0.5

    # Fraction of pairs comparing a must-find ("Top") to a Bad. Promotes must-find
    # content twins from "ordinary positive" to "extra-strong positive". Off: it
    # improves memorized recall but doesn't generalize past 0.1.
    must_find_pair_fraction: float = 0.0


# ------------------------------------------------------------------ technical


@dataclass
class TechnicalConfig:
    # Mixing weight for the soft technical penalty in the stage-1+2 score. Off:
    # the penalty hurt must-find recall because face-quality metrics are
    # unreliable on wide stage shots where the face is small.
    alpha: float = 0.0

    # Master switch for the stage-2 hard-reject. When False no frame is slammed
    # to -inf; the MLP (which sees EAR / exposure / sharpness as features) is
    # trusted to rank eyes-closed and blown-out frames down on its own.
    #
    # OFF in production: of 126 hard-rejected frames, 29 were keepers — the
    # exposure-clipped rule alone dropped ~23% of good frames, because a dark
    # stage background reads as "clipped". Technical metrics are still computed
    # and stored per frame; they're informational, not a gate.
    hard_reject_enabled: bool = False
    hard_reject_eyes_closed_ear: float = 0.12
    hard_reject_severe_blur_lapvar: float = 4.0
    hard_reject_clipped_face_frac: float = 0.40
    hard_reject_min_face_frac: float = 0.005


# --------------------------------------------------------------------- bursts


@dataclass
class BurstConfig:
    """How to cluster frames into bursts.

    Camera identity comes from EXIF only (no filename heuristics):
      - Canon → BodySerialNumber (per-body unique)
      - Sony  → Make/Model/LensModel (the ILCE-1 exposes no body serial; lens
                model is the next-best per-rig discriminator)
      - else  → Make/Model

    Within one camera, two consecutive frames join the same burst when::

        dt <= short_gap_seconds                                    (always merge)
        OR (dt <= long_gap_seconds AND cosine_sim >= similarity_threshold)
    """

    # legacy single-gap fallback (used when the embedding-aware path can't run)
    gap_seconds: float = 2.5

    short_gap_seconds: float = 0.6      # below this, always merge
    # Similar frames up to this far apart are still one moment: a photographer
    # holding a composition and re-shooting it after a pause. 8 s split ~1 in 4
    # consecutive moments that were near-identical (same pose, same slide) into
    # separate grid tiles; at 30 s with the guards below, reviewed merges on two
    # shoots were the same moment (GTC 203 → 177 moments, Rory 65 → 47).
    long_gap_seconds: float = 30.0
    similarity_threshold: float = 0.91  # cosine to any of the moment's last `lookback` frames
    lookback: int = 3                   # so one odd frame (a turn, a flash) doesn't split a moment
    max_span_seconds: float = 45.0      # a moment never lasts longer than this…
    anchor_threshold: float = 0.89      # …and every frame still resembles its first one (no slow drift)

    # Pixel-identical dedup: cosine >= this AND sha1 differs => same image,
    # different bytes (metadata-only edits, JPEG re-encodes, " 2.jpg" copies).
    # Deliberately extreme so genuine burst-mates are never merged.
    pixel_identical_cosine: float = 0.9995

    representative_score_col: str = "score_s12_after_hard"


# ---------------------------------------------------------------- shot scale


@dataclass
class ShotConfig:
    """Shot scale (wide / medium / close), from where the frame cuts the
    speaker's body (scene.py keypoints):

      wide   — the whole body is in frame (feet visible), the person is small,
               or there's no prominent person
      medium — the frame cuts the body somewhere above the feet
      close  — cut well above the waist AND the head is big in the frame AND
               the person fills most of its height (an upward-angled shot of a
               distant speaker, merely cropped at the waist, isn't a close-up)

    Distances are in "head-units": eyes → shoulders, vertically. It doesn't
    shrink when the speaker turns sideways, unlike shoulder width.

    The earlier SigLIP zero-shot axis (wide vs close text prompts) was fooled by
    darkness and content — near-black frames came out "close" — and is kept
    only as a fallback for runs without pose data.
    """

    enabled: bool = True
    chest_cut: float = 2.2      # close: frame cuts < this many head-units below the shoulders
    close_head: float = 0.20    # close: eyes → shoulders is at least this much of the frame height
    close_span: float = 0.6     # close: the person fills at least this much of the frame height
    medium_min_h: float = 0.3   # a cut-off figure smaller than this is still "wide"
    # Fallback (no pose data): SigLIP closeness thresholds, matched to the prompts in shots.py
    wide_max: float = -0.038
    close_min: float = 0.0


# ----------------------------------------------------------- text on screen


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class OcrConfig:
    """Reading the text in each photo (slides, screens, signs) for search and
    auto-export. PP-OCRv4 English on TensorRT; see ``ocr.py``.

    Every frame is read — burst-mates don't reliably show the same slide.
    On a GB10, decoding included: 703 photos in ~21 s with four workers (3: 26 s,
    5: 19 s). It runs in the background and pauses while an ingest is using the
    GPU, so it never delays photos appearing.
    """

    # IMAGE_FILTERER_OCR_WORKERS=0 turns it off.
    workers: int = field(default_factory=lambda: max(0, _env_int("IMAGE_FILTERER_OCR_WORKERS", 4)))
    # Longest side the detector sees. Slide text is small in a wide stage shot;
    # 2048 is what the benchmarks (and the engine profile) were tuned at.
    max_side: int = 2048
    min_score: float = 0.6       # drop lines the recognizer isn't sure of


# --------------------------------------------------------------- app config


@dataclass
class Config:
    features: FeatureConfig = field(default_factory=FeatureConfig)
    ranker: RankerConfig = field(default_factory=RankerConfig)
    technical: TechnicalConfig = field(default_factory=TechnicalConfig)
    bursts: BurstConfig = field(default_factory=BurstConfig)
    shots: ShotConfig = field(default_factory=ShotConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    scene: "SceneConfig" = field(default=None)  # populated in __post_init__

    # Labeled dataset — used ONLY by `python -m image_filterer.train`.
    train_root: Path = field(default_factory=lambda: Path(
        os.environ.get("IMAGE_FILTERER_TRAIN_ROOT", str(Path.cwd() / "dataset"))))
    data_root: Path = field(default_factory=_data_root)

    def __post_init__(self) -> None:
        if self.scene is None:
            from .scene import SceneConfig
            self.scene = SceneConfig()

    # Some helpers duck-type on `cfg.paths.cache_dir`; expose a shim so a plain
    # Config works everywhere a Paths-style object is expected.
    @property
    def paths(self) -> "Config":
        return self

    @property
    def cache_dir(self) -> Path:
        return Path(os.environ.get("IMAGE_FILTERER_CACHE_DIR", str(self.data_root / "cache")))

    @property
    def runs_dir(self) -> Path:
        return self.data_root / "runs"

    @property
    def db_path(self) -> Path:
        return self.data_root / "registry.db"

    @property
    def model_path(self) -> Path:
        return Path(os.environ.get("IMAGE_FILTERER_MODEL_PATH", str(PACKAGE_DIR / "model" / "ranker.pt")))

    @property
    def export_dir(self) -> Optional[Path]:
        """Server-side export destination (e.g. a LucidLink share). Unset = the
        viewer only offers browser downloads and browser-picked local folders."""
        v = os.environ.get("IMAGE_FILTERER_EXPORT_DIR", "").strip()
        return Path(v).expanduser() if v else None

    @property
    def password(self) -> Optional[str]:
        """Shared password gating the whole server. Unset = auth disabled.

        There are no accounts — this is one password for anyone with the
        link, meant to keep randoms off a link shared at a live event, not to
        distinguish who's who (that's the separate per-browser viewer id).
        """
        return os.environ.get("IMAGE_FILTERER_PASSWORD") or None

    def ensure_dirs(self) -> None:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)


_BASE_EXTS = [".jpg", ".jpeg", ".png", ".webp"]
# RAW is read via its embedded preview (see imaging.open_image), so these are
# first-class inputs rather than a special case.
_RAW_EXTS = [".cr2", ".cr3", ".arw", ".nef", ".raf", ".orf", ".rw2", ".dng"]
IMAGE_EXTS: List[str] = (
    [e for e in _BASE_EXTS] + [e.upper() for e in _BASE_EXTS]
    + [e for e in _RAW_EXTS] + [e.upper() for e in _RAW_EXTS]
)

# Training-set folder names. -1 means "genuinely unlabeled" — those frames are
# excluded from training rather than treated as negatives.
LABEL_FOLDERS = {
    "Good": 1,
    "Bad": 0,
    "Debatable": -1,
    "Questionable": -1,
}

MUST_FIND_FOLDERS: List[str] = ["Top-5to10", "MustFind"]


def default_config() -> Config:
    """The production configuration (what the shipped model was trained with)."""
    return Config()
