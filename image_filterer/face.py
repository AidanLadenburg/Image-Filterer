"""Face detection + per-face quality scalars + (optional) valence / arousal.

Three things this module produces, per image:

1. **Crops**: full frame, expanded face crop, and a person/body crop.
2. **Quality scalars** on the face crop:
   - `sharpness` — variance of Laplacian on the grayscale face crop
   - `face_frac` — face area / frame area
   - `ear` — eye aspect ratio (mean over both eyes)
   - `exposure_clipped_frac` — fraction of face pixels that are pure black or pure white
   - `face_centeredness` — 1 - normalized distance from face center to frame center
   - `mouth_open` — MAR (mouth aspect ratio); large = mouth open mid-speech
3. **Valence / arousal** on the face crop, via HSEmotion ONNX (continuous; optional).

If MediaPipe doesn't detect a face the image is *not* rejected — the crop
falls back to a center square and quality scalars are filled with neutral values
(so wide stage shots still get scored by the embedding stage).

Uses MediaPipe's modern Tasks API (FaceLandmarker). The model file is downloaded
on first use to ``$IMAGE_FILTERER_ASSET_DIR/mp_models/face_landmarker.task``.
"""

from __future__ import annotations

import os
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


# MediaPipe FaceLandmarker uses the same 478-point face-mesh topology, so these
# indices match the legacy mp.solutions.face_mesh layout.
_LEFT_EYE = (362, 385, 387, 263, 373, 380)
_RIGHT_EYE = (33, 160, 158, 133, 153, 144)
_MOUTH = (78, 308, 13, 14)  # left, right, top, bottom

_FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


def _ensure_face_landmarker_model(model_root: Optional[Path] = None) -> Path:
    """Download the face_landmarker.task model on first use; return its path."""
    from .config import asset_dir
    root = model_root or (asset_dir() / "mp_models")
    root.mkdir(parents=True, exist_ok=True)
    target = root / "face_landmarker.task"
    if target.exists() and target.stat().st_size > 0:
        return target
    print(f"[face] downloading MediaPipe face_landmarker.task to {target} ...")
    tmp = target.with_suffix(".task.part")
    urllib.request.urlretrieve(_FACE_LANDMARKER_URL, tmp.as_posix())
    os.replace(tmp, target)
    return target


@dataclass
class FaceMetrics:
    face_detected: bool = False
    face_frac: float = 0.0
    sharpness: float = 0.0
    ear: float = 1.0
    mar: float = 0.0
    exposure_clipped_frac: float = 0.0
    face_centeredness: float = 0.5
    bbox_xyxy: Optional[Tuple[int, int, int, int]] = None
    valence: float = 0.0
    arousal: float = 0.0
    va_available: bool = False

    def as_dict(self) -> Dict[str, float]:
        return {
            "face_detected": float(self.face_detected),
            "face_frac": float(self.face_frac),
            "sharpness": float(self.sharpness),
            "ear": float(self.ear),
            "mar": float(self.mar),
            "exposure_clipped_frac": float(self.exposure_clipped_frac),
            "face_centeredness": float(self.face_centeredness),
            "valence": float(self.valence),
            "arousal": float(self.arousal),
            "va_available": float(self.va_available),
        }


def _euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _ear_from_landmarks(lms: np.ndarray, idx: Tuple[int, ...]) -> float:
    p1, p2, p3, p4, p5, p6 = [lms[i] for i in idx]
    h1 = _euclid(p2, p6)
    h2 = _euclid(p3, p5)
    w = _euclid(p1, p4)
    return float((h1 + h2) / (2.0 * w)) if w else 0.0


def _mar_from_landmarks(lms: np.ndarray) -> float:
    left, right, top, bot = [lms[i] for i in _MOUTH]
    w = _euclid(left, right)
    h = _euclid(top, bot)
    return float(h / w) if w else 0.0


# --------------------------------------------------------------------- detector


class FaceAnalyzer:
    """Wrap MediaPipe FaceLandmarker + lightweight quality stats. Reuse across many images."""

    def __init__(self, *, face_expand: float = 1.55, va_provider: Optional["ValenceArousalModel"] = None) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_py
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        model_path = _ensure_face_landmarker_model()
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_py.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.2,
            min_face_presence_confidence=0.2,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self.face_expand = face_expand
        self.va = va_provider

    # ----- crops -----

    @staticmethod
    def _center_crop(pil: Image.Image, side: Optional[int] = None) -> Image.Image:
        w, h = pil.size
        s = side or min(w, h)
        left = max(0, (w - s) // 2)
        top = max(0, (h - s) // 2)
        return pil.crop((left, top, left + s, top + s))

    def _square_around(self, cx: float, cy: float, side: float, w: int, h: int) -> Tuple[int, int, int, int]:
        half = side / 2
        x0 = int(max(0, min(w - 1, round(cx - half))))
        y0 = int(max(0, min(h - 1, round(cy - half))))
        x1 = int(max(x0 + 1, min(w, round(cx + half))))
        y1 = int(max(y0 + 1, min(h, round(cy + half))))
        return x0, y0, x1, y1

    # ----- main entry -----

    def analyze(self, pil: Image.Image) -> Tuple[Image.Image, Image.Image, Image.Image, FaceMetrics]:
        rgb = np.asarray(pil.convert("RGB"))
        h, w = rgb.shape[:2]
        full_pil = pil if pil.mode == "RGB" else pil.convert("RGB")

        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        metrics = FaceMetrics()

        if not result.face_landmarks:
            face_pil = self._center_crop(full_pil)
            body_pil = full_pil.copy()
            metrics.sharpness = self._sharpness(np.asarray(face_pil.convert("L")))
            metrics.exposure_clipped_frac = self._clipped_frac(np.asarray(face_pil.convert("L")))
            return full_pil, face_pil, body_pil, metrics

        lm = result.face_landmarks[0]
        pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        cx, cy = float(x_min + bbox_w / 2), float(y_min + bbox_h / 2)
        side = float(max(bbox_w, bbox_h)) * self.face_expand

        x0, y0, x1, y1 = self._square_around(cx, cy, side, w, h)
        face_pil = full_pil.crop((x0, y0, x1, y1))

        # body = same horizontal as face but extending to image bottom + a bit of headroom
        body_x0 = int(max(0, x0 - bbox_w * 0.5))
        body_x1 = int(min(w, x1 + bbox_w * 0.5))
        body_y0 = int(max(0, y0 - bbox_h * 0.25))
        body_y1 = int(min(h, max(y1 + 4 * bbox_h, h)))
        body_pil = full_pil.crop((body_x0, body_y0, body_x1, body_y1))

        metrics.face_detected = True
        metrics.bbox_xyxy = (x0, y0, x1, y1)
        metrics.face_frac = float(((x1 - x0) * (y1 - y0)) / (w * h))
        metrics.face_centeredness = 1.0 - float(
            np.hypot((cx - w / 2) / (w / 2), (cy - h / 2) / (h / 2))
        ) / np.sqrt(2.0)

        face_gray = np.asarray(face_pil.convert("L"))
        metrics.sharpness = self._sharpness(face_gray)
        metrics.exposure_clipped_frac = self._clipped_frac(face_gray)

        # EAR / MAR are best computed in the original frame coordinates
        metrics.ear = 0.5 * (_ear_from_landmarks(pts, _LEFT_EYE) + _ear_from_landmarks(pts, _RIGHT_EYE))
        metrics.mar = _mar_from_landmarks(pts)

        if self.va is not None:
            try:
                metrics.valence, metrics.arousal = self.va.predict(face_pil)
                metrics.va_available = True
            except Exception as e:  # noqa: BLE001
                warnings.warn(f"Valence/arousal model failed on a frame: {e}", stacklevel=2)

        return full_pil, face_pil, body_pil, metrics

    # ----- quality helpers -----

    @staticmethod
    def _sharpness(gray: np.ndarray) -> float:
        if gray.size == 0:
            return 0.0
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _clipped_frac(gray: np.ndarray) -> float:
        if gray.size == 0:
            return 0.0
        clipped = (gray <= 2) | (gray >= 253)
        return float(clipped.mean())

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass


# ------------------------------------------------- YuNet two-stage face analyzer


def find_yunet_model() -> Optional[Path]:
    """Locate the YuNet ONNX model (``YUNET_MODEL`` env override, else the asset dir)."""
    from .config import asset_dir
    env = os.environ.get("YUNET_MODEL")
    if env and Path(env).is_file():
        return Path(env)
    root = asset_dir() / "yunet"
    cands = sorted(root.glob("*.onnx")) if root.is_dir() else []
    return cands[0] if cands else None


class YuNetTwoStageAnalyzer:
    """Drop-in replacement for FaceAnalyzer using YuNet for detection.

    YuNet (fast, small-face-capable) finds the face box on the full frame; the box
    is expanded and cropped; then MediaPipe FaceLandmarker runs on that crop —
    where the face now fills the frame, so eyelid/mouth landmarks (EAR/MAR) are
    reliable. Same ``analyze() -> (full, face, body, metrics)`` contract.
    """

    def __init__(self, *, face_expand: float = 1.55, det_size: int = 1024, score_threshold: float = 0.6) -> None:
        model = find_yunet_model()
        if model is None:
            raise RuntimeError(
                "YuNet model not found. Run `python -m scripts.fetch_assets`, or set "
                "YUNET_MODEL, or place face_detection_yunet_*.onnx in the asset dir."
            )
        self._det = cv2.FaceDetectorYN.create(str(model), "", (320, 320), score_threshold, 0.3, 5000)
        self._mp = FaceAnalyzer(face_expand=face_expand)  # landmark extraction on the crop
        self.face_expand = face_expand
        self.det_size = int(det_size)

    def _detect_box(self, pil: Image.Image):
        img = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        scale = self.det_size / max(h, w)
        rw, rh = max(1, int(w * scale)), max(1, int(h * scale))
        r = cv2.resize(img, (rw, rh))
        self._det.setInputSize((rw, rh))
        _, faces = self._det.detect(r)
        if faces is None or len(faces) == 0:
            return None
        f = max(faces, key=lambda ff: ff[2] * ff[3])
        x, y, bw, bh = (float(v) / scale for v in f[:4])
        return x, y, bw, bh

    def analyze(self, pil: Image.Image) -> Tuple[Image.Image, Image.Image, Image.Image, FaceMetrics]:
        full = pil if pil.mode == "RGB" else pil.convert("RGB")
        W, H = full.size
        box = self._detect_box(full)
        metrics = FaceMetrics()

        if box is None:
            face = FaceAnalyzer._center_crop(full)
            metrics.sharpness = FaceAnalyzer._sharpness(np.asarray(face.convert("L")))
            metrics.exposure_clipped_frac = FaceAnalyzer._clipped_frac(np.asarray(face.convert("L")))
            return full, face, full.copy(), metrics

        x, y, bw, bh = box
        cx, cy = x + bw / 2, y + bh / 2
        side = max(bw, bh) * self.face_expand
        x0 = int(max(0, cx - side / 2)); y0 = int(max(0, cy - side / 2))
        x1 = int(min(W, cx + side / 2)); y1 = int(min(H, cy + side / 2))
        face = full.crop((x0, y0, x1, y1))

        body_x0 = int(max(0, x0 - bw * 0.5)); body_x1 = int(min(W, x1 + bw * 0.5))
        body_y0 = int(max(0, y0 - bh * 0.25)); body_y1 = int(min(H, max(y1 + 4 * bh, H)))
        body = full.crop((body_x0, body_y0, body_x1, body_y1))

        metrics.face_detected = True
        metrics.bbox_xyxy = (x0, y0, x1, y1)
        metrics.face_frac = float(((x1 - x0) * (y1 - y0)) / (W * H))
        metrics.face_centeredness = 1.0 - float(
            np.hypot((cx - W / 2) / (W / 2), (cy - H / 2) / (H / 2))
        ) / np.sqrt(2.0)

        # EAR / MAR from MediaPipe landmarks on the (now face-filling) crop.
        _, _, _, mc = self._mp.analyze(face)
        if mc.face_detected:
            metrics.ear = mc.ear
            metrics.mar = mc.mar
        metrics.sharpness = FaceAnalyzer._sharpness(np.asarray(face.convert("L")))
        metrics.exposure_clipped_frac = FaceAnalyzer._clipped_frac(np.asarray(face.convert("L")))
        return full, face, body, metrics

    def close(self) -> None:
        self._mp.close()


# --------------------------------------------------- optional valence / arousal


class ValenceArousalModel:
    """Wrapper around HSEmotion ONNX (Andrey Savchenko) for continuous V/A.

    Falls back to no-op if hsemotion-onnx isn't installed. Uses the multi-task
    EfficientNet-B2 head trained on AffectNet which directly outputs valence and
    arousal scalars per face crop in [-1, 1].
    """

    _INSTANCE: Optional["ValenceArousalModel"] = None

    def __init__(self) -> None:
        try:
            from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
        except ImportError as e:
            raise RuntimeError(
                "hsemotion-onnx is not installed. Install it (`pip install hsemotion-onnx`) or "
                "set use_valence_arousal=False in config."
            ) from e
        # `enet_b2_8_va_mtl` outputs 8 emotions + valence + arousal jointly.
        self._model = HSEmotionRecognizer(model_name="enet_b2_8_va_mtl")

    def predict(self, face_pil: Image.Image) -> Tuple[float, float]:
        face_bgr = cv2.cvtColor(np.asarray(face_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        # The recognizer expects a face crop and returns (label, scores) — for the V/A
        # multitask head, scores includes valence and arousal in the last two slots.
        try:
            _, scores = self._model.predict_emotions(face_bgr, logits=False)
            scores = np.asarray(scores).ravel()
            valence = float(scores[-2])
            arousal = float(scores[-1])
        except Exception:
            # Some hsemotion-onnx versions expose multi_task differently
            scores = self._model.predict_multi_emotions(face_bgr, logits=False)
            arr = np.asarray(scores).ravel()
            valence = float(arr[-2])
            arousal = float(arr[-1])
        return valence, arousal

    @classmethod
    def get(cls) -> "ValenceArousalModel":
        if cls._INSTANCE is None:
            cls._INSTANCE = cls()
        return cls._INSTANCE


def maybe_get_va() -> Optional[ValenceArousalModel]:
    """Return a V/A model if installed, else None (with a warning)."""
    try:
        return ValenceArousalModel.get()
    except RuntimeError as e:
        warnings.warn(str(e), stacklevel=2)
        return None
