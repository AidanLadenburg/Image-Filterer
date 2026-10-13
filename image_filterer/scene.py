"""People, pose and the speaker's expression — one pass per frame, cached raw.

Per frame (cached by content hash, so re-tuning any threshold below costs
nothing — no image is reprocessed):

  • YOLO11x-pose at 1280 px → every person's box and 17 body keypoints.
    (YOLOv8s at its default 640 px saw a speaker on a wide stage ~15 px tall
    and kept only box sizes.)
  • The speaker: see :func:`subject`.
  • The speaker's face, cropped from a ~3000 px decode around the face
    keypoints → MediaPipe FaceLandmarker: 52 expression blendshapes (blink,
    squint, smile, jaw open mid-word, brow raise, lip press …) and head
    yaw / pitch / roll. These feed the ranker: in held-out tests they were the
    biggest single gain at picking the best frame of a burst (70% → 77-83%).
  • How lit the background is around the speaker (for Hero).

Read-time classification on top of that (all thresholds in SceneConfig /
ShotConfig): Subject (people / stage), Hero, and shot scale (shots.py).

Cost on the GB10: ~26 ms GPU (pose, fp16) + ~7 ms CPU (face) per new frame;
decoding and resizing run on worker threads alongside.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

from .cache_io import FeatureCache, config_namespace

SCENE_CACHE_ID = "pose11x"   # cache namespace for the raw per-frame scene data
N_PERSONS = 5                # people kept per frame (largest first)
N_BLENDSHAPES = 52
DECODE_LONG = 3000           # decode size for detection and the face crop
KP_CONF = 0.35               # a keypoint counts as visible above this


@dataclass
class SceneConfig:
    enabled: bool = True
    yolo_model: str = "yolo11x-pose.pt"
    imgsz: int = 1280
    # Subject: people vs stage
    person_min_area: float = 0.04      # the speaker's box covers at least this much of the frame
    # Hero: the speaker alone against a mostly dark, plain background
    hero_min_height: float = 0.45      # speaker fills at least this much of the frame height
    hero_bg_max: float = 0.06          # at most this share of the area around them is lit
    bg_luma: int = 55                  # luma (0-255) above which a background pixel is "lit"
    bg_small: int = 240                # long side of the grayscale copy the background is measured on


@dataclass
class ScenePrep:
    rgb: np.ndarray             # upright RGB, ~DECODE_LONG on the long side (face crop)
    det: np.ndarray             # BGR at the detector's size (resized on the worker thread)
    gray_small: np.ndarray      # bg_small-px grayscale for the background measure


@dataclass
class SceneData:
    """Everything cached for one frame. Coordinates are fractions of the
    (upright) frame."""
    boxes: np.ndarray           # (N_PERSONS, 5)  x0, y0, x1, y1, confidence  (0 = no person)
    kps: np.ndarray             # (N_PERSONS, 17, 3) x, y, confidence
    subject: int                # index into boxes of the speaker, -1 if none
    blendshapes: np.ndarray     # (52,) zeros when the face wasn't measured
    head_pose: np.ndarray       # (3,) yaw, pitch, roll in degrees
    face_found: bool
    face_px: float              # face-crop height in px at DECODE_LONG scale
    bg_lit: float               # share of the area around the speaker that is lit (1.0 = unknown)

    def expression_vector(self) -> np.ndarray:
        """The ranker's expression input: 52 blendshapes + head pose + found + crop size."""
        hp = np.r_[self.head_pose / 45.0, float(self.face_found), np.log1p(self.face_px)]
        return np.r_[self.blendshapes, hp].astype(np.float32)

    def subject_box(self) -> Optional[np.ndarray]:
        return self.boxes[self.subject] if self.subject >= 0 else None

    def to_arrays(self) -> dict:
        return {"boxes": self.boxes, "kps": self.kps, "subject": np.array(self.subject),
                "bs": self.blendshapes, "head_pose": self.head_pose, "face_found": np.array(self.face_found),
                "face_px": np.array(self.face_px), "bg_lit": np.array(self.bg_lit)}

    @classmethod
    def from_arrays(cls, z) -> "SceneData":
        return cls(z["boxes"], z["kps"], int(z["subject"]), z["bs"], z["head_pose"], bool(z["face_found"]),
                   float(z["face_px"]), float(z["bg_lit"]))


def subject(boxes: np.ndarray, kps: np.ndarray, min_h: float = 0.12) -> int:
    """The main person, or -1.

    Each confident person is scored by height × how clearly their face and
    shoulders are visible. People cut off by the left/right edge (figures on a
    projected slide, passers-by) count for less, and low, barely visible
    figures (the backs of audience heads) are skipped. Picking simply the
    largest box chose a slide figure or an audience head in many tight shots."""
    best, best_s = -1, 0.0
    for i in range(len(boxes)):
        if boxes[i, 4] < 0.4:
            continue
        x0, y0, x1, y1 = boxes[i, :4]
        h = y1 - y0
        q = float(kps[i, [0, 1, 2, 5, 6], 2].mean())        # nose, eyes, shoulders
        if h < min_h or q < 0.3:
            continue
        if y0 > 0.55 and q < 0.6:
            continue                                         # audience in the foreground
        edge = 0.4 if (x0 < 0.01 or x1 > 0.99) and (x1 - x0) < 0.5 else 1.0
        sc = h * (0.3 + q) * edge
        if sc > best_s:
            best, best_s = i, sc
    return best


def _head_crop_box(kps: np.ndarray, W: int, H: int) -> Optional[Tuple[int, int, int, int]]:
    """Square-ish box around the face keypoints, in pixels; None if the face isn't visible."""
    fk = kps[:5]; ok = fk[:, 2] > 0.4
    if ok.sum() < 2:
        return None
    cx, cy = fk[ok, 0].mean() * W, fk[ok, 1].mean() * H
    sh = kps[5:7]
    span = np.ptp(fk[ok, 0] * W) * 1.6
    if (sh[:, 2] > 0.4).all():
        span = max(span, abs(sh[0, 0] - sh[1, 0]) * W * 0.55)
    half = max(span, 12) * 1.25
    x0, y0 = max(0, int(cx - half)), max(0, int(cy - half * 1.1))
    x1, y1 = min(W, int(cx + half)), min(H, int(cy + half * 0.9))
    if x1 - x0 < 24 or y1 - y0 < 24:
        return None
    return x0, y0, x1, y1


def _bg_lit(gray: np.ndarray, boxes: np.ndarray, subj: int, cfg: SceneConfig) -> float:
    """Share of lit pixels in the area around and behind the speaker: 1.5 body
    widths either side, head to feet, every detected person excluded. Measuring
    only there (not the whole frame) keeps the stage lighting rig along the top
    edge from disqualifying a clean shot."""
    if subj < 0:
        return 1.0
    H, W = gray.shape
    x0, y0, x1, y1 = boxes[subj, :4]
    w, h = x1 - x0, y1 - y0
    m = np.zeros((H, W), bool)
    m[int(max(0, y0 - 0.15 * h) * H):int(min(1, y1) * H), int(max(0, x0 - 1.5 * w) * W):int(min(1, x1 + 1.5 * w) * W)] = True
    for b in boxes:
        if b[4] >= 0.3:
            m[int(b[1] * H):int(b[3] * H) + 1, int(b[0] * W):int(b[2] * W) + 1] = False
    return float((gray[m] > cfg.bg_luma).mean()) if m.any() else 1.0


class SceneAnalyzer:
    """Computes and caches :class:`SceneData` per frame."""

    def __init__(self, cfg: SceneConfig, cache: FeatureCache, device: Optional[str] = None) -> None:
        self.cfg = cfg
        self.cache = cache
        self.device = device
        self._model = None
        self._landmarker = None
        defaults = SceneConfig()
        measured = ("yolo_model", "imgsz", "bg_luma", "bg_small")
        self.cache_id = config_namespace(SCENE_CACHE_ID,
            {k: getattr(cfg, k) for k in measured},
            {k: getattr(defaults, k) for k in measured})

    def _yolo(self):
        if self._model is None:
            from ultralytics import YOLO

            from .config import asset_dir
            # Prefer a pre-fetched local copy; otherwise pass the bare name and
            # let ultralytics download it (see scripts/fetch_assets.py).
            local = asset_dir() / self.cfg.yolo_model
            self._model = YOLO(str(local) if local.is_file() else self.cfg.yolo_model)
        return self._model

    def _face(self):
        if self._landmarker is None:
            from mediapipe.tasks import python as mp_py
            from mediapipe.tasks.python import vision as mp_vision

            from .face import _ensure_face_landmarker_model
            opts = mp_vision.FaceLandmarkerOptions(
                base_options=mp_py.BaseOptions(model_asset_path=str(_ensure_face_landmarker_model())),
                running_mode=mp_vision.RunningMode.IMAGE, num_faces=1,
                output_face_blendshapes=True, output_facial_transformation_matrixes=True,
                min_face_detection_confidence=0.3)
            from .face import quiet_native_stderr
            with quiet_native_stderr():
                self._landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
        return self._landmarker

    def prepare(self, img: Image.Image) -> ScenePrep:
        """Everything per-image that doesn't need a model — safe on worker
        threads, so the GPU never waits on a 25 MP decode."""
        f = DECODE_LONG / max(img.size)
        if f < 1:
            try:
                img.draft("RGB", (max(1, int(img.size[0] * f)), max(1, int(img.size[1] * f))))
            except Exception:  # noqa: BLE001 — not a JPEG; decoded at full size
                pass
        if img.getexif().get(0x0112, 1) != 1:
            img = ImageOps.exif_transpose(img)        # upright, as shown in the viewer
        img = img.convert("RGB")
        if max(img.size) > DECODE_LONG * 1.25:
            img.thumbnail((DECODE_LONG, DECODE_LONG))
        gray = img.convert("L"); gray.thumbnail((self.cfg.bg_small, self.cfg.bg_small))
        import cv2
        rgb = np.asarray(img)
        h, w = rgb.shape[:2]; f = self.cfg.imgsz / max(h, w)
        # Resizing here, on the worker thread, keeps the GPU thread from
        # spending ~20 ms per frame shrinking a 3000 px array itself.
        det = cv2.resize(rgb, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)[:, :, ::-1] \
            if f < 1 else rgb[:, :, ::-1]
        return ScenePrep(rgb, np.ascontiguousarray(det), np.asarray(gray))

    def cached(self, sha1: str) -> bool:
        return self.cache.has(sha1, "scene", self.cache_id)

    def load(self, sha1: str) -> SceneData:
        return SceneData.from_arrays(self.cache.load(sha1, "scene", self.cache_id))

    def analyze(self, sha1: str, src) -> SceneData:
        """``src`` may be a ScenePrep, an image, or a zero-arg callable
        returning an image (opened only on a cache miss)."""
        if self.cached(sha1):
            return self.load(sha1)
        if isinstance(src, ScenePrep):
            prep = src
        else:
            img = src() if callable(src) else src
            try:
                prep = self.prepare(img)
            finally:
                if callable(src):
                    img.close()
        data = self._measure(prep)
        self.cache.save(sha1, "scene", data.to_arrays(), self.cache_id)
        return data

    def _measure(self, prep: ScenePrep) -> SceneData:
        import cv2
        rgb = prep.rgb; H, W = rgb.shape[:2]
        # fp16 on the GPU: 48 → 26 ms per frame, same people found on every test frame.
        res = self._yolo().predict(prep.det, imgsz=self.cfg.imgsz, verbose=False, device=self.device, conf=0.25,
                                   quantize=16 if self.device is not None else None)[0]
        persons = []
        if res.boxes is not None and len(res.boxes):
            b = res.boxes.xyxy.cpu().numpy(); c = res.boxes.conf.cpu().numpy()
            k = res.keypoints.data.cpu().numpy() if res.keypoints is not None else np.zeros((len(b), 17, 3))
            persons = sorted(zip(b, c, k), key=lambda t: -((t[0][2] - t[0][0]) * (t[0][3] - t[0][1]) * t[1]))
        boxes = np.zeros((N_PERSONS, 5), np.float32); kps = np.zeros((N_PERSONS, 17, 3), np.float32)
        dh, dw = prep.det.shape[:2]                  # detections are in the detector image's pixels
        for i, (bb, cc, kk) in enumerate(persons[:N_PERSONS]):
            boxes[i] = [bb[0] / dw, bb[1] / dh, bb[2] / dw, bb[3] / dh, cc]
            kps[i, :, 0] = kk[:, 0] / dw; kps[i, :, 1] = kk[:, 1] / dh; kps[i, :, 2] = kk[:, 2]
        subj = subject(boxes, kps)
        bs = np.zeros(N_BLENDSHAPES, np.float32); pose = np.zeros(3, np.float32); found = False; px = 0.0
        if subj >= 0:
            box = _head_crop_box(kps[subj], W, H)
            if box is not None:
                x0, y0, x1, y1 = box
                crop = rgb[y0:y1, x0:x1]
                px = float(crop.shape[0]) * DECODE_LONG / max(H, W)
                c = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA if crop.shape[0] > 256 else cv2.INTER_CUBIC)
                import mediapipe as mp
                r = self._face().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(c)))
                if r.face_blendshapes:
                    found = True
                    bs = np.array([x.score for x in r.face_blendshapes[0]], np.float32)
                    m = np.array(r.facial_transformation_matrixes[0])[:3, :3]
                    pose = np.array([np.degrees(np.arctan2(-m[2, 0], np.hypot(m[2, 1], m[2, 2]))),
                                     np.degrees(np.arctan2(m[2, 1], m[2, 2])),
                                     np.degrees(np.arctan2(m[1, 0], m[0, 0]))], np.float32)
        return SceneData(boxes, kps, subj, bs, pose, found, px, _bg_lit(prep.gray_small, boxes, subj, self.cfg))

    def close(self) -> None:
        self._model = None
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:  # noqa: BLE001
                pass
            self._landmarker = None


def classify_scene(scene: SceneData, cfg: SceneConfig) -> Tuple[str, bool]:
    """(subject_class, is_hero).

    Subject: "people" when the speaker's box covers at least person_min_area of
    the frame, else "stage" (empty stage, distant figure, a slide).
    Hero: the speaker alone (nobody else at least half their height), filling
    hero_min_height of the frame, against a mostly dark area around them."""
    b = scene.subject_box()
    if b is None:
        return "stage", False
    w, h = b[2] - b[0], b[3] - b[1]
    subject_class = "people" if w * h >= cfg.person_min_area else "stage"
    others = [j for j in range(len(scene.boxes)) if j != scene.subject and scene.boxes[j, 4] >= 0.4
              and (scene.boxes[j, 3] - scene.boxes[j, 1]) >= 0.5 * h]
    hero = h >= cfg.hero_min_height and not others and scene.bg_lit <= cfg.hero_bg_max
    return subject_class, bool(hero)
