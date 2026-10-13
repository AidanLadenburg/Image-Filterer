"""Per-image multi-source feature extraction with on-disk caching.

Each image yields a `FeatureBundle`:
  - emb_full: context encoder on full frame
  - emb_face: face encoder on the face crop (FaRL or SigLIP2)
  - emb_body: context encoder on the body crop
  - va: (valence, arousal) — zero if unavailable
  - quality: dict of scalar quality metrics (see face.FaceMetrics)

All embeddings are L2-normalized. The cache key is `(file_sha1, kind, encoder_id)`,
so swapping encoders does not invalidate unrelated entries.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm

from .cache_io import FeatureCache, Sha1Memo, config_namespace
from .config import FeatureConfig
from .imaging import open_image, ordered_map
from .encoders import FrozenEncoder, build_encoder, build_face_encoder
from .face import FaceAnalyzer, FaceMetrics, maybe_get_va


class _StubFaceAnalyzer:
    """Drop-in replacement for FaceAnalyzer that skips mediapipe entirely.

    Returns full image as all three crops and neutral default metrics. Used
    only for fast smoke-tests via FeatureConfig.skip_face_detect.
    """

    def analyze(self, pil: Image.Image):
        full = pil if pil.mode == "RGB" else pil.convert("RGB")
        return full, full, full, FaceMetrics()

    def close(self) -> None:
        pass


@dataclass
class FeatureBundle:
    path: Path
    sha1: str
    emb_full: np.ndarray
    emb_face: np.ndarray
    emb_body: np.ndarray
    va: np.ndarray   # (2,) valence, arousal
    quality: Dict[str, float] = field(default_factory=dict)

    @property
    def has_va(self) -> bool:
        return bool(self.quality.get("va_available", 0.0))

    def concat(
        self,
        *,
        use_full: bool,
        use_face: bool,
        use_body: bool,
        use_va: bool,
        use_quality: bool,
    ) -> np.ndarray:
        parts: List[np.ndarray] = []
        if use_full:
            parts.append(self.emb_full)
        if use_face:
            parts.append(self.emb_face)
        if use_body:
            parts.append(self.emb_body)
        if use_va:
            parts.append(self.va)
        if use_quality:
            parts.append(_quality_vector(self.quality))
        return np.concatenate(parts).astype(np.float32)


# Stable order so the linear/MLP feature dimension is deterministic across runs.
QUALITY_KEYS: Sequence[str] = (
    "face_detected",
    "face_frac",
    "sharpness_log",     # log1p of raw Laplacian variance, more linear
    "ear",
    "mar",
    "exposure_clipped_frac",
    "face_centeredness",
)


# Eye / mouth openness when no face was found. These used to default to
# 1.0 / 0.0 — "eyes far wider open than any real face" — and the ranker learned
# that faceless frames (wide stage shots, slides) were keepers: on a new shoot
# they scored ~3 points above frames with a face. Typical values instead; the
# face_detected flag already says the face is missing.
NO_FACE_EAR = 0.25
NO_FACE_MAR = 0.05


def _quality_vector(q: Dict[str, float]) -> np.ndarray:
    sharpness_log = float(np.log1p(max(0.0, q.get("sharpness", 0.0))))
    face = bool(q.get("face_detected", 0.0))
    src = {
        "face_detected": q.get("face_detected", 0.0),
        "face_frac": q.get("face_frac", 0.0),
        "sharpness_log": sharpness_log,
        "ear": q.get("ear", NO_FACE_EAR) if face else NO_FACE_EAR,
        "mar": q.get("mar", NO_FACE_MAR) if face else NO_FACE_MAR,
        "exposure_clipped_frac": q.get("exposure_clipped_frac", 0.0),
        "face_centeredness": q.get("face_centeredness", 0.5),
    }
    return np.array([src[k] for k in QUALITY_KEYS], dtype=np.float32)


# --------------------------------------------------------------------- driver


@contextmanager
def _single_threaded_opencv():
    """While a worker pool runs one image per thread, OpenCV's own 20-way
    parallelism inside each call only oversubscribes the cores."""
    try:
        import cv2
    except ImportError:
        yield
        return
    prev = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        yield
    finally:
        cv2.setNumThreads(prev)


class FeatureExtractor:
    """Builds and caches FeatureBundles. Encoders are loaded lazily on first miss."""

    def __init__(self, fc: FeatureConfig, cache: FeatureCache) -> None:
        self.fc = fc
        self.cache = cache
        # Lives beside the feature cache: both are keyed to the same content and
        # both are safe to share between machines and delete at will.
        self._memo = Sha1Memo(cache.root / "sha1_memo.db")
        self._ctx: FrozenEncoder | None = None
        self._face_enc: FrozenEncoder | None = None
        self._face_an: FaceAnalyzer | None = None
        # Non-default face detectors change the crops, so their emb_face + face
        # meta must not collide with the MediaPipe-cropped cache entries.
        det = getattr(fc, "face_detector", "mediapipe")
        measured = ("face_expand", "yunet_score_threshold", "yunet_det_size",
                    "skip_face_detect", "use_valence_arousal")
        defaults = FeatureConfig()
        self._face_ns = config_namespace(
            "" if det == "mediapipe" else f"+det-{det}",
            {k: getattr(fc, k) for k in measured},
            {k: getattr(defaults, k) for k in measured})
        self._extract_body = getattr(fc, "extract_body", True)
        # Face analyzers for the decode/face worker threads, pooled: a worker
        # borrows one and hands it back. (Keeping one per *thread* leaked: each
        # ingest starts new worker threads, so a long-lived extractor — the
        # server keeps one for its lifetime — gained ~12 analyzers, ~1.1 GB,
        # per hot-folder re-ingest and was OOM-killed mid-event.)
        self._idle_ans: "queue.SimpleQueue" = queue.SimpleQueue()
        self._an_lock = threading.Lock()
        self._thread_ans: list = []            # every analyzer created, for close()
        # Cumulative timings, read by ingest for its per-phase breakdown.
        self.stats = {"decode_face_s": 0.0, "embed_s": 0.0, "computed": 0, "cached": 0,
                      "scene_shared": 0, "scene_s": 0.0}
        # Set by ingest for the duration of a run: new photos then get their
        # scene analysis (scene.py) from the decode done here instead of a
        # second full decode later. Cached scenes are skipped as before.
        self.scene_analyzer = None
        self._scene_pool = None
        self._scene_futures: list = []

    @property
    def _face_meta_kind(self) -> str:
        return "face" + self._face_ns

    # lazy loaders so e.g. listing the cache doesn't load 10GB of weights
    def _ctx_encoder(self) -> FrozenEncoder:
        if self._ctx is None:
            self._ctx = build_encoder(
                self.fc.context_encoder,
                cradio_input_side=self.fc.cradio_input_side,
                role="context",
            )
        return self._ctx

    def _face_encoder(self) -> FrozenEncoder:
        if self._face_enc is None:
            self._face_enc = build_face_encoder(self.fc.face_encoder)
        return self._face_enc

    @contextmanager
    def _pooled_face_analyzer(self):
        """Borrow an analyzer for one image. The pool only grows to the number
        of threads working at once, however many ingests run."""
        try:
            an = self._idle_ans.get_nowait()
        except queue.Empty:
            an = self._new_face_analyzer()
            with self._an_lock:
                self._thread_ans.append(an)
        try:
            yield an
        finally:
            self._idle_ans.put(an)

    def _face_analyzer(self):
        if self._face_an is None:
            self._face_an = self._new_face_analyzer()
        return self._face_an

    def _new_face_analyzer(self):
        if getattr(self.fc, "skip_face_detect", False):
            return _StubFaceAnalyzer()
        if getattr(self.fc, "face_detector", "mediapipe") == "yunet":
            from .face import YuNetTwoStageAnalyzer
            return YuNetTwoStageAnalyzer(
                face_expand=self.fc.face_expand,
                det_size=getattr(self.fc, "yunet_det_size", 1024),
                score_threshold=getattr(self.fc, "yunet_score_threshold", 0.6),
            )
        va = maybe_get_va() if self.fc.use_valence_arousal else None
        return FaceAnalyzer(face_expand=self.fc.face_expand, va_provider=va)

    # ----- scene stage: one GPU thread, overlapping the CPU work -----

    SCENE_QUEUE_MAX = 16      # prepared frames waiting (~25 MB each): backpressure

    def _submit_scene(self, scene, sha1: str, prep) -> None:
        """Hand a prepared frame to the scene thread. Pose + expression run
        one frame at a time (~55 ms); done inline they stalled the decode
        workers, so ingest was bounded by their sum, not their max."""
        from concurrent.futures import ThreadPoolExecutor
        if self._scene_pool is None:
            self._scene_pool = ThreadPoolExecutor(1, thread_name_prefix="scene")
            self._scene_slots = threading.BoundedSemaphore(self.SCENE_QUEUE_MAX)
        self._scene_slots.acquire()

        def run():
            t = time.perf_counter()
            try:
                scene.analyze(sha1, prep)
            finally:
                self.stats["scene_s"] += time.perf_counter() - t
                self._scene_slots.release()
        self._scene_futures.append(self._scene_pool.submit(run))
        self.stats["scene_shared"] += 1

    def wait_scenes(self) -> None:
        """Block until every submitted scene is analyzed and cached; re-raise
        the first failure."""
        futures, self._scene_futures = self._scene_futures, []
        for f in futures:
            f.result()

    # ----- public -----

    def extract_paths(
        self, paths: Sequence[Path], desc: str = "features", chunk_size: int = 128
    ) -> List[FeatureBundle]:
        """Extract or load-from-cache features for a list of image paths.

        Processes paths in chunks of ``chunk_size`` so that decoded PIL images
        and crops stay bounded in memory. Without chunking, every full-res RGB
        image is held in RAM until the embedding phase, which for thousands of
        ~25 MP JPEGs exhausts memory and drives the system into swap.
        """
        if chunk_size <= 0 or len(paths) <= chunk_size:
            return self._extract_chunk(paths, desc)
        out: List[FeatureBundle] = []
        for start in range(0, len(paths), chunk_size):
            sub = paths[start : start + chunk_size]
            out.extend(self._extract_chunk(sub, desc=f"{desc} {start}-{start+len(sub)}/{len(paths)}"))
        return out

    def _extract_chunk(self, paths: Sequence[Path], desc: str) -> List[FeatureBundle]:
        # Triage which paths actually need recompute (per kind).
        sha1s = self._memo.sha1_many(paths)
        ctx_id = self._ctx_id_lazy()
        face_id = self._face_id_lazy()
        # Body crop depends on the detected face box, so its embedding is also
        # detector-specific (ns is "" for MediaPipe → existing cache stays valid).
        body_id = ctx_id + self._face_ns

        # Phase 1: decode + face crop + quality, then each encoder's own
        # preprocessing — all per-image CPU work, run on a thread pool (face
        # analyzers borrowed from a pool; YuNet/MediaPipe instances aren't shareable). This
        # was the whole bottleneck: one core decoding, detecting and resizing
        # 25 MP photos while the GPU idled. Workers hand back small model-ready
        # tensors, so full-resolution images are freed as soon as they're used.
        # Same transforms, same order, same batches → identical embeddings.
        quality: Dict[int, Dict[str, float]] = {}
        need_recrop, need_full, need_face, need_body = [], set(), set(), set()
        for i, (p, sha) in enumerate(zip(paths, sha1s)):
            if not self.cache.has(sha, "emb_full", ctx_id):
                need_full.add(i)
            if not self.cache.has(sha, "emb_face", face_id):
                need_face.add(i)
            if self._extract_body and not self.cache.has(sha, "emb_body", body_id):
                need_body.add(i)
            if i in need_full or i in need_face or i in need_body \
                    or self.cache.load_meta(sha, self._face_meta_kind) is None:
                need_recrop.append(i)

        ctx = self._ctx_encoder() if need_full or need_body else None
        face_enc = self._face_encoder() if need_face else None
        full_t, face_t, body_t = {}, {}, {}

        scene = self.scene_analyzer

        def prep(i):
            with open_image(paths[i]) as source:
                im = source.convert("RGB")
            sp = None
            if scene is not None and not scene.cached(sha1s[i]):
                # The scene step's own decode (reduced-size JPEG decode +
                # resize, exactly as before so results are unchanged), done
                # here on the worker threads alongside the face work.
                with open_image(paths[i]) as source:
                    sp = scene.prepare(source)
            with self._pooled_face_analyzer() as an:
                full, face, body, m = an.analyze(im)
            return (i, m,
                    ctx.prepare(full) if i in need_full else None,
                    face_enc.prepare(face) if i in need_face else None,
                    ctx.prepare(body) if i in need_body else None, sp)

        t_prep = time.perf_counter()
        if need_recrop:
            with _single_threaded_opencv():
                for i, m, ft, fa, bt, sp in tqdm(ordered_map(prep, need_recrop), total=len(need_recrop),
                                             desc=f"{desc}: decode/face/prep",
                                             disable=os.environ.get("TQDM_DISABLE") == "1"):
                    quality[i] = m.as_dict()
                    if sp is not None:                 # pose + expression on their own thread
                        self._submit_scene(scene, sha1s[i], sp)
                    self.cache.save_meta(sha1s[i], self._face_meta_kind, quality[i])
                    if ft is not None:
                        full_t[i] = ft
                    if fa is not None:
                        face_t[i] = fa
                    if bt is not None:
                        body_t[i] = bt

        self.stats["decode_face_s"] += time.perf_counter() - t_prep
        self.stats["computed"] += len(need_recrop)
        self.stats["cached"] += len(paths) - len(need_recrop)
        t_embed = time.perf_counter()
        # Phase 2: embeddings, batched per (encoder, kind). Only what isn't cached:
        # when the user switches just the face encoder, full/body stay cached.
        for kind, tensors, enc, enc_id in (
                ("emb_full", full_t, ctx, lambda e: e.encoder_id()),
                ("emb_body", body_t, ctx, lambda e: e.encoder_id() + self._face_ns),
                ("emb_face", face_t, face_enc, lambda e: e.encoder_id() + self._face_ns)):
            if not tensors:
                continue
            order = sorted(tensors)
            X = enc.embed_tensors([tensors[i] for i in order], batch_size=self.fc.batch_size)
            for i, z in zip(order, X):
                self.cache.save(sha1s[i], kind, {"z": z.astype(np.float32)}, enc_id(enc))

        self.stats["embed_s"] += time.perf_counter() - t_embed
        # Phase 3: assemble bundles from cache (everything is now present).
        bundles: List[FeatureBundle] = []
        for p, sha in zip(paths, sha1s):
            meta = self.cache.load_meta(sha, self._face_meta_kind) or {}
            va = np.array([meta.get("valence", 0.0), meta.get("arousal", 0.0)], dtype=np.float32)
            ef = self.cache.load(sha, "emb_full", ctx_id)["z"]
            efa = self.cache.load(sha, "emb_face", face_id)["z"]
            eb = self.cache.load(sha, "emb_body", body_id)["z"] if self._extract_body else np.zeros(0, dtype=np.float32)
            bundles.append(
                FeatureBundle(
                    path=p,
                    sha1=sha,
                    emb_full=ef.astype(np.float32),
                    emb_face=efa.astype(np.float32),
                    emb_body=eb.astype(np.float32),
                    va=va,
                    quality=meta,
                )
            )
        return bundles

    # ----- ids: avoid loading encoders just to compute strings -----

    def _ctx_id_lazy(self) -> str:
        # Production architecture IDs are fixed; cache hits need no model load.
        known = {"siglip2_so400m_14": "siglip2_so400m_14@256d1152",
                 "siglip2_so400m_16_384": "siglip2_so400m_16_384@384d1152"}
        name = self.fc.context_encoder.strip().lower()
        return known[name] if name in known else self._ctx_encoder().encoder_id()

    def _face_id_lazy(self) -> str:
        name = self.fc.face_encoder.strip().lower()
        known = {"farl_base": "farl_base@224d512",
                 "siglip2_so400m_14": "siglip2_so400m_14@256d1152"}
        base = known[name] if name in known else self._face_encoder().encoder_id()
        return base + self._face_ns

    def close(self) -> None:
        if self._scene_pool is not None:
            self._scene_pool.shutdown(wait=True)
            self._scene_pool = None
        with self._an_lock:
            thread_ans, self._thread_ans = self._thread_ans, []
        for an in thread_ans:
            an.close()
        self._idle_ans = queue.SimpleQueue()
        if self._face_an is not None:
            self._face_an.close()
