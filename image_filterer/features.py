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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm

from .cache_io import FeatureCache, Sha1Memo, file_sha1
from .config import FeatureConfig
from .imaging import open_image
from .encoders import FrozenEncoder, build_encoder, build_face_encoder_with_fallback
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


def _quality_vector(q: Dict[str, float]) -> np.ndarray:
    sharpness_log = float(np.log1p(max(0.0, q.get("sharpness", 0.0))))
    src = {
        "face_detected": q.get("face_detected", 0.0),
        "face_frac": q.get("face_frac", 0.0),
        "sharpness_log": sharpness_log,
        "ear": q.get("ear", 1.0),
        "mar": q.get("mar", 0.0),
        "exposure_clipped_frac": q.get("exposure_clipped_frac", 0.0),
        "face_centeredness": q.get("face_centeredness", 0.5),
    }
    return np.array([src[k] for k in QUALITY_KEYS], dtype=np.float32)


# --------------------------------------------------------------------- driver


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
        self._face_ns = "" if det == "mediapipe" else f"+det-{det}"
        self._extract_body = getattr(fc, "extract_body", True)

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
            self._face_enc = build_face_encoder_with_fallback(self.fc.face_encoder)
        return self._face_enc

    def _face_analyzer(self):
        if self._face_an is None:
            if getattr(self.fc, "skip_face_detect", False):
                self._face_an = _StubFaceAnalyzer()
            elif getattr(self.fc, "face_detector", "mediapipe") == "yunet":
                from .face import YuNetTwoStageAnalyzer
                self._face_an = YuNetTwoStageAnalyzer(
                    face_expand=self.fc.face_expand,
                    det_size=getattr(self.fc, "yunet_det_size", 1024),
                    score_threshold=getattr(self.fc, "yunet_score_threshold", 0.6),
                )
            else:
                va = maybe_get_va() if self.fc.use_valence_arousal else None
                self._face_an = FaceAnalyzer(face_expand=self.fc.face_expand, va_provider=va)
        return self._face_an

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

        # Phase 1: face crops + quality + VA (cheap-ish, per-image).
        # We need crops to embed, so compute these for any path missing either
        # the crops cache, the embeddings, or the meta.
        face_crops: Dict[int, Image.Image] = {}
        full_imgs: Dict[int, Image.Image] = {}
        body_crops: Dict[int, Image.Image] = {}
        quality: Dict[int, Dict[str, float]] = {}

        need_recrop = []
        for i, (p, sha) in enumerate(zip(paths, sha1s)):
            need_emb = (
                not self.cache.has(sha, "emb_full", ctx_id)
                or not self.cache.has(sha, "emb_face", face_id)
                or (self._extract_body and not self.cache.has(sha, "emb_body", body_id))
                or self.cache.load_meta(sha, self._face_meta_kind) is None
            )
            if need_emb:
                need_recrop.append(i)

        if need_recrop:
            an = self._face_analyzer()
            for i in tqdm(need_recrop, desc=f"{desc}: face/quality"):
                im = open_image(paths[i]).convert("RGB")
                full, face, body, m = an.analyze(im)
                full_imgs[i] = full
                face_crops[i] = face
                body_crops[i] = body
                quality[i] = m.as_dict()
                self.cache.save_meta(sha1s[i], self._face_meta_kind, quality[i])

        # Phase 2: embeddings batch by (encoder, kind). Only embed what isn't cached:
        # when the user switches just the face encoder, full/body should stay cached.
        full_to_embed = [(i, full_imgs[i]) for i in need_recrop if not self.cache.has(sha1s[i], "emb_full", ctx_id)]
        face_to_embed = [(i, face_crops[i]) for i in need_recrop if not self.cache.has(sha1s[i], "emb_face", face_id)]
        body_to_embed = (
            [(i, body_crops[i]) for i in need_recrop if not self.cache.has(sha1s[i], "emb_body", body_id)]
            if self._extract_body else []
        )

        ctx = None
        face_enc = None
        if full_to_embed or body_to_embed:
            ctx = self._ctx_encoder()
        if face_to_embed:
            face_enc = self._face_encoder()

        if full_to_embed:
            X = ctx.embed_images([img for _, img in full_to_embed], batch_size=self.fc.batch_size)
            for (i, _), z in zip(full_to_embed, X):
                self.cache.save(sha1s[i], "emb_full", {"z": z.astype(np.float32)}, ctx.encoder_id())
        if body_to_embed:
            X = ctx.embed_images([img for _, img in body_to_embed], batch_size=self.fc.batch_size)
            for (i, _), z in zip(body_to_embed, X):
                self.cache.save(sha1s[i], "emb_body", {"z": z.astype(np.float32)}, ctx.encoder_id() + self._face_ns)
        if face_to_embed:
            X = face_enc.embed_images([img for _, img in face_to_embed], batch_size=self.fc.batch_size)
            for (i, _), z in zip(face_to_embed, X):
                self.cache.save(sha1s[i], "emb_face", {"z": z.astype(np.float32)}, face_enc.encoder_id() + self._face_ns)

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
        # Mirror the encoder_id() format so we can probe the cache without instantiating weights.
        if self._ctx is not None:
            return self._ctx.encoder_id()
        # Best-effort string: the actual id includes embed_dim / nominal_input_size, which we don't
        # know without instantiation. We *load* the encoder if we'd otherwise have a cache miss; if
        # the cache is fully populated we still need a stable id, so we instantiate once. Cheaper
        # alternative: store a sidecar mapping. For simplicity, instantiate.
        return self._ctx_encoder().encoder_id()

    def _face_id_lazy(self) -> str:
        base = self._face_enc.encoder_id() if self._face_enc is not None else self._face_encoder().encoder_id()
        return base + self._face_ns

    def close(self) -> None:
        if self._face_an is not None:
            self._face_an.close()
