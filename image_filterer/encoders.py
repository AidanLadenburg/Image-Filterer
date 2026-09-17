"""Frozen vision encoders (context + face).

Two encoder families:

* **Context** — global frame / body crop. Defaults to SigLIP2 SO400M; supports
  C-RADIOv4-SO400M for high-res and DINOv2-giant as an alternate.
* **Face** — face crop. Defaults to FaRL (Microsoft, CLIP pretrained on 20M
  LAION face captions). Missing weights fail explicitly to preserve the trained
  ranker’s feature space.

Outputs are L2-normalized float32 row vectors.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


CONTEXT_CHOICES = (
    "siglip2_so400m_14",
    "siglip2_so400m_16_384",
    "cradio_v4_so400m",
    "dinov2_giant",
    "clip_h14_336",
)

FACE_CHOICES = (
    "farl_base",
    "siglip2_so400m_14",
    "dinov2_large",
)


@dataclass
class FrozenEncoder:
    name: str
    device: torch.device
    nominal_input_size: int
    embed_dim: int
    model: nn.Module
    preprocess: Callable[[Image.Image], torch.Tensor]
    _encode_batch: Callable[[torch.Tensor], torch.Tensor]

    def encoder_id(self) -> str:
        """Stable identifier used as part of the cache key."""
        return f"{self.name}@{self.nominal_input_size}d{self.embed_dim}"

    @torch.inference_mode()
    def embed_images(self, images: Sequence[Image.Image], batch_size: int = 8) -> np.ndarray:
        if len(images) == 0:
            return np.zeros((0, self.embed_dim), dtype=np.float32)
        use_amp = self.device.type == "cuda"
        feats: List[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            batch_imgs = images[start : start + batch_size]
            tensors = [self.preprocess(img.convert("RGB")) for img in batch_imgs]
            x = torch.stack(tensors, dim=0).to(self.device, non_blocking=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    z = self._encode_batch(x)
            else:
                z = self._encode_batch(x)
            z = F.normalize(z.float(), dim=1)
            feats.append(z.cpu().numpy().astype(np.float32))
        return np.vstack(feats)


# ----------------------------------------------------------------------- helpers


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _open_clip_dim(model: nn.Module) -> int:
    if getattr(model, "embed_dim", None) is not None:
        return int(model.embed_dim)
    v = model.visual
    for attr in ("output_dim", "embed_dim"):
        val = getattr(v, attr, None)
        if val is not None:
            return int(val)
    proj = getattr(v, "proj", None)
    if proj is not None:
        return int(proj.shape[0])
    trunk = getattr(v, "trunk", None)
    if trunk is not None and getattr(trunk, "num_features", None) is not None:
        return int(trunk.num_features)
    raise RuntimeError("Could not infer OpenCLIP visual embedding dimension.")


# ----------------------------------------------------------------------- backends


def _open_clip_encoder(
    device: torch.device,
    model_name: str,
    pretrained: str,
    key: str,
    nominal_size: int,
) -> FrozenEncoder:
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.eval().to(device)
    dim = _open_clip_dim(model)

    def encode_batch(x: torch.Tensor) -> torch.Tensor:
        return model.encode_image(x)

    return FrozenEncoder(
        name=key,
        device=device,
        nominal_input_size=nominal_size,
        embed_dim=dim,
        model=model,
        preprocess=preprocess,
        _encode_batch=encode_batch,
    )


def _dinov2_encoder(device: torch.device, model_id: str, key: str, default_size: int) -> FrozenEncoder:
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).eval().to(device)

    sz = getattr(processor, "size", None)
    if isinstance(sz, dict):
        size = int(sz.get("height") or sz.get("shortest_edge") or default_size)
    elif isinstance(sz, int):
        size = sz
    else:
        size = default_size

    def preprocess_pil(img: Image.Image) -> torch.Tensor:
        return processor(images=img, return_tensors="pt")["pixel_values"][0]

    def encode_batch(x: torch.Tensor) -> torch.Tensor:
        return model(pixel_values=x).last_hidden_state[:, 0]

    return FrozenEncoder(
        name=key,
        device=device,
        nominal_input_size=size,
        embed_dim=int(model.config.hidden_size),
        model=model,
        preprocess=preprocess_pil,
        _encode_batch=encode_batch,
    )


def _cradio_encoder(device: torch.device, hf_repo: str, key: str, input_side: int) -> FrozenEncoder:
    from transformers import AutoModel, CLIPImageProcessor

    processor = CLIPImageProcessor.from_pretrained(hf_repo)
    model = AutoModel.from_pretrained(hf_repo, trust_remote_code=True).eval().to(device)

    side = max(16, (int(input_side) // 16) * 16)
    if side != input_side:
        warnings.warn(f"C-RADIO side {input_side} rounded to {side} (multiple of 16).", stacklevel=2)
    size_kw = {"height": side, "width": side}

    def preprocess_pil(img: Image.Image) -> torch.Tensor:
        return processor(images=img.convert("RGB"), return_tensors="pt", do_resize=True, size=size_kw)[
            "pixel_values"
        ][0]

    def _align(x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        sh, sw = max(16, (h // 16) * 16), max(16, (w // 16) * 16)
        if sh != h or sw != w:
            x = F.interpolate(x, size=(sh, sw), mode="bicubic", align_corners=False)
        return x

    with torch.inference_mode():
        dummy = Image.new("RGB", (side, side))
        pv = processor(images=dummy, return_tensors="pt", do_resize=True, size=size_kw)["pixel_values"].to(device)
        out = model(_align(pv))
        summary = out.summary if hasattr(out, "summary") else (out[0] if isinstance(out, (tuple, list)) else None)
        if summary is None:
            raise RuntimeError("Unexpected C-RADIO output; cannot infer dim.")
        dim = int(summary.shape[-1])

    def encode_batch(x: torch.Tensor) -> torch.Tensor:
        x = _align(x)
        out = model(x)
        if hasattr(out, "summary"):
            return out.summary
        if isinstance(out, (tuple, list)):
            return out[0]
        raise RuntimeError("Unexpected C-RADIO output.")

    return FrozenEncoder(
        name=key,
        device=device,
        nominal_input_size=side,
        embed_dim=dim,
        model=model,
        preprocess=preprocess_pil,
        _encode_batch=encode_batch,
    )


def _farl_encoder(device: torch.device) -> FrozenEncoder:
    """FaRL = CLIP-base/16 pretrained on LAION-Face-20M (FacePerceiver / Microsoft).

    The released .pth uses OpenAI-CLIP key layout (`visual.conv1`, `visual.proj`,
    `visual.transformer.resblocks.*`), which matches open_clip's ViT-B-16. We
    instantiate an empty ViT-B-16 via open_clip and load the FaRL state_dict
    directly. The HF mirrors that the original loader tried don't exist.

    Search order for the checkpoint:
      1. ``FARL_LOCAL_CKPT`` env var
      2. ``$IMAGE_FILTERER_ASSET_DIR/farl/FaRL-*.pth`` (auto-discovered)

    ``python -m scripts.fetch_assets`` puts it in the right place.
    """
    import os

    import open_clip

    from .config import asset_dir

    local_ckpt = os.environ.get("FARL_LOCAL_CKPT")
    if not local_ckpt:
        default = asset_dir() / "farl"
        cands = sorted(default.glob("FaRL-*.pth")) if default.is_dir() else []
        if cands:
            local_ckpt = str(cands[0])
    if not local_ckpt:
        raise RuntimeError(
            f"FaRL checkpoint not found. Run `python -m scripts.fetch_assets`, or set "
            f"FARL_LOCAL_CKPT, or place FaRL-Base-Patch16-LAIONFace20M-ep64.pth under "
            f"{asset_dir() / 'farl'}/ (download it from "
            "https://github.com/FacePerceiver/FaRL/releases)."
        )

    # Empty ViT-B/16 backbone — its key layout matches the FaRL checkpoint.
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained=None
    )

    sd = torch.load(local_ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    missing, unexpected = model.load_state_dict(sd, strict=False)
    # FaRL ships visual + text towers and projections. We only need visual.* and
    # visual.proj; missing keys for text-side training-only buffers are fine.
    visual_missing = [k for k in missing if k.startswith("visual.")]
    if len(visual_missing) > 5:
        raise RuntimeError(
            f"FaRL checkpoint {local_ckpt}: visual tower has {len(visual_missing)} "
            f"missing keys (first 5: {visual_missing[:5]}). Wrong checkpoint?"
        )

    model.eval().to(device)
    dim = int(model.visual.proj.shape[1])  # 512 for FaRL-B/16

    def encode_batch(x: torch.Tensor) -> torch.Tensor:
        return model.encode_image(x)

    return FrozenEncoder(
        name="farl_base",
        device=device,
        nominal_input_size=224,
        embed_dim=dim,
        model=model,
        preprocess=preprocess,
        _encode_batch=encode_batch,
    )


# ----------------------------------------------------------------------- registry


def build_encoder(
    name: str,
    *,
    cradio_input_side: Optional[int] = None,
    role: str = "context",
) -> FrozenEncoder:
    device = _device()
    n = name.strip().lower()

    if n == "siglip2_so400m_14":
        return _open_clip_encoder(device, "ViT-SO400M-14-SigLIP2", "webli", n, 256)
    if n == "siglip2_so400m_16_384":
        return _open_clip_encoder(device, "ViT-SO400M-16-SigLIP2-384", "webli", n, 384)
    if n == "clip_h14_336":
        return _open_clip_encoder(device, "ViT-H-14-CLIPA-336", "laion2b", n, 336)
    if n == "cradio_v4_so400m":
        return _cradio_encoder(device, "nvidia/C-RADIOv4-SO400M", n, cradio_input_side or 1024)
    if n == "dinov2_giant":
        return _dinov2_encoder(device, "facebook/dinov2-giant", n, 518)
    if n == "dinov2_large":
        return _dinov2_encoder(device, "facebook/dinov2-large", n, 518)
    if n == "farl_base":
        return _farl_encoder(device)

    raise ValueError(
        f"Unknown {role} encoder {name!r}. "
        f"Context: {CONTEXT_CHOICES}. Face: {FACE_CHOICES}."
    )


# ----------------------------------------------------------------------- text tower
#
# Semantic (text→image) search reuses the *context* encoder's joint image-text
# space. SigLIP2 / CLIP are dual-encoder contrastive models: the image tower
# (already run over every frame → ``emb_full``) and the text tower live in the
# same L2-normalized space, so a query embedding can be cosine-ranked directly
# against the cached ``emb_full`` vectors — no re-embedding of images required.
#
# Only CLIP-family context encoders expose a text tower. DINOv2 and C-RADIO are
# vision-only, so text search is unavailable when those are the context encoder.

# context-encoder name → (open_clip model_name, pretrained tag)
_TEXT_TOWER_SPECS = {
    "siglip2_so400m_14": ("ViT-SO400M-14-SigLIP2", "webli"),
    "siglip2_so400m_16_384": ("ViT-SO400M-16-SigLIP2-384", "webli"),
    "clip_h14_336": ("ViT-H-14-CLIPA-336", "laion2b"),
}


def context_encoder_has_text_tower(name: str) -> bool:
    return name.strip().lower() in _TEXT_TOWER_SPECS


@dataclass
class TextEncoder:
    """Text tower matching a CLIP-family context encoder.

    ``embed_texts`` returns L2-normalized float32 rows in the *same* space as the
    context encoder's ``emb_full`` image embeddings.
    """

    name: str
    device: torch.device
    embed_dim: int
    model: nn.Module
    tokenizer: Callable[[Sequence[str]], torch.Tensor]

    @torch.inference_mode()
    def embed_texts(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.embed_dim), dtype=np.float32)
        use_amp = self.device.type == "cuda"
        out: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            toks = self.tokenizer(list(texts[start : start + batch_size])).to(self.device)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    z = self.model.encode_text(toks)
            else:
                z = self.model.encode_text(toks)
            z = F.normalize(z.float(), dim=1)
            out.append(z.cpu().numpy().astype(np.float32))
        return np.vstack(out)


def build_text_encoder(context_encoder_name: str) -> TextEncoder:
    """Build the text tower paired with a CLIP-family context encoder.

    Raises ValueError if the context encoder has no text tower (DINOv2, C-RADIO).
    """
    import open_clip

    n = context_encoder_name.strip().lower()
    if n not in _TEXT_TOWER_SPECS:
        raise ValueError(
            f"Context encoder {context_encoder_name!r} has no text tower; "
            f"text search requires one of {tuple(_TEXT_TOWER_SPECS)}."
        )
    model_name, pretrained = _TEXT_TOWER_SPECS[n]
    device = _device()
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    dim = _open_clip_dim(model)
    return TextEncoder(name=n, device=device, embed_dim=dim, model=model, tokenizer=tokenizer)


def build_face_encoder(name: str) -> FrozenEncoder:
    """Load the requested feature space or fail with its original error."""
    return build_encoder(name, role="face")
