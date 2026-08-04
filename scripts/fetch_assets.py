"""Download the third-party model weights the pipeline needs.

Four files, none of which are in git (they total ~700 MB):

  FaRL-Base-Patch16-LAIONFace20M-ep64.pth   652 MB  face embedding
  face_detection_yunet_2023mar.onnx         227 KB  face detection
  face_landmarker.task                      3.7 MB  eyelid/mouth landmarks
  yolov8s-pose.pt                            23 MB  person detection (scene tags)

They land in ``$IMAGE_FILTERER_ASSET_DIR`` (default ``~/.local/share/image-filterer/assets``).
Re-running is cheap: anything already present is skipped.

    python -m scripts.fetch_assets
    python -m scripts.fetch_assets --from-cache /path/to/old/cache   # copy instead of download

The two encoder backbones (SigLIP2-SO400M for context, and the ViT-B-16 skeleton
FaRL loads into) are pulled by open_clip on first use and cached in
``~/.cache/huggingface`` — they need no action here.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

from image_filterer.config import asset_dir

FARL_NAME = "FaRL-Base-Patch16-LAIONFace20M-ep64.pth"
YUNET_NAME = "face_detection_yunet_2023mar.onnx"
LANDMARKER_NAME = "face_landmarker.task"
YOLO_NAME = "yolov8s-pose.pt"

# (subdir, filename, url)
ASSETS = [
    ("farl", FARL_NAME,
     "https://github.com/FacePerceiver/FaRL/releases/download/pretrained_weights/" + FARL_NAME),
    ("yunet", YUNET_NAME,
     "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/" + YUNET_NAME),
    ("mp_models", LANDMARKER_NAME,
     "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/"
     "float16/latest/" + LANDMARKER_NAME),
    ("", YOLO_NAME,
     "https://github.com/ultralytics/assets/releases/download/v8.2.0/" + YOLO_NAME),
]


def _target(root: Path, subdir: str, name: str) -> Path:
    return (root / subdir / name) if subdir else (root / name)


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")

    def hook(blocks: int, block_size: int, total: int) -> None:
        if total <= 0:
            return
        pct = min(100.0, 100.0 * blocks * block_size / total)
        print(f"\r  {target.name}: {pct:5.1f}%  of {total/1e6:.0f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, tmp.as_posix(), reporthook=hook)
    tmp.replace(target)
    print()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("fetch_assets", description=__doc__.splitlines()[0])
    ap.add_argument("--from-cache", type=Path, default=None,
                    help="Copy from an existing cache tree instead of downloading "
                         "(searches recursively by filename).")
    ap.add_argument("--force", action="store_true", help="Re-fetch even if present.")
    args = ap.parse_args(argv)

    root = asset_dir()
    root.mkdir(parents=True, exist_ok=True)
    print(f"asset dir: {root}")

    failed = []
    for subdir, name, url in ASSETS:
        target = _target(root, subdir, name)
        if target.is_file() and target.stat().st_size > 0 and not args.force:
            print(f"  {name}: present, skipping")
            continue
        try:
            if args.from_cache:
                src = next(args.from_cache.rglob(name), None)
                if src is None:
                    raise FileNotFoundError(f"{name} not found under {args.from_cache}")
                target.parent.mkdir(parents=True, exist_ok=True)
                print(f"  {name}: copying from {src}")
                shutil.copy2(src, target)
            else:
                print(f"  {name}: downloading")
                _download(url, target)
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: FAILED — {e}")
            failed.append(name)

    if failed:
        print(f"\n{len(failed)} asset(s) failed: {', '.join(failed)}")
        print("Fetch them by hand into the asset dir shown above, then re-run.")
        return 1
    print("\nAll assets present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
