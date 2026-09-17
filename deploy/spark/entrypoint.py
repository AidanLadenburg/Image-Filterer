"""Spark startup gate and an explicit model warm-up / real-image check."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def gpu_check():
    import torch
    import torchvision

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the Spark driver and Docker GPU access.")
    # Exercise real kernels, including the torchvision extension used by YOLO.
    x = torch.ones((32, 32), device="cuda")
    assert (x @ x).sum().item() == 32768
    boxes = torch.tensor([[0., 0., 1., 1.]], device="cuda")
    torchvision.ops.nms(boxes, torch.ones(1, device="cuda"), 0.5)
    torch.cuda.synchronize()
    print(f"GPU OK: {torch.cuda.get_device_name(0)}; torch={torch.__version__}", flush=True)


def warmup(sample):
    import numpy as np
    import torch
    import rawpy  # Verify native-library imports even when the sample is JPEG.
    import pyexiv2
    from PIL import Image
    from image_filterer.cache_io import FeatureCache
    from image_filterer.config import default_config
    from image_filterer.encoders import build_text_encoder
    from image_filterer.features import FeatureExtractor
    from image_filterer.imaging import open_image
    from image_filterer.ranker import load_model, predict, stack_features
    from image_filterer.config import RankerConfig
    from image_filterer.scene import SceneAnalyzer

    cfg = default_config()
    with tempfile.TemporaryDirectory(prefix="spark-check-") as tmp:
        if sample:
            path = Path(sample)
            if not path.is_file():
                raise FileNotFoundError(path)
            with pyexiv2.Image(str(path)) as exif:
                metadata = exif.read_exif()
            print(f"Sample EXIF readable ({len(metadata)} fields)")
        else:
            path = Path(tmp) / "probe.jpg"
            Image.new("RGB", (640, 480), (100, 120, 140)).save(path)
        # Fresh temporary cache forces inference even if this image was seen before.
        cache = FeatureCache(Path(tmp) / "features")
        extractor = FeatureExtractor(cfg.features, cache)
        try:
            bundles = extractor.extract_paths([path])
        finally:
            extractor.close()
        model, rc, _ = load_model(cfg.model_path, torch.device("cuda"))
        scores = predict(model, stack_features(bundles, RankerConfig(**rc)), torch.device("cuda"))
        if not np.isfinite(scores).all():
            raise RuntimeError("Ranker produced non-finite scores")
        scene = SceneAnalyzer(cfg.scene, cache, device="0")
        try:
            scene.raw(bundles[0].sha1, lambda: open_image(path))
        finally:
            scene.close()
        text = build_text_encoder(cfg.features.context_encoder)
        if not np.isfinite(text.embed_texts(["a person on stage"])).all():
            raise RuntimeError("Text encoder produced non-finite features")
        print(f"Model check passed; face_detected={bundles[0].quality.get('face_detected')}")
        if not sample:
            print("Synthetic probe only: also check a real face photo and each camera's RAW format.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("serve", "setup", "check"))
    parser.add_argument("--sample", help="Real image path inside the container, e.g. /photos/test.CR3")
    args = parser.parse_args()
    for path in ("/data/home", "/data/app", "/data/assets", "/data/huggingface", "/data/ultralytics"):
        Path(path).mkdir(parents=True, exist_ok=True)
    gpu_check()
    if args.action == "setup":
        subprocess.run([sys.executable, "-m", "scripts.fetch_assets"], check=True, cwd="/app")
        warmup(args.sample)
    elif args.action == "check":
        warmup(args.sample)
    else:
        password = os.environ.get("IMAGE_FILTERER_PASSWORD", "")
        if not password or password == "replace-with-your-event-password":
            raise RuntimeError("Set a real IMAGE_FILTERER_PASSWORD before starting the server.")
        from scripts.fetch_assets import ASSETS
        from image_filterer.config import asset_dir
        missing = [name for sub, name, _ in ASSETS if not (asset_dir() / sub / name).is_file()]
        if missing:
            raise RuntimeError(f"Missing assets: {missing}. Run bash start-spark.sh setup first.")
        os.execvp("image-filterer-server", ["image-filterer-server", "--host", "0.0.0.0",
                                           "--port", "8600", "--https"])


if __name__ == "__main__":
    main()
