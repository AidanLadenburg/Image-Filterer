"""Smoke tests — no GPU, no model weights, no images.

These cover the wiring that breaks silently: config paths honoring the
environment, the scene classifier's thresholds, and the Flask app starting with
an empty registry. Anything needing real inference is out of scope here.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_config_paths_follow_env(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGE_FILTERER_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("IMAGE_FILTERER_CACHE_DIR", raising=False)

    from image_filterer.config import default_config

    cfg = default_config()
    assert cfg.data_root == tmp_path
    assert cfg.runs_dir == tmp_path / "runs"
    assert cfg.db_path == tmp_path / "registry.db"
    assert cfg.cache_dir == tmp_path / "cache"

    cfg.ensure_dirs()
    assert cfg.runs_dir.is_dir() and cfg.cache_dir.is_dir()


def test_cache_dir_override(tmp_path, monkeypatch):
    """An existing feature cache can be reused from anywhere."""
    monkeypatch.setenv("IMAGE_FILTERER_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("IMAGE_FILTERER_CACHE_DIR", str(tmp_path / "elsewhere" / "cache"))

    from image_filterer.config import default_config

    assert default_config().cache_dir == tmp_path / "elsewhere" / "cache"


def test_production_defaults_are_the_shipped_ones():
    """Guards the four ablation-backed choices against accidental reversion."""
    from image_filterer.config import default_config

    cfg = default_config()
    assert cfg.features.face_detector == "yunet"
    assert cfg.features.use_valence_arousal is False
    assert cfg.features.extract_body is False
    assert cfg.ranker.use_body is False
    assert cfg.ranker.use_va is False
    assert cfg.technical.hard_reject_enabled is False


def test_ranker_feature_flags_match_shipped_checkpoint():
    """The checkpoint stores its own flags; config must not drift from them."""
    import torch

    from image_filterer.config import default_config

    cfg = default_config()
    if not cfg.model_path.exists():
        pytest.skip("no production model checkpoint present")
    ckpt = torch.load(cfg.model_path, map_location="cpu", weights_only=False)
    saved = ckpt["config"]
    for flag in ("use_full", "use_face", "use_body", "use_va", "use_quality"):
        assert saved[flag] == getattr(cfg.ranker, flag), f"{flag} drifted from the checkpoint"


@pytest.mark.parametrize(
    "vec, subject, hero",
    [
        # area, n_prominent, second_area, bg_bright
        ([0.20, 1, 0.00, 0.01], "people", True),    # solo subject, dark background
        ([0.20, 2, 0.05, 0.01], "people", False),   # a second person → not a hero
        ([0.20, 1, 0.00, 0.40], "people", False),   # lit screen behind → not a hero
        ([0.01, 0, 0.00, 0.30], "stage", False),    # empty stage / distant crowd
        ([0.05, 1, 0.00, 0.01], "people", False),   # present but too small for hero
    ],
)
def test_classify_scene(vec, subject, hero):
    from image_filterer.scene import SceneConfig, classify_scene

    got_subject, got_hero = classify_scene(np.array(vec, dtype=np.float32), SceneConfig())
    assert (got_subject, got_hero) == (subject, hero)


def test_app_starts_with_empty_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGE_FILTERER_DATA_ROOT", str(tmp_path))

    from image_filterer.config import default_config
    from image_filterer.server import create_app

    app = create_app(default_config())
    client = app.test_client()

    state = client.get("/api/state")
    assert state.status_code == 200
    assert state.get_json() == {"current": None, "runs": []}

    # No run loaded: the bursts endpoint must answer, not explode.
    assert client.get("/api/bursts").status_code in (200, 400, 409)


def test_image_extensions_cover_common_cases():
    from image_filterer.config import IMAGE_EXTS

    for ext in (".jpg", ".JPG", ".png", ".webp"):
        assert ext in IMAGE_EXTS
