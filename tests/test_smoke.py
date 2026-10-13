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


def test_shipped_checkpoint_matches_runtime_encoders():
    """The checkpoint records which encoders it was trained on; the runtime
    config must produce the same features."""
    import torch

    from image_filterer.config import default_config
    from image_filterer.ranker import load_any

    cfg = default_config()
    if not cfg.model_path.exists():
        pytest.skip("no production model checkpoint present")
    kind, model = load_any(cfg.model_path, torch.device("cpu"))
    meta = model.meta if kind == "v2" else model[2]
    for key in ("context_encoder", "face_encoder"):
        if meta.get(key):
            assert meta[key] == getattr(cfg.features, key)


def _scene(boxes, kps=None, bg_lit=0.0):
    from image_filterer.scene import SceneData, subject
    b = np.zeros((5, 5), np.float32)
    if len(boxes):
        b[:len(boxes)] = boxes
    k = np.zeros((5, 17, 3), np.float32)
    if kps is not None and len(kps):
        k[:len(kps)] = kps
    return SceneData(b, k, subject(b, k), np.zeros(52, np.float32), np.zeros(3, np.float32), False, 0.0, bg_lit)


def _person_kps(x0, y0, x1, y1, visible=range(17)):
    """Keypoints spread down a person box (face at the top, feet at the bottom)."""
    ys = {0: .05, 1: .04, 2: .04, 3: .05, 4: .05, 5: .18, 6: .18, 7: .35, 8: .35, 9: .5, 10: .5,
          11: .55, 12: .55, 13: .75, 14: .75, 15: .97, 16: .97}
    k = np.zeros((17, 3), np.float32)
    for j, f in ys.items():
        k[j] = [(x0 + x1) / 2 + (0.05 if j % 2 else -0.05) * (x1 - x0), y0 + f * (y1 - y0), 0.9 if j in visible else 0.0]
    return k


@pytest.mark.parametrize(
    "boxes, bg, subject, hero",
    [
        ([[0.4, 0.2, 0.6, 0.95, 0.9]], 0.01, "people", True),                               # alone, big, dark
        ([[0.4, 0.2, 0.6, 0.95, 0.9], [0.1, 0.3, 0.25, 0.9, 0.9]], 0.01, "people", False),  # a second person
        ([[0.4, 0.2, 0.6, 0.95, 0.9]], 0.40, "people", False),                              # lit screen around them
        ([[0.48, 0.7, 0.52, 0.85, 0.9]], 0.01, "stage", False),                            # small figure
        ([], 0.0, "stage", False),                                                          # nobody
    ],
)
def test_classify_scene(boxes, bg, subject, hero):
    from image_filterer.scene import SceneConfig, classify_scene

    kps = [_person_kps(*b[:4]) for b in boxes]
    got_subject, got_hero = classify_scene(_scene(boxes, kps, bg), SceneConfig())
    assert (got_subject, got_hero) == (subject, hero)


def test_speaker_choice_skips_slide_figures_and_audience_heads():
    from image_filterer.scene import subject
    speaker = [0.45, 0.3, 0.55, 0.9, 0.9]
    slide_figure = [0.0, 0.05, 0.12, 0.9, 0.9]          # taller, but cut by the frame edge
    audience = [0.6, 0.62, 0.95, 1.0, 0.9]              # low in frame, back of a head
    b = np.zeros((5, 5), np.float32); b[:3] = [speaker, slide_figure, audience]
    k = np.zeros((5, 17, 3), np.float32)
    k[0] = _person_kps(*speaker[:4]); k[1] = _person_kps(*slide_figure[:4])
    k[2] = _person_kps(*audience[:4], visible=[5, 6])   # shoulders only, faint
    k[2, 5:7, 2] = 0.5
    assert subject(b, k) == 0


@pytest.mark.parametrize(
    "box, visible, label",
    [
        ([0.4, 0.1, 0.6, 0.95], range(17), "wide"),                    # feet in frame
        ([0.4, 0.2, 0.6, 1.0], range(15), "medium"),                   # cut above the feet
        ([0.3, 0.0, 0.7, 1.0], range(11), "medium"),                   # cut above the waist, but small head
    ],
)
def test_shot_from_pose(box, visible, label):
    from image_filterer.config import ShotConfig
    from image_filterer.shots import shot_from_pose
    sc = _scene([box + [0.9]], [_person_kps(*box, visible=visible)])
    assert shot_from_pose(sc, ShotConfig())[0] == label


def test_close_needs_a_big_head_not_just_a_tight_crop():
    from image_filterer.config import ShotConfig
    from image_filterer.shots import shot_from_pose
    k = np.zeros((17, 3), np.float32)
    k[0:3] = [[0.5, 0.25, .9], [0.47, 0.22, .9], [0.53, 0.22, .9]]      # face
    k[5:7] = [[0.35, 0.55, .9], [0.65, 0.55, .9]]                      # shoulders: head-unit 0.33
    close = _scene([[0.25, 0.1, 0.75, 1.0, 0.9]], [k])
    assert shot_from_pose(close, ShotConfig())[0] == "close"
    far = k.copy(); far[:, 1] = far[:, 1] * 0.25 + 0.75                 # same pose, small, at the bottom
    angled = _scene([[0.4, 0.72, 0.6, 1.0, 0.9]], [far])
    assert shot_from_pose(angled, ShotConfig())[0] != "close"


def test_app_starts_with_empty_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGE_FILTERER_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("IMAGE_FILTERER_EXPORT_DIR", raising=False)

    from image_filterer.config import default_config
    from image_filterer.server import create_app

    app = create_app(default_config())
    client = app.test_client()

    state = client.get("/api/state")
    assert state.status_code == 200
    assert state.get_json() == {"current": None, "runs": [],
                                "server_export": {"enabled": False, "name": "", "path": ""}}

    # No run loaded: the bursts endpoint must answer, not explode.
    assert client.get("/api/bursts").status_code in (200, 400, 409)


def test_image_extensions_cover_common_cases():
    from image_filterer.config import IMAGE_EXTS

    for ext in (".jpg", ".JPG", ".png", ".webp"):
        assert ext in IMAGE_EXTS
