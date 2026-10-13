"""Ranker v2, moment grouping, no-face defaults."""
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from image_filterer.bursts import FrameMeta, cluster_bursts


def _frames(times, cam="A"):
    t0 = datetime(2026, 3, 16, 11, 0, 0)
    return [FrameMeta(Path(f"/p/{i}.jpg"), cam, t0 + timedelta(seconds=s)) for i, s in enumerate(times)]


def _emb(vectors):
    return {Path(f"/p/{i}.jpg"): np.asarray(v, np.float32) for i, v in enumerate(vectors)}


def test_similar_frames_after_a_pause_stay_one_moment():
    metas = _frames([0, 0.3, 20])                       # a 20 s pause, same shot
    groups = cluster_bursts(metas, embeddings=_emb([[1, 0]] * 3))
    assert len(groups) == 1


def test_one_odd_frame_does_not_split_a_moment():
    # The odd frame (a turn, a flash) rides in on the short-gap rule; the next
    # frame is compared with the moment's last few frames, not just that one.
    metas = _frames([0, 0.3, 4])
    v = [[1, 0], [0, 1], [1, 0]]
    assert len(cluster_bursts(metas, embeddings=_emb(v), lookback=3)) == 1
    assert len(cluster_bursts(metas, embeddings=_emb(v), lookback=1)) == 2


def test_moments_have_a_maximum_length():
    metas = _frames([0, 20, 40, 60])                    # identical frames every 20 s
    groups = cluster_bursts(metas, embeddings=_emb([[1, 0]] * 4), max_span_seconds=45)
    assert [len(g) for g in groups] == [3, 1]


def test_slow_drift_starts_a_new_moment():
    a = np.array([1.0, 0.0]); b = np.array([np.cos(0.4), np.sin(0.4)]); c = np.array([np.cos(0.8), np.sin(0.8)])
    metas = _frames([0, 5, 10])                         # each step similar (0.92), first vs last not (0.70)
    groups = cluster_bursts(metas, embeddings=_emb([a, b, c]), similarity_threshold=0.91, anchor_threshold=0.89,
                            lookback=1)
    assert len(groups) == 2


def test_no_face_does_not_look_like_wide_open_eyes():
    from image_filterer.features import NO_FACE_EAR, QUALITY_KEYS, _quality_vector
    v = _quality_vector({"face_detected": 0.0, "ear": 1.0})
    assert v[list(QUALITY_KEYS).index("ear")] == NO_FACE_EAR
    seen = _quality_vector({"face_detected": 1.0, "ear": 0.31})
    assert abs(seen[list(QUALITY_KEYS).index("ear")] - 0.31) < 1e-6


def test_v2_ranker_trains_saves_and_scores_identically(tmp_path):
    from image_filterer.config import RankerConfig
    from image_filterer.ranker import RankerV2, feature_blocks, fit_v2, load_any, save_v2
    rng = np.random.default_rng(0)
    n = 120
    y = (np.arange(n) % 4 == 0).astype(int)
    bundles = [SimpleNamespace(emb_full=rng.normal(size=96).astype(np.float32) + y[i],
                               emb_face=rng.normal(size=80).astype(np.float32),
                               quality={"face_detected": 1.0, "ear": 0.3}) for i in range(n)]
    blocks = feature_blocks(bundles)
    cfg = RankerConfig(epochs=5, pairs_per_epoch=256)
    dev = torch.device("cpu")
    model = fit_v2(blocks, y, np.arange(n) // 3, cfg, device=dev, pca=16, seeds=(1, 2), meta={"context_encoder": "x"})
    s = model.score(blocks, dev)
    assert s.shape == (n,) and s[y == 1].mean() > s[y == 0].mean()
    save_v2(tmp_path / "r.pt", model, cfg)
    kind, again = load_any(tmp_path / "r.pt", dev)
    assert kind == "v2" and isinstance(again, RankerV2) and again.meta["context_encoder"] == "x"
    assert np.allclose(again.score(blocks, dev), s, atol=1e-5)


def test_scene_coordinates_are_fractions_of_the_frame(tmp_path, monkeypatch):
    """Detection runs on a shrunken copy; boxes must still be fractions of the
    frame (dividing by the full-size dimensions once made every person tiny)."""
    from image_filterer.cache_io import FeatureCache
    from image_filterer.scene import SceneAnalyzer, SceneConfig, ScenePrep

    class T:
        def __init__(self, a): self.a = np.asarray(a, np.float32)
        def cpu(self): return self
        def numpy(self): return self.a

    det = np.zeros((853, 1280, 3), np.uint8)
    kp = np.zeros((1, 17, 3), np.float32); kp[0, :, 0] = 960; kp[0, :, 1] = 400; kp[0, :, 2] = 0.9
    class Boxes:
        xyxy = T([[640, 0, 1280, 853]]); conf = T([0.9])
        def __len__(self): return 1

    result = SimpleNamespace(boxes=Boxes(), keypoints=SimpleNamespace(data=T(kp)))
    an = SceneAnalyzer(SceneConfig(), FeatureCache(tmp_path))
    monkeypatch.setattr(an, "_yolo", lambda: SimpleNamespace(predict=lambda *a, **k: [result]))
    monkeypatch.setattr(an, "_face", lambda: SimpleNamespace(detect=lambda img: SimpleNamespace(face_blendshapes=None)))
    prep = ScenePrep(np.zeros((2000, 3000, 3), np.uint8), det, np.zeros((160, 240), np.uint8))
    sc = an._measure(prep)
    assert np.allclose(sc.boxes[0, :4], [0.5, 0.0, 1.0, 1.0], atol=1e-3)
    assert np.allclose(sc.kps[0, 0, :2], [0.75, 400 / 853], atol=1e-3)


def test_face_analyzers_are_pooled_not_one_per_thread(tmp_path):
    """Each ingest runs on fresh worker threads; per-thread analyzers in a
    long-lived extractor accumulated ~1 GB per hot-folder re-ingest until the
    server was OOM-killed. The pool must stay at the peak concurrency."""
    import threading
    from image_filterer.cache_io import FeatureCache
    from image_filterer.config import FeatureConfig
    from image_filterer.features import FeatureExtractor
    ex = FeatureExtractor(FeatureConfig(skip_face_detect=True), FeatureCache(tmp_path))
    gate = threading.Barrier(4)

    def work():
        with ex._pooled_face_analyzer():
            gate.wait(timeout=5)              # 4 threads hold one at the same time

    for _ in range(25):                       # 25 "ingests", each on brand-new threads
        ts = [threading.Thread(target=work) for _ in range(4)]
        [t.start() for t in ts]; [t.join() for t in ts]
    assert len(ex._thread_ans) == 4


# ---------------------------------------------------------- in-memory cache

def test_memory_lru_is_capped_by_bytes():
    from image_filterer.cache_io import MemoryLRU
    m = MemoryLRU(1000)
    for i in range(5):
        m.put(i, i, 300)
    assert m.bytes <= 1000 and 0 not in m and 4 in m
    m.get(2); m.put(5, 5, 300)                     # 2 was just used: 3 goes first
    assert 2 in m and 3 not in m
    m.put("huge", 0, 5000)                         # bigger than the cap: not kept
    assert "huge" not in m


def test_feature_cache_serves_from_memory(tmp_path):
    from image_filterer.cache_io import MEMORY, FeatureCache
    fc = FeatureCache(tmp_path)
    fc.save("ab" * 20, "emb_full", {"z": np.arange(4, dtype=np.float32)}, "enc")
    fc.save_meta("ab" * 20, "faceq", {"ear": 0.3})
    for f in tmp_path.rglob("*.*"):
        f.unlink()                                 # disk gone: memory still answers
    assert fc.has("ab" * 20, "emb_full", "enc")
    z = fc.load("ab" * 20, "emb_full", "enc")["z"]
    assert z.tolist() == [0, 1, 2, 3] and not z.flags.writeable
    assert fc.load_meta("ab" * 20, "faceq") == {"ear": 0.3}
    MEMORY.clear()
    assert not fc.has("ab" * 20, "emb_full", "enc") and fc.load_meta("ab" * 20, "faceq") is None


def test_frame_meta_memo_follows_file_changes(tmp_path, monkeypatch):
    import os
    from image_filterer import bursts
    p = tmp_path / "a.cr3"; p.write_bytes(b"x")
    calls = []
    monkeypatch.setattr(bursts, "_raw_exif", lambda _: calls.append(1) or {"DateTimeOriginal": "2026:01:01 00:00:00"})
    assert bursts.read_frame_meta(p).captured_at.year == 2026
    bursts.read_frame_meta(p)
    assert len(calls) == 1
    os.utime(p, ns=(1, 1))                         # touched: read again
    bursts.read_frame_meta(p)
    assert len(calls) == 2


def test_dedup_matches_greedy_rule():
    from types import SimpleNamespace
    from image_filterer.ingest import _dedup
    rng = np.random.default_rng(0)
    base = rng.normal(size=(300, 32)).astype(np.float32)
    embs = np.concatenate([base, base[[5, 5, 17]], base[[299]] + 1e-6])
    order = rng.permutation(len(embs))
    bundles = [SimpleNamespace(sha1=str(i), emb_full=embs[i]) for i in order]
    kept, paths = _dedup(bundles, list(range(len(bundles))), 0.9995)
    # the brute-force rule: keep a frame unless an earlier kept frame is identical
    n = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    want = []
    for i in order:
        if not any(n[i] @ n[j] >= 0.9995 for j in want):
            want.append(i)
    assert [b.sha1 for b in kept] == [str(i) for i in want] and len(want) == 300


def test_sha1_memo_answers_from_memory(tmp_path, monkeypatch):
    from image_filterer import cache_io
    f = tmp_path / "a.jpg"; f.write_bytes(b"abc")
    want = cache_io.file_sha1(f)
    assert cache_io.Sha1Memo(tmp_path / "memo.db").sha1_many([f]) == [want]
    monkeypatch.setattr(cache_io, "file_sha1", lambda p: 1 / 0)       # no re-hashing allowed
    (tmp_path / "memo.db").unlink()                                     # and SQLite gone
    assert cache_io.Sha1Memo(tmp_path / "memo2.db").sha1_many([f]) == [want]
