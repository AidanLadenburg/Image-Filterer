"""Auto-export: selection rules (pure plan()) and the server/worker round trip."""
import json
import time

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from image_filterer.auto_export import (AutoExportSettings, confidence_in_top, plan,
                                        prompt_match_mask, STATE_FILE)
from test_regressions import shoot  # noqa: F401 — shared fixture

ONLY_BEST = {"Best": True, "Hero": False, "Close": False, "Medium": False, "Wide": False,
             "People": False, "Stage": False, "Starred": False}


def frames(n, *, bursts=None, t0="2026-01-01 10:00:00", gap=30, cam="A", **cols):
    """n frames, one per burst unless ``bursts`` given, best score first."""
    t = pd.Timestamp(t0)
    df = pd.DataFrame({
        "path": [f"/p/{i:03d}.jpg" for i in range(n)],
        "score_s1": np.linspace(10, 0, n),
        "burst_id": bursts if bursts is not None else range(n),
        "captured_at": [str(t + pd.Timedelta(seconds=gap * i)) for i in range(n)],
        "camera_id": cam, "shot_type": "wide", "subject_class": "people", "is_hero": False,
    })
    for k, v in cols.items():
        df[k] = v
    return df


def settings(**kw):
    kw.setdefault("outputs", ONLY_BEST)
    kw.setdefault("min_per_folder", 0)
    return AutoExportSettings(enabled=True, **kw)


def test_confidence_matches_beta_tail():
    assert confidence_in_top(1, 10, 0.1) == pytest.approx(1 - 0.9 ** 10)
    assert confidence_in_top(2, 10, 0.1) == pytest.approx(0.2639, abs=1e-4)
    assert confidence_in_top(3, 5000, 0.2) == pytest.approx(1.0)   # no underflow at large n
    assert confidence_in_top(11, 10, 0.5) == 0.0


def test_first_small_batch_exports_nothing_then_only_standouts():
    s = settings(top_percent=10)                         # balanced: 70% sure
    assert plan(frames(5), s, {}, {}, now=1e9, final=True).exports == []
    got = plan(frames(20), s, {}, {}, now=1e9, final=True).exports
    assert [e.path for e in got] == ["/p/000.jpg"]       # rank 1 of 20 is 88% sure; rank 2 only 61%
    got = plan(frames(200), s, {}, {}, now=1e9, final=True).exports
    assert 10 <= len(got) <= 20                          # well inside the top 10% of 200


def test_folder_minimum_fills_small_folders_but_never_with_filler():
    df = frames(200)
    df["is_hero"] = False
    df.loc[[5, 150, 190, 199], "is_hero"] = True          # a rare category: one strong, three weak
    outs = {**ONLY_BEST, "Best": False, "Hero": True}
    pct_only = plan(df, settings(top_percent=10, outputs=outs), {}, {}, now=1e9, final=True).exports
    floored = plan(df, settings(top_percent=10, outputs=outs, min_per_folder=3), {}, {}, now=1e9, final=True).exports
    assert [e.path for e in pct_only] == []                # 10% of 4 moments is nothing confident
    assert [e.path for e in floored] == ["/p/005.jpg"]      # the minimum admits the strong one only


def test_append_only_and_no_repeats():
    s = settings(top_percent=10)
    df = frames(40)
    first = plan(df, s, {}, {}, now=1e9, final=True).exports
    exported = {"Best": {e.path: e.dest_name for e in first}}
    # A better frame arrives; nothing already sent is repeated or withdrawn.
    df2 = pd.concat([df, frames(1, t0="2026-01-01 11:00:00").assign(path="/p/new.jpg", score_s1=99, burst_id=999)])
    again = plan(df2, s, exported, {}, now=1e9, final=True).exports
    assert {e.path for e in again}.isdisjoint(exported["Best"])
    assert "/p/new.jpg" in {e.path for e in again}


def test_unfinished_burst_waits_until_camera_moves_on_or_settle_time():
    s = settings(top_percent=50, settle_seconds=120)
    df = frames(30, gap=60)
    seen = {p: 1000.0 for p in df.path}
    df.loc[29, "score_s1"] = 50                          # the best is the camera's latest burst
    res = plan(df, s, {}, seen, now=1010.0)
    assert "/p/029.jpg" not in {e.path for e in res.exports} and res.waiting["Best"] >= 1
    res = plan(df, s, {}, seen, now=1000.0 + 121)
    assert "/p/029.jpg" in {e.path for e in res.exports}


def test_lookalikes_decide_within_a_burst_too():
    # One rule in and out of a burst: near-identical burst mates are skipped,
    # a burst mate that genuinely differs (pose changed) is still sent.
    # Strictness is a share of photos, so burst-mates compete like any photo.
    df = frames(40, bursts=[0, 0, 0, 0] + list(range(1, 37)), gap=1)
    emb = np.random.default_rng(1).normal(size=(40, 8)).astype(np.float32)
    emb[1] = emb[0] + 1e-3; emb[2] = emb[0] + 2e-3          # frames 1, 2 ≈ frame 0; frame 3 differs
    df.loc[[1, 2], "score_s1"] = df.at[0, "score_s1"] - 0.01  # near ties: never "clearly better"
    pick = lambda variety: [e.path for e in plan(df, settings(top_percent=25, variety=variety), {}, {},
                                                 now=1e9, emb=emb, final=True).exports]
    assert pick("more")[:2] == ["/p/000.jpg", "/p/003.jpg"]
    assert set(pick("off")) >= {"/p/000.jpg", "/p/001.jpg", "/p/002.jpg", "/p/003.jpg"}


def test_strictness_is_a_share_of_photos_not_moments():
    # 10 moments of 10 frames each, all distinct: top 20% of 100 photos ≈ 20
    # photos — a moment-based 20% would have been 2 moments' worth.
    df = frames(100, bursts=np.repeat(range(10), 10), gap=1)
    emb = np.eye(100, dtype=np.float32)
    got = plan(df, settings(top_percent=20, variety="off"), {}, {}, now=1e9, emb=emb, final=True).exports
    assert 12 <= len(got) <= 22


def test_clearly_better_lookalike_still_gets_in():
    # A scene's first photo was exported; a much better near-copy arrives later.
    df = frames(40)
    emb = np.random.default_rng(2).normal(size=(40, 8)).astype(np.float32)
    emb[5] = emb[0] + 1e-3                                    # frame 5 ≈ frame 0 …
    df.loc[5, "score_s1"] = df.at[0, "score_s1"] + 5          # … and far better
    exported = {"Best": {"/p/000.jpg": "000.jpg"}}
    got = [e.path for e in plan(df, settings(top_percent=10, variety="most"), exported, {}, now=1e9,
                                emb=emb, final=True).exports]
    assert "/p/005.jpg" in got
    df.loc[5, "score_s1"] = df.at[0, "score_s1"] + 0.01       # a near tie stays a repeat
    got = [e.path for e in plan(df, settings(top_percent=10, variety="most"), exported, {}, now=1e9,
                                emb=emb, final=True).exports]
    assert "/p/005.jpg" not in got


def test_lookalike_bursts_are_skipped():
    df = frames(40)
    emb = np.random.default_rng(0).normal(size=(40, 8)).astype(np.float32)
    emb[1] = emb[0] + 1e-3                               # burst 1 looks just like burst 0
    s = settings(top_percent=20, variety="more")
    res = plan(df, s, {}, {}, now=1e9, emb=emb, final=True)
    paths = [e.path for e in res.exports]
    assert "/p/000.jpg" in paths and "/p/001.jpg" not in paths and res.lookalikes["Best"] == 1
    off = plan(df, settings(top_percent=20, variety="off"), {}, {}, now=1e9, emb=emb, final=True)
    assert "/p/001.jpg" in [e.path for e in off.exports]


def test_prompt_must_beat_background():
    emb = np.array([[1, 0, 0], [0, 1, 0], [0.6, 0.8, 0]], np.float32)
    bg = np.array([[0, 1, 0]], np.float32)
    assert prompt_match_mask(emb, np.array([1, 0, 0], np.float32), bg, 0.0).tolist() == [True, False, False]
    assert not prompt_match_mask(emb, np.array([0, 0, 1], np.float32), bg, 0.0).any()


def test_prompt_folder_and_starred_folder():
    df = frames(40)
    emb = np.zeros((40, 3), np.float32); emb[:, 1] = 1; emb[:20, 0] = 2      # first 20 are "robots"
    s = AutoExportSettings.from_dict({"enabled": True, "top_percent": 20, "variety": "off",
                                      "outputs": {**ONLY_BEST, "Best": False, "Starred": True},
                                      "prompts": [{"name": "Robots", "text": "robot"}]})
    res = plan(df, s, {}, {}, now=1e9, emb=emb, prompt_vecs={"robot": np.array([1, 0, 0], np.float32)},
               background_vecs=np.array([[0, 1, 0]], np.float32), starred=["/p/039.jpg"], final=True)
    robots = [e.path for e in res.exports if e.output == "Robots"]
    assert robots and all(int(p[3:6]) < 20 for p in robots) and res.matches["Robots"] == 20
    assert [e.path for e in res.exports if e.output == "Starred"] == ["/p/039.jpg"]


def test_settings_are_validated():
    s = AutoExportSettings.from_dict({"top_percent": 900, "per_burst": -3, "speed": "warp",
                                      "prompts": [{"name": "../../etc", "text": "x"}, {"text": ""},
                                                  {"name": "Best", "text": "clash"}]})
    assert s.top_percent == 100 and s.speed == "balanced" and not hasattr(s, "per_burst")
    assert [p.name for p in s.prompts] == ["etc"]         # path characters stripped; "Best" clash dropped


def test_server_round_trip_copies_into_auto_folders(shoot, tmp_path, monkeypatch):
    c = shoot.client
    body = {"run_id": shoot.id, "settings": {"enabled": True, "top_percent": 100, "speed": "fast",
                                             "settle_seconds": 0, "outputs": {**ONLY_BEST, "Starred": True}}}
    assert c.post('/api/auto_export', json=body).status_code == 400      # no export dir yet
    root = tmp_path / 'lucid' / 'exports'
    monkeypatch.setenv('IMAGE_FILTERER_EXPORT_DIR', str(root))
    d = c.post('/api/auto_export', json=body).json
    assert d['settings']['enabled'] and d['export_configured']
    c.post('/api/star', json={'run_id': shoot.id, 'path': str(shoot.paths[1]), 'viewer': 'v'})
    preview = c.post('/api/auto_export/preview', json=body).json
    assert preview['folders']['Starred']['count'] == 1

    status = None
    for _ in range(100):                                  # worker runs in the background
        status = c.get('/api/auto_export', query_string={'run_id': shoot.id}).json['status']
        if status.get('sent', {}).get('Starred') and status.get('sent', {}).get('Best') == 2:
            break
        time.sleep(0.05)
    assert status['sent'] == {'Best': 2, 'Starred': 1}                 # both frames differ → both sent
    best = sorted(f.name for f in (root / 'test' / 'Auto' / 'Best').iterdir() if not f.name.startswith('.'))
    assert best == ['best.jpg', 'early.jpg']                          # original names, unchanged
    assert (root / 'test' / 'Auto' / 'Starred').exists()
    state = json.loads((shoot.run / STATE_FILE).read_text())
    assert set(state['exported']) == {'Best', 'Starred'}


def test_new_watch_inherits_last_saved_setup(shoot, tmp_path, monkeypatch):
    monkeypatch.setenv('IMAGE_FILTERER_EXPORT_DIR', str(tmp_path / 'exports'))
    c = shoot.client
    c.post('/api/auto_export', json={'run_id': shoot.id, 'settings': {
        'enabled': True, 'top_percent': 5, 'prompts': [{'name': 'Robots', 'text': 'a robot'}]}})
    watched = tmp_path / 'cards'; watched.mkdir()
    r = c.post('/api/hotfolder/start', json={'path': str(watched), 'name': 'Live'}).json
    try:
        assert r['auto_export'] is True
        s = c.get('/api/auto_export', query_string={'run_id': r['run_id']}).json['settings']
        assert s['top_percent'] == 5 and s['prompts'][0]['name'] == 'Robots'
    finally:
        c.post('/api/hotfolder/stop')


def test_each_session_competes_only_with_itself():
    evening = frames(60, t0="2026-01-01 19:00:00")                        # scores 10 → 0
    morning = frames(60, t0="2026-01-02 10:00:00").assign(
        path=[f"/q/{i:03d}.jpg" for i in range(60)], burst_id=range(100, 160),
        score_s1=np.linspace(-2, -12, 60))                              # every score lower
    df = pd.concat([evening, morning], ignore_index=True)
    one = plan(df, settings(top_percent=10, session_gap_minutes=0), {}, {}, now=1e9, final=True).exports
    per = plan(df, settings(top_percent=10), {}, {}, now=1e9, final=True).exports
    assert not any(e.path.startswith("/q/") for e in one)                # a shoot-wide % ignores the morning
    assert any(e.path.startswith("/q/") for e in per) and any(e.path.startswith("/p/") for e in per)



def test_shot_list_folder_ranks_by_match_not_ranker():
    # The ranker scores screen/product shots low; a shot-list folder must still
    # take its best *matches* (GTC: "computer chip" exported a rack, blocked chips).
    df = frames(40)
    emb = np.zeros((40, 3), np.float32); emb[:, 1] = 1
    for i, strength in [(38, 3.0), (37, 2.0), (1, 0.3)]:            # 38, 37 = strong matches, low scores
        emb[i, 0] = strength
    s = AutoExportSettings.from_dict({"enabled": True, "top_percent": 10, "min_per_folder": 2, "variety": "off",
                                      "outputs": {**ONLY_BEST, "Best": False},
                                      "prompts": [{"name": "Chips", "text": "chip", "match": "loose"}]})
    res = plan(df, s, {}, {}, now=1e9, emb=emb, prompt_vecs={"chip": np.array([1, 0, 0], np.float32)},
               background_vecs=np.array([[0, 1, 0]], np.float32), final=True)
    assert [e.path for e in res.exports] == ["/p/038.jpg", "/p/037.jpg"]


def test_exports_keep_the_original_file_name():
    from image_filterer.auto_export import dest_name
    assert dest_name({"path": "/cards/A/DN204522.JPG", "captured_at": "2026-01-01 10:00:00"}) == "DN204522.JPG"


def test_auto_export_refuses_to_write_into_the_photo_folder(shoot, monkeypatch):
    # Export folder = the photos' parent and the shoot is named like the photo
    # folder → destination would be inside the photos (the GTC-TPE-imgs loop).
    photos = shoot.paths[0].parent
    shoot.db.update(shoot.id, name=photos.name)
    monkeypatch.setenv('IMAGE_FILTERER_EXPORT_DIR', str(photos.parent))
    c = shoot.client
    body = {"run_id": shoot.id, "settings": {"enabled": True, "top_percent": 100, "speed": "fast",
                                             "settle_seconds": 0, "outputs": {**ONLY_BEST, "Starred": True}}}
    assert c.post('/api/auto_export', json=body).status_code == 200
    st = {}
    for _ in range(100):
        st = c.get('/api/auto_export', query_string={'run_id': shoot.id}).json['status']
        if st.get('last_error'):
            break
        time.sleep(0.05)
    assert "inside this run's photo folder" in st['last_error']
    assert not (photos / 'Auto').exists()
    r = c.post('/api/export/server', json={'run_id': shoot.id, 'paths': [str(shoot.paths[0])]})
    assert r.status_code == 400 and "inside this run's photo folder" in r.json['error']


def test_one_shot_host_run_inherits_saved_setup(shoot, tmp_path, monkeypatch):
    monkeypatch.setenv('IMAGE_FILTERER_EXPORT_DIR', str(tmp_path / 'exports'))
    c = shoot.client
    c.post('/api/auto_export', json={'run_id': shoot.id, 'settings': {'enabled': True, 'top_percent': 5}})
    src = tmp_path / 'card'; src.mkdir()
    Image.new('RGB', (8, 8)).save(src / 'a.jpg')
    r = c.post('/api/ingest/host', json={'path': str(src), 'name': 'once'})
    assert r.status_code == 200 and r.json['auto_export'] is True
    s = c.get('/api/auto_export', query_string={'run_id': r.json['run_id']}).json['settings']
    assert s['enabled'] and s['top_percent'] == 5



def test_shot_list_folder_sends_every_match_regardless_of_quality():
    df = frames(60)                                            # scores 10 → 0
    emb = np.zeros((60, 6), np.float32); emb[:, 1] = 1
    for k, i in enumerate((5, 30, 59)):                        # three distinct matches, one scored lowest
        emb[i, 0] = 2.0; emb[i, 3 + k] = 2.0
    s = AutoExportSettings.from_dict({"enabled": True, "top_percent": 2, "min_per_folder": 0, "variety": "more",
                                      "outputs": {**ONLY_BEST, "Best": False},
                                      "prompts": [{"name": "Robots", "text": "robot", "match": "loose"}]})
    res = plan(df, s, {}, {}, now=1e9, emb=emb, prompt_vecs={"robot": np.array([1, 0, 0, 0, 0, 0], np.float32)},
               background_vecs=np.array([[0, 1, 0, 0, 0, 0]], np.float32), final=True)
    assert sorted(e.path for e in res.exports) == ["/p/005.jpg", "/p/030.jpg", "/p/059.jpg"]
