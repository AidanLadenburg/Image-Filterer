"""Regression coverage for file boundaries, persistence, and live browsing."""
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from image_filterer.config import Config
from image_filterer.db import RunDB
from image_filterer.pipeline import output_dir, publish_outputs, read_version
from image_filterer.server import create_app


@pytest.fixture
def shoot(tmp_path, monkeypatch):
    monkeypatch.delenv('IMAGE_FILTERER_PASSWORD', raising=False)
    monkeypatch.delenv('IMAGE_FILTERER_CACHE_DIR', raising=False)
    cfg = Config(data_root=tmp_path / 'data')
    cfg.ensure_dirs()
    db = RunDB(cfg.db_path)
    rid = db.create('test', 'shoot')
    run = cfg.runs_dir / 'shoot'
    run.mkdir()
    photos = tmp_path / 'photos'
    photos.mkdir()
    paths = [photos / 'best.jpg', photos / 'early.jpg']
    for i, p in enumerate(paths):
        Image.new('RGB', (32, 32), ('red', 'blue')[i]).save(p)
    frame = pd.DataFrame([
        dict(filename=p.name, path=str(p), burst_id=0, within_burst_rank=i+1,
             is_representative=i == 0, burst_rank=1, score_s1=10-i*10,
             score_s12_after_hard=10-i*10, shot_type=('close', 'wide')[i],
             subject_class='people', is_hero=True,
             captured_at=f'2026-01-01 00:00:0{1-i}')
        for i, p in enumerate(paths)
    ])
    frame.to_csv(run / 'ranked.csv', index=False)
    frame.iloc[:1].assign(burst_size=2).to_csv(run / 'bursts.csv', index=False)
    np.save(run / 'search_index.npy', np.eye(2, dtype=np.float32))
    (run / 'version.json').write_text('{"version":2}')
    db.update(rid, status='ready', n_images=2, n_bursts=1)
    app = create_app(cfg)
    app.testing = True
    return SimpleNamespace(cfg=cfg, db=db, id=rid, run=run, paths=paths,
                           app=app, client=app.test_client(), frame=frame)


def test_upload_rejects_sibling_prefix_escape(shoot):
    c = shoot.client
    rid = c.post('/api/upload/start', data={'name': 'upload'}).json['run_id']
    for name in ('../uploads_escape/probe.txt', '../../probe.txt'):
        response = c.post('/api/upload/chunk', data={
            'run_id': str(rid), 'files': (io.BytesIO(b'probe'), name)},
            content_type='multipart/form-data')
        assert response.status_code == 400
    assert not list(shoot.cfg.runs_dir.rglob('probe.txt'))
    response = c.post('/api/upload/chunk', data={
        'run_id': str(rid), 'files': (io.BytesIO(shoot.paths[0].read_bytes()), 'card/photo.jpg')},
        content_type='multipart/form-data')
    assert response.status_code == 200


def test_only_indexed_files_can_be_served(shoot):
    secret = shoot.paths[0].parent / 'private.txt'
    secret.write_text('private')
    for route in ('/download', '/img', '/thumb'):
        assert shoot.client.get(route, query_string={'run_id': shoot.id, 'path': str(secret)}).status_code == 403
        assert shoot.client.get(route, query_string={'run_id': shoot.id, 'path': str(shoot.paths[0])}).status_code == 200
    # Replacing an indexed filename with a symlink must not expand access.
    shoot.paths[0].unlink()
    shoot.paths[0].symlink_to(secret)
    assert shoot.client.get('/download', query_string={'path': str(shoot.paths[0])}).status_code == 403


def test_deleted_run_revokes_downloads_and_tokens(shoot):
    c = shoot.client
    token = c.post('/api/download/prepare', json={'run_id': shoot.id, 'paths': [str(shoot.paths[0])]}).json['token']
    assert c.post('/api/delete_run', json={'run_id': shoot.id}).status_code == 200
    assert c.get('/download', query_string={'run_id': shoot.id, 'path': str(shoot.paths[0])}).status_code == 404
    assert c.get('/api/download/zip', query_string={'token': token}).status_code == 404
    assert shoot.paths[0].exists()


def test_stars_survive_concurrent_updates_and_reload(shoot):
    def star(i):
        with shoot.app.test_client() as c:
            return c.post('/api/star', json={'run_id': shoot.id, 'path': str(shoot.paths[i % 2]),
                                             'viewer': f'viewer-{i}'}).status_code
    with ThreadPoolExecutor(8) as pool:
        assert list(pool.map(star, range(16))) == [200] * 16
    (shoot.run / 'version.json').write_text('{"version":4}')
    assert star(16) == 200
    saved = json.loads((shoot.run / 'stars.json').read_text())['stars']
    assert sum(map(len, saved.values())) == 17
    shoot.client.post('/api/stars/clear', json={'run_id': shoot.id, 'viewer': 'viewer-0'})
    saved = json.loads((shoot.run / 'stars.json').read_text())['stars']
    assert sum(map(len, saved.values())) == 16


def test_failed_star_write_does_not_change_memory(tmp_path, monkeypatch):
    from image_filterer.stars import StarStore
    store = StarStore(tmp_path)
    store.set('a', 'alice', True)
    def fail(*args):
        raise OSError('disk full')
    monkeypatch.setattr('image_filterer.stars.atomic_write', fail)
    with pytest.raises(OSError):
        store.set('b', 'bob', True)
    assert store.stars == {'a': {'alice'}}


def test_password_rotation_and_unicode(shoot, monkeypatch):
    monkeypatch.setenv('IMAGE_FILTERER_PASSWORD', 'old-pässword')
    c = shoot.client
    assert c.get('/api/state').status_code == 401
    assert c.post('/login', json={'password': 'old-pässword'}).status_code == 200
    assert c.get('/api/state').status_code == 200
    monkeypatch.setenv('IMAGE_FILTERER_PASSWORD', 'new-password')
    assert c.get('/api/state').status_code == 401


@pytest.mark.parametrize('sort', ['time', 'stars', 'rank'])
def test_sorted_tiles_obey_filters(shoot, sort):
    shoot.client.post('/api/star', json={'path': str(shoot.paths[1]), 'viewer': 'alice'})
    d = shoot.client.get('/api/bursts', query_string={
        'sort': sort, 'top_pct': 50, 'shot': 'close'}).json
    assert [r['rep_path'] for r in d['bursts']] == [str(shoot.paths[0])]


def test_filters_find_eligible_nonrepresentative(shoot):
    d = shoot.client.get('/api/bursts?shot=wide').json
    assert [r['rep_path'] for r in d['bursts']] == [str(shoot.paths[1])]


@pytest.mark.parametrize('route', ['/api/bursts', '/api/starred', '/api/search?q=test', '/api/burst/0'])
def test_old_generation_is_rejected(shoot, route):
    sep = '&' if '?' in route else '?'
    d = shoot.client.get(route + sep + 'version=0').json
    assert d['stale'] is True
    assert d['version'] == 2


def test_search_filters_before_grouping(shoot, monkeypatch):
    encoder = SimpleNamespace(name='siglip2_so400m_14', embed_texts=lambda _: np.array([[1., 0.]]))
    monkeypatch.setattr('image_filterer.encoders.build_text_encoder', lambda _: encoder)
    d = shoot.client.get('/api/search?q=test&shot=wide').json
    assert [r['rep_path'] for r in d['bursts']] == [str(shoot.paths[1])]


def test_publication_failure_preserves_last_generation(tmp_path):
    def write(stage):
        (stage / 'ranked.csv').write_text('complete')
        (stage / 'bursts.csv').write_text('complete')
    assert publish_outputs(tmp_path, write) == 2
    previous = output_dir(tmp_path)
    def fail(stage):
        (stage / 'ranked.csv').write_text('partial')
        raise OSError('disk full')
    with pytest.raises(OSError):
        publish_outputs(tmp_path, fail)
    assert read_version(tmp_path) == 2
    assert output_dir(tmp_path) == previous
    assert (tmp_path / 'ranked.csv').read_text() == 'complete'
    assert publish_outputs(tmp_path, write) == 4
    assert previous.exists()


def test_watcher_recurses_and_rechecks_repaired_files(tmp_path):
    from image_filterer.hotfolder import HotFolderWatcher
    nested = tmp_path / 'card'
    nested.mkdir()
    photo = nested / 'photo.JpG'
    photo.write_bytes(b'bad')
    watcher = HotFolderWatcher(tmp_path, 1, lambda _: True, stable_ticks=1,
                               decodable=lambda p: p.read_bytes() == b'good')
    watcher._scan(); watcher._scan()
    assert str(photo) in watcher._unusable
    photo.write_bytes(b'good')
    watcher._scan(); watcher._scan()
    assert str(photo) in watcher._ready
    assert str(photo) not in watcher._unusable


def test_watcher_retries_failed_batch(tmp_path):
    from image_filterer.hotfolder import HotFolderWatcher
    attempts = []
    def ingest(paths):
        attempts.append(paths)
        if len(attempts) == 1:
            raise OSError('temporary')
        return True
    watcher = HotFolderWatcher(tmp_path, 1, ingest, batch_cooldown=0)
    watcher._ready.add(str(tmp_path / 'a.jpg'))
    with pytest.raises(OSError):
        watcher._maybe_ingest()
    assert watcher._ready and not watcher._known
    watcher._maybe_ingest()
    assert len(attempts) == 2 and watcher._known and not watcher._ready


def test_truncated_jpeg_is_rejected_and_unchanged_valid_file_is_cached(tmp_path, monkeypatch):
    from image_filterer.imaging import is_decodable, validated_images
    good = tmp_path / 'good.jpg'; bad = tmp_path / 'bad.jpg'
    Image.new('RGB', (32, 32), 'red').save(good)
    bad.write_bytes(good.read_bytes()[:-10])
    assert not is_decodable(bad)
    assert validated_images([good, bad], tmp_path) == [good]
    monkeypatch.setattr('image_filterer.imaging.is_decodable', lambda _: pytest.fail('decoded cached image'))
    assert validated_images([good], tmp_path) == [good]


@pytest.mark.parametrize('fraction,micros', [('001', 1000), ('1', 100000), ('01', 10000), ('123456', 123456)])
def test_exif_subseconds(tmp_path, monkeypatch, fraction, micros):
    from image_filterer.bursts import read_frame_meta
    monkeypatch.setattr('image_filterer.bursts._raw_exif', lambda _: {
        'DateTimeOriginal': '2026:01:01 00:00:00', 'SubsecTimeOriginal': fraction})
    assert read_frame_meta(tmp_path / 'frame.cr3').captured_at.microsecond == micros


def test_cross_validation_is_a_partition_and_auc_handles_ties():
    from image_filterer.ranker import stratified_kfold, auc_mannwhitney
    y = np.array([0]*20 + [1]*20)
    folds = stratified_kfold(y, 5, 42)
    assert np.array_equal(np.sort(np.concatenate([v for _, v in folds])), np.arange(40))
    for train, valid in folds:
        assert not set(train) & set(valid)
    assert auc_mannwhitney(np.array([1, 0]), np.array([0., 0.])) == .5


def test_block_dedup_matches_greedy_reference():
    from image_filterer.ingest import _dedup
    rng = np.random.default_rng(42)
    emb = rng.normal(size=(600, 8)).astype(np.float32)
    emb[300] = emb[0]
    normalized = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    threshold = .85
    keep = []
    for i, e in enumerate(normalized):
        if not any(e @ normalized[j] >= threshold for j in keep):
            keep.append(i)
    bundles = [SimpleNamespace(sha1=str(i), emb_full=e) for i, e in enumerate(emb)]
    _, paths = _dedup(bundles, list(range(len(emb))), threshold)
    assert paths == keep


def test_cached_features_do_not_load_models(tmp_path, monkeypatch):
    from image_filterer.cache_io import FeatureCache, file_sha1
    from image_filterer.features import FeatureExtractor
    from image_filterer.config import FeatureConfig
    photo = tmp_path / 'a.jpg'; photo.write_bytes(b'cached')
    sha = file_sha1(photo); cache = FeatureCache(tmp_path / 'cache')
    ex = FeatureExtractor(FeatureConfig(), cache)
    cache.save(sha, 'emb_full', {'z': np.ones(1152)}, ex._ctx_id_lazy())
    cache.save(sha, 'emb_face', {'z': np.ones(512)}, ex._face_id_lazy())
    cache.save_meta(sha, ex._face_meta_kind, {'face_detected': 0.})
    monkeypatch.setattr(ex, '_ctx_encoder', lambda: pytest.fail('context model loaded'))
    monkeypatch.setattr(ex, '_face_encoder', lambda: pytest.fail('face model loaded'))
    assert len(ex.extract_paths([photo])) == 1


def test_measurement_config_changes_cache_namespaces(tmp_path):
    from image_filterer.cache_io import FeatureCache
    from image_filterer.config import FeatureConfig
    from image_filterer.features import FeatureExtractor
    from image_filterer.scene import SceneAnalyzer, SceneConfig
    cache = FeatureCache(tmp_path)
    default = FeatureExtractor(FeatureConfig(), cache)
    changed = FeatureExtractor(FeatureConfig(face_expand=2.), cache)
    assert default._face_meta_kind != changed._face_meta_kind
    baseline = SceneAnalyzer(SceneConfig(), cache)
    assert baseline.cache_id != SceneAnalyzer(SceneConfig(bg_luma=100), cache).cache_id
    assert baseline.cache_id == SceneAnalyzer(SceneConfig(hero_min_area=.1), cache).cache_id


def test_ingestion_publication_and_reload_end_to_end(tmp_path, monkeypatch):
    from dataclasses import asdict
    from image_filterer.features import FeatureBundle
    from image_filterer.ingest import ingest_folder
    from image_filterer.cache_io import file_sha1
    monkeypatch.delenv('IMAGE_FILTERER_PASSWORD', raising=False)
    cfg = Config(data_root=tmp_path / 'data')
    cfg.scene.enabled = False
    cfg.shots.enabled = False
    cfg.ensure_dirs()
    db = RunDB(cfg.db_path)
    rid = db.create('integration', 'integration')
    run = cfg.runs_dir / 'integration'
    photos = tmp_path / 'photos'; photos.mkdir()
    a = photos / 'a.jpg'; bad = photos / 'broken.jpg'
    Image.new('RGB', (32, 32), 'red').save(a)
    bad.write_bytes(a.read_bytes()[:-10])
    def extract(cfg, paths, **kwargs):
        return [FeatureBundle(p, file_sha1(p), np.eye(len(paths), dtype=np.float32)[i],
                              np.ones(2), np.zeros(0), np.zeros(2), {}) for i, p in enumerate(paths)]
    monkeypatch.setattr('image_filterer.ingest.load_or_extract', extract)
    monkeypatch.setattr('image_filterer.ingest.load_model', lambda *a: (None, asdict(cfg.ranker), {}))
    monkeypatch.setattr('image_filterer.ingest.predict', lambda model, x, device: np.arange(len(x), dtype=np.float32))
    result = ingest_folder(photos, cfg, run)
    assert result == {'n_images': 1, 'n_bursts': 1, 'n_skipped': 1}
    db.update(rid, status='ready', n_images=1, n_bursts=1)
    app = create_app(cfg)
    # Existing shot labels are absent; don't invoke a text model in this test.
    monkeypatch.setattr('image_filterer.encoders.context_encoder_has_text_tower', lambda _: False)
    client = app.test_client()
    assert client.get('/api/bursts').json['version'] == 2
    client.post('/api/star', json={'path': str(a), 'viewer': 'alice'})
    b = photos / 'b.jpg'; Image.new('RGB', (32, 32), 'blue').save(b)
    def fail(*args):
        raise OSError('disk full during publication')
    original = __import__('image_filterer.ingest', fromlist=['write_bursts_csv']).write_bursts_csv
    monkeypatch.setattr('image_filterer.ingest.write_bursts_csv', fail)
    with pytest.raises(OSError):
        ingest_folder(photos, cfg, run)
    assert client.get('/api/bursts').json['version'] == 2
    monkeypatch.setattr('image_filterer.ingest.write_bursts_csv', original)
    result = ingest_folder(photos, cfg, run)
    db.update(rid, n_images=result['n_images'], n_bursts=result['n_bursts'])
    d = client.get('/api/bursts').json
    assert d['version'] == 4 and len(d['bursts']) == 2
    assert client.get('/api/stars?viewer=alice').json['mine'] == [str(a)]
    assert json.loads((run / 'config.json').read_text())['shots']['enabled'] is False


def test_shot_axis_cache_avoids_model_reload(tmp_path, monkeypatch):
    from image_filterer.cache_io import FeatureCache
    from image_filterer.shots import compute_shot_types
    from image_filterer.config import ShotConfig
    encoder = SimpleNamespace(embed_texts=lambda texts: np.ones((len(texts), 2), dtype=np.float32))
    cache = FeatureCache(tmp_path)
    compute_shot_types(np.ones((1, 2)), 'siglip2_so400m_14', ShotConfig(), text_encoder=encoder, cache=cache)
    monkeypatch.setattr('image_filterer.shots.build_text_encoder', lambda _: pytest.fail('model loaded'))
    labels, _ = compute_shot_types(np.ones((1, 2)), 'siglip2_so400m_14', ShotConfig(), cache=cache)
    assert len(labels) == 1


def test_hotfolder_server_failure_keeps_batch_and_cancel_stops_watch(shoot, monkeypatch):
    from image_filterer.hotfolder import HotFolderWatcher
    def closure(fn, key):
        return dict(zip(fn.__code__.co_freevars, (v.cell_contents for v in fn.__closure__)))[key]
    hot = closure(shoot.app.view_functions['api_hotfolder_start'], '_hot_ingest')
    state = closure(hot, 'state')
    watcher = HotFolderWatcher(shoot.paths[0].parent, shoot.id, lambda batch: hot(batch, watcher), batch_cooldown=0)
    watcher._ready.add(str(shoot.paths[0]))
    state['cancel_event'].set()  # stale cancellation from an unrelated ingest
    def fail(*args, **kwargs):
        assert not state['cancel_event'].is_set()
        raise OSError('temporary')
    monkeypatch.setattr('image_filterer.server.ingest_folder', fail)
    with pytest.raises(OSError):
        watcher._maybe_ingest()
    assert watcher._ready and not watcher._known
    def cancel(*args, **kwargs):
        state['cancel_event'].set()
        kwargs['progress_cb'](5, 'cancel now')
    monkeypatch.setattr('image_filterer.server.ingest_folder', cancel)
    watcher._maybe_ingest()
    assert watcher._stop.is_set() and watcher._ready and not watcher._known


def test_delete_with_empty_registry_folder_cannot_remove_other_runs(shoot):
    rid = shoot.db.create('incomplete', '')
    assert shoot.client.post('/api/delete_run', json={'run_id': rid}).status_code == 400
    assert (shoot.run / 'ranked.csv').exists()


def test_queued_watch_is_selectable(shoot):
    rid = shoot.db.create('watch', 'watch')
    r = shoot.client.post('/api/select_run', json={'run_id': rid})
    assert r.status_code == 202 and r.json['pending']

