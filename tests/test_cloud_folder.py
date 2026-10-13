"""Hot folder on a cloud mount (LucidLink) and server-side export back into it."""
import time

from PIL import Image

from image_filterer.export import EXPORT_MARKER, ensure_export_root, export_files
from image_filterer.hotfolder import HotFolderWatcher
from test_regressions import shoot  # noqa: F401 — shared fixture


def _watcher(folder, decodable, **kw):
    kw.setdefault('stable_ticks', 1)
    return HotFolderWatcher(folder, 1, lambda _: True, decodable=decodable, **kw)


def test_file_listed_before_its_bytes_arrive_is_retried_not_dropped(tmp_path):
    # Lucid lists the file at final size immediately; size/mtime never change
    # while the data uploads, so only a later decode can tell it's arrived.
    photo = tmp_path / 'a.jpg'
    photo.write_bytes(b'x' * 100)
    arrived = []
    w = _watcher(tmp_path, lambda p: bool(arrived), scan_interval=0.01)
    w._scan(); w._scan()                       # stable → decode fails
    assert str(photo) not in w._unusable
    assert w.status()['n_waiting_data'] == 1
    arrived.append(True)
    time.sleep(0.02)                           # past the backoff
    w._scan()
    assert str(photo) in w._ready


def test_undecodable_file_backs_off_then_gives_up(tmp_path):
    photo = tmp_path / 'a.jpg'
    photo.write_bytes(b'junk')
    calls = []
    w = _watcher(tmp_path, lambda p: calls.append(p) and False,
                 scan_interval=60, decode_retry_for=3600)
    w._scan(); w._scan(); w._scan(); w._scan()
    assert len(calls) == 1                     # backoff: not re-decoded every scan
    w._pending[str(photo)].first_decode_failure -= 3601
    w._pending[str(photo)].retry_at = 0
    w._scan()
    assert str(photo) in w._unusable and str(photo) not in w._pending


def test_watcher_skips_hidden_files_and_export_folder(tmp_path):
    (tmp_path / 'card').mkdir()
    keep = tmp_path / 'card' / 'keep.jpg'
    keep.write_bytes(b'ok')
    (tmp_path / 'card' / '.keep.jpg.partial').write_bytes(b'ok')
    (tmp_path / '.hidden').mkdir()
    (tmp_path / '.hidden' / 'x.jpg').write_bytes(b'ok')
    exports = ensure_export_root(tmp_path / 'exports')
    (exports / 'shoot').mkdir()
    (exports / 'shoot' / 'pick.jpg').write_bytes(b'ok')
    w = _watcher(tmp_path, lambda p: True)
    w._scan(); w._scan()
    assert w._ready == {str(keep)}


def test_export_is_atomic_skips_identical_and_suffixes_clashes(tmp_path):
    src = tmp_path / 'src'
    (src / 'a').mkdir(parents=True); (src / 'b').mkdir()
    one, two = src / 'a' / 'IMG.jpg', src / 'b' / 'IMG.jpg'
    Image.new('RGB', (8, 8), 'red').save(one)
    Image.new('RGB', (8, 8), 'blue').save(two)
    dest = tmp_path / 'out' / 'shoot'
    r = export_files([one, two], dest)
    assert r['written'] == 2 and r['failed'] == 0
    assert sorted(p.name for p in dest.iterdir()) == ['IMG-2.jpg', 'IMG.jpg']
    r = export_files([one, two], dest)          # re-export → nothing recopied
    assert r['written'] == 0 and r['skipped'] == 2
    r = export_files([tmp_path / 'missing.jpg', one], dest)
    assert r['failed'] == 1 and r['skipped'] == 1
    assert not any(p.name.startswith('.') for p in dest.iterdir())


def test_server_export_endpoint(shoot, tmp_path, monkeypatch):
    c = shoot.client
    assert c.get('/api/state').json['server_export']['enabled'] is False
    assert c.post('/api/export/server', json={'run_id': shoot.id, 'paths': [str(shoot.paths[0])]}).status_code == 400

    root = tmp_path / 'lucid' / 'exports'
    monkeypatch.setenv('IMAGE_FILTERER_EXPORT_DIR', str(root))
    assert c.get('/api/state').json['server_export'] == {'enabled': True, 'name': 'exports', 'path': str(root)}
    outside = tmp_path / 'secret.jpg'
    Image.new('RGB', (8, 8)).save(outside)
    d = c.post('/api/export/server', json={'run_id': shoot.id,
                                          'paths': [str(shoot.paths[0]), str(outside)]}).json
    assert d['total'] == 1                       # unindexed path is refused
    for _ in range(100):
        s = c.get(f"/api/export/server/{d['job']}").json
        if s['done']:
            break
        time.sleep(0.05)
    assert s['result']['written'] == 1 and not s['error']
    assert (root / EXPORT_MARKER).exists()
    assert [p.name for p in (root / 'test').iterdir() if not p.name.startswith('.')] == ['best.jpg']


def test_ordered_map_keeps_order_and_bounds_work_in_flight():
    import threading
    from image_filterer.imaging import ordered_map
    live, peak, lock = [0], [0], threading.Lock()

    def slow(x):
        with lock:
            live[0] += 1; peak[0] = max(peak[0], live[0])
        time.sleep(0.01 * (x % 3))                 # finish out of order
        with lock:
            live[0] -= 1
        return x * 2

    got = []
    for y in ordered_map(slow, range(40), workers=4, lookahead=6):
        got.append(y)
    assert got == [x * 2 for x in range(40)]
    assert peak[0] <= 4
    assert list(ordered_map(lambda x: x + 1, [1, 2], workers=1)) == [2, 3]   # serial path


def test_host_browse_and_export_folder_setting(shoot, tmp_path, monkeypatch):
    c = shoot.client
    monkeypatch.delenv('IMAGE_FILTERER_EXPORT_DIR', raising=False)
    allowed = tmp_path / 'lucid'; (allowed / 'cards' / 'A').mkdir(parents=True)
    Image.new('RGB', (8, 8)).save(allowed / 'cards' / 'x.jpg')
    monkeypatch.setenv('IMAGE_FILTERER_HOST_ROOTS', str(allowed))
    top = c.get('/api/host/browse').json
    assert top['roots'] == [str(allowed)]
    d = c.get('/api/host/browse', query_string={'path': str(allowed / 'cards')}).json
    assert [x['name'] for x in d['dirs']] == ['A'] and d['n_images'] == 1 and d['parent'] == str(allowed)
    assert c.get('/api/host/browse', query_string={'path': str(tmp_path)}).status_code == 400   # outside roots

    assert c.post('/api/host/export_dir', json={'path': str(tmp_path / 'nope')}).status_code == 400
    assert c.post('/api/host/export_dir', json={'path': str(tmp_path)}).status_code == 400      # outside roots
    ok = c.post('/api/host/export_dir', json={'path': str(allowed)}).json
    assert ok['export_dir'] == str(allowed) and ok['export_dir_source'] == 'settings'
    assert c.get('/api/state').json['server_export']['path'] == str(allowed)
    assert not any(p.name.startswith('.image-filterer-write-test') for p in allowed.iterdir())
    cleared = c.post('/api/host/export_dir', json={'path': ''}).json
    assert cleared['export_dir'] == ''


def test_host_one_shot_ingest_validates_path(shoot, tmp_path):
    r = shoot.client.post('/api/ingest/host', json={'path': str(tmp_path / 'missing')})
    assert r.status_code == 400 and 'Not a folder' in r.json['error']


def test_host_new_folder(shoot, tmp_path, monkeypatch):
    import os
    c = shoot.client
    allowed = tmp_path / 'lucid'; allowed.mkdir()
    monkeypatch.setenv('IMAGE_FILTERER_HOST_ROOTS', str(allowed))
    r = c.post('/api/host/mkdir', json={'parent': str(allowed), 'name': 'exports'})
    assert r.status_code == 200 and (allowed / 'exports').is_dir()
    assert c.post('/api/host/mkdir', json={'parent': str(allowed), 'name': 'exports'}).status_code == 409
    for bad in ['', '..', '.hidden', 'a/b', 'a\\b']:
        assert c.post('/api/host/mkdir', json={'parent': str(allowed), 'name': bad}).status_code == 400, bad
    assert c.post('/api/host/mkdir', json={'parent': str(tmp_path), 'name': 'x'}).status_code == 400  # outside roots
    ro = allowed / 'ro'; ro.mkdir(); ro.chmod(0o555)
    try:
        if os.geteuid() != 0:
            r = c.post('/api/host/mkdir', json={'parent': str(ro), 'name': 'x'})
            assert r.status_code == 400 and "Can't create" in r.json['error']
    finally:
        ro.chmod(0o755)



def test_ingest_scan_skips_export_folders_at_any_depth(tmp_path):
    from image_filterer.export import mark_export_dir
    from image_filterer.pipeline import scan_images
    photos = tmp_path / 'GTC'; photos.mkdir()
    Image.new('RGB', (8, 8)).save(photos / 'a.jpg')
    auto = mark_export_dir(photos / 'Auto' / 'Best')           # exports nested inside the photos
    Image.new('RGB', (8, 8)).save(auto / 'copy.jpg')
    (photos / '.hidden.jpg').write_bytes(b'x')
    assert scan_images(photos) == [photos / 'a.jpg']


def test_watcher_hands_over_batches_within_its_decode_budget(tmp_path):
    # On a cloud mount every decode check is a download; a 700-photo dump must
    # not wait for all 700 before anything is ingested.
    for i in range(40):
        (tmp_path / f'{i:03d}.jpg').write_bytes(b'x')
    def slow_ok(_):
        time.sleep(0.02)
        return True
    w = HotFolderWatcher(tmp_path, 1, lambda _: True, stable_ticks=1, decodable=slow_ok,
                         decode_budget=0.05)
    w._scan(); w._scan()                          # first sight, then stable → checks begin
    first = len(w._ready)
    assert 0 < first < 40                          # a partial batch is ready right away
    for _ in range(40):
        if len(w._ready) == 40:
            break
        w._scan()
    assert len(w._ready) == 40 and not w._pending
    assert sorted(w._ready)[0].endswith('000.jpg')  # oldest names first


def test_starting_a_watch_discards_empty_earlier_watch_runs(shoot, tmp_path):
    c = shoot.client
    a = tmp_path / 'a'; a.mkdir(); b = tmp_path / 'b'; b.mkdir()
    first = c.post('/api/hotfolder/start', json={'path': str(a), 'name': 'first'}).json['run_id']
    second = c.post('/api/hotfolder/start', json={'path': str(b), 'name': 'second'}).json['run_id']
    try:
        assert shoot.db.get(first) is None            # never got a photo → removed
        assert shoot.db.get(second) is not None
        assert shoot.db.get(shoot.id) is not None     # real runs untouched
    finally:
        c.post('/api/hotfolder/stop')


def test_watcher_mirrors_network_photos_once(tmp_path):
    src = tmp_path / 'lucid'; (src / 'card').mkdir(parents=True)
    mirror = tmp_path / 'run' / 'mirror'
    photo = src / 'card' / 'a.jpg'
    photo.write_bytes(b'half')                         # bytes still uploading
    checked = []
    def decodable(p):
        checked.append(p)
        return p.read_bytes() == b'good'
    w = HotFolderWatcher(src, 1, lambda _: True, stable_ticks=1, decodable=decodable,
                         mirror_dir=mirror, scan_interval=0.001)
    w._scan(); w._scan()
    assert all(str(p).startswith(str(mirror)) for p in checked)   # checks read the local copy
    assert not (mirror / 'card' / 'a.jpg').exists()                # half copy discarded
    photo.write_bytes(b'good'); os_time = photo.stat().st_mtime_ns
    w._scan(); w._scan(); time.sleep(0.01); w._scan()
    assert w.ingest_paths() == [mirror / 'card' / 'a.jpg']
    assert (mirror / 'card' / 'a.jpg').read_bytes() == b'good'
    assert (mirror / 'card' / 'a.jpg').stat().st_mtime_ns == os_time  # mtime kept


def test_mirror_copy_reuses_an_unchanged_copy(tmp_path):
    from image_filterer.mirror import mirror_copy
    src = tmp_path / 'a.jpg'; src.write_bytes(b'x' * 10)
    dst = tmp_path / 'm' / 'a.jpg'
    mirror_copy(src, dst)
    src.chmod(0o000)                                   # unreadable now: a reuse must not read it
    try:
        assert mirror_copy(src, dst) == dst
    finally:
        src.chmod(0o644)


def test_ingest_runs_in_background_and_new_arrivals_wait_for_next_batch(tmp_path):
    import threading
    gate, seen_batches = threading.Event(), []
    w = None
    def ingest(batch):
        seen_batches.append(([p.name for p in batch], [p.name for p in w.ingest_paths()]))
        gate.wait(5)
        return True
    (tmp_path / 'a.jpg').write_bytes(b'x')
    w = HotFolderWatcher(tmp_path, 1, ingest, stable_ticks=1, decodable=lambda p: True, batch_cooldown=0)
    w._scan(); w._scan()
    w._start_ingest()                                   # returns at once; ingest blocks on the gate
    for _ in range(100):
        if seen_batches: break
        time.sleep(0.01)
    (tmp_path / 'b.jpg').write_bytes(b'x')
    w._scan(); w._scan()                                # scanning carries on during the ingest
    assert str(tmp_path / 'b.jpg') in w._ready
    assert seen_batches[0] == (['a.jpg'], ['a.jpg'])    # the running batch didn't pick b up
    gate.set(); w._ingest_thread.join(5)
    assert w._known == {str(tmp_path / 'a.jpg')} and w._ready == {str(tmp_path / 'b.jpg')}


def test_handled_files_are_not_restatted_every_scan(tmp_path, monkeypatch):
    from pathlib import Path as P
    (tmp_path / 'a.jpg').write_bytes(b'x')
    w = HotFolderWatcher(tmp_path, 1, lambda _: True, stable_ticks=1, decodable=lambda p: True,
                         batch_cooldown=0, full_check_every=3600)
    w._scan(); w._scan(); w._maybe_ingest()
    assert w._known
    calls = []
    real = P.stat
    monkeypatch.setattr(P, 'stat', lambda self, *a, **k: (calls.append(self), real(self, *a, **k))[1])
    w._scan()
    assert not any(c.name == 'a.jpg' for c in calls)


def test_channel_picker_spreads_reads_over_mounts_of_the_same_share(monkeypatch, tmp_path):
    from pathlib import Path as P
    import image_filterer.mirror as m
    table = [["//127.0.0.1/lucid", "/mnt/lucid", "cifs"], ["//127.0.0.1/lucid", "/mnt/lucid-ch1", "cifs"],
             ["//127.0.0.1/lucid", "/mnt/lucid-ch2", "cifs"], ["//other/share", "/mnt/other", "cifs"],
             ["systemd-1", "/mnt/lucid", "autofs"], ["/dev/nvme0n1p2", "/", "ext4"]]
    monkeypatch.setattr(m, "_mount_table", lambda: table)
    monkeypatch.setattr(m.Path, "resolve", lambda self, strict=False: self)
    c = m.ChannelPicker(P("/mnt/lucid/intake"))
    assert c.mount == P("/mnt/lucid") and len(c.roots) == 3
    got = [c.source_for(P("/mnt/lucid/intake/a.jpg")) for _ in range(3)]
    assert sorted(map(str, got)) == ["/mnt/lucid-ch1/intake/a.jpg", "/mnt/lucid-ch2/intake/a.jpg", "/mnt/lucid/intake/a.jpg"]
    local = m.ChannelPicker(P("/home/x/photos"))
    assert local.source_for(P("/home/x/photos/a.jpg")) == P("/home/x/photos/a.jpg")


def test_raw_twins_found_case_insensitively(tmp_path):
    from image_filterer.pipeline import list_photo_files, raw_twins, scan_images
    (tmp_path / 'IMG_1.JPG').write_bytes(b'j'); (tmp_path / 'img_1.cr3').write_bytes(b'r')
    (tmp_path / 'IMG_2.JPG').write_bytes(b'j'); (tmp_path / 'IMG_3.DNG').write_bytes(b'r')
    files = list_photo_files(tmp_path)
    assert raw_twins(files) == {str(tmp_path / 'IMG_1.JPG'): str(tmp_path / 'img_1.cr3')}
    assert sorted(p.name for p in scan_images(tmp_path)) == ['IMG_1.JPG', 'IMG_2.JPG', 'IMG_3.DNG']


def test_raw_is_what_gets_handed_out(shoot, tmp_path, monkeypatch):
    import io, zipfile
    raw = shoot.paths[0].with_suffix('.CR3'); raw.write_bytes(b'RAW-ORIGINAL')
    frame = shoot.frame.assign(raw_path=[str(raw), ''])
    frame.to_csv(shoot.run / 'ranked.csv', index=False)
    (shoot.run / 'version.json').write_text('{"version":4}')          # as a real ingest would
    c, q = shoot.client, {'run_id': shoot.id}
    r = c.get('/download', query_string={**q, 'path': str(shoot.paths[0])})            # single download
    assert r.data == b'RAW-ORIGINAL' and 'best.CR3' in r.headers['Content-Disposition']
    stray = shoot.paths[0].parent / 'other.CR3'; stray.write_bytes(b'x')
    assert c.get('/download', query_string={**q, 'path': str(stray)}).status_code == 403  # not this run's
    tok = c.post('/api/download/prepare', json={**q, 'paths': [str(p) for p in shoot.paths]}).json['token']
    names = zipfile.ZipFile(io.BytesIO(c.get('/api/download/zip', query_string={'token': tok}).data)).namelist()
    assert sorted(names) == ['best.CR3', 'early.jpg']                                  # zip
    root = tmp_path / 'exports'; monkeypatch.setenv('IMAGE_FILTERER_EXPORT_DIR', str(root))
    job = c.post('/api/export/server', json={**q, 'paths': [str(shoot.paths[0])]}).json['job']
    for _ in range(100):
        if c.get(f'/api/export/server/{job}').json['done']: break
        time.sleep(0.05)
    assert [p.name for p in (root / 'test').iterdir() if not p.name.startswith('.')] == ['best.CR3']


def test_auto_export_sends_the_raw_twin():
    import numpy as np, pandas as pd, tempfile
    from pathlib import Path as P
    from image_filterer.auto_export import AutoExportSettings, plan
    d = P(tempfile.mkdtemp()); raw = d / 'a.CR3'; raw.write_bytes(b'r')
    df = pd.DataFrame({'path': [str(d / 'a.jpg')], 'raw_path': [str(raw)], 'score_s1': [1.0], 'burst_id': [0],
                       'captured_at': ['2026-01-01 10:00:00'], 'camera_id': ['A'], 'shot_type': ['wide'],
                       'subject_class': ['people'], 'is_hero': [False]})
    s = AutoExportSettings.from_dict({'enabled': True, 'top_percent': 100, 'speed': 'fast',
                                      'outputs': {'Best': True}})
    e = plan(df, s, {}, {}, now=1e9, final=True).exports[0]
    assert (e.path, e.source, e.dest_name) == (str(d / 'a.jpg'), str(raw), 'a.CR3')


def test_host_roots_can_come_from_local_settings(shoot, tmp_path, monkeypatch):
    import json
    monkeypatch.delenv("IMAGE_FILTERER_HOST_ROOTS", raising=False)
    allowed = tmp_path / "fast-mount"; (allowed / "intake").mkdir(parents=True)
    hidden = tmp_path / "slow-mount"; hidden.mkdir()
    (shoot.cfg.data_root / "host_settings.json").write_text(json.dumps({"host_roots": [str(allowed)]}))
    roots = shoot.client.get("/api/host/browse").json
    listed = json.dumps(roots)
    assert str(allowed) in listed and str(hidden) not in listed
    assert shoot.client.get("/api/host/browse", query_string={"path": str(hidden)}).status_code in (400, 403)


def test_new_photo_is_checked_on_first_sight_without_a_settle_wait(tmp_path):
    photo = tmp_path / 'a.jpg'
    photo.write_bytes(b'x' * 100)
    w = HotFolderWatcher(tmp_path, 1, lambda _: True, decodable=lambda p: True)
    w._scan()
    assert str(photo) in w._ready
    st = photo.stat()
    assert w.prevalidated() == {str(photo): [st.st_size, st.st_mtime_ns]}


def test_raw_waits_one_scan_and_a_growing_file_waits_until_unchanged(tmp_path):
    import os
    calls = []
    w = HotFolderWatcher(tmp_path, 1, lambda _: True, scan_interval=0.0,
                         decodable=lambda p: calls.append(p.name) or p.stat().st_size > 1)
    raw = tmp_path / 'b.cr3'; raw.write_bytes(b'x' * 10)
    w._scan()
    assert calls == []                              # RAW: a half-copied CR3 can pass, so one settle scan
    w._scan()
    assert calls == ['b.cr3']
    part = tmp_path / 'a.jpg'; part.write_bytes(b'x')
    w._scan()
    assert calls[-1] == 'a.jpg' and str(part) not in w._ready     # first sight: checked, half-written
    calls.clear()
    part.write_bytes(b'x' * 50); os.utime(part, ns=(1, 2))        # the copy carries on
    w._scan()
    assert calls == []                              # seen growing: no wasted read
    w._scan()
    assert calls == ['a.jpg'] and str(part) in w._ready           # unchanged for a scan → checked, done


def test_validation_skips_files_the_watcher_already_decoded(tmp_path, monkeypatch):
    from image_filterer import imaging
    a = tmp_path / 'a.jpg'; Image.new('RGB', (8, 8)).save(a)
    b = tmp_path / 'b.jpg'; Image.new('RGB', (8, 8)).save(b)
    run = tmp_path / 'run'; run.mkdir()
    st = a.stat()
    stats = {}
    assert imaging.validated_images([a, b], run, {str(a): [st.st_size, st.st_mtime_ns]}, stats) == [a, b]
    assert stats['decoded'] == 1                    # only b
    stats = {}
    imaging.validated_images([a, b], run, {str(a): [st.st_size + 1, st.st_mtime_ns]}, stats)
    assert stats['decoded'] == 0                    # both remembered from the last ingest
