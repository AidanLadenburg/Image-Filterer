"""Live-event resilience: health banner/alerts, and a watch that survives a restart."""
import json
import time

from image_filterer.health import HealthMonitor, check
from image_filterer.hotfolder import HotFolderWatcher
from image_filterer.server import create_app
from test_regressions import shoot  # noqa: F401 — shared fixture


def test_alert_once_after_grace_and_once_on_recovery():
    current, sent = {}, []
    m = HealthMonitor(lambda: dict(current), grace=30, notify=sent.append)
    current["export-missing"] = "Export folder missing"
    m.tick(now=0); m.tick(now=20)
    assert sent == [] and m.problems() == []          # a blip shorter than the grace stays quiet
    m.tick(now=31); m.tick(now=60)
    assert len(sent) == 1 and "Export folder missing" in sent[0] and m.problems() == ["Export folder missing"]
    current.clear(); m.tick(now=70)
    assert len(sent) == 2 and "resolved" in sent[1] and m.problems() == []


def test_broken_probe_is_itself_a_problem():
    sent = []
    m = HealthMonitor(lambda: 1 / 0, grace=0, notify=sent.append)
    m.tick(now=0)
    assert m.problems() and "Health check failed" in sent[0]


def test_check_finds_unmounted_watch_and_export_folders(tmp_path):
    watched = tmp_path / 'lucid' / 'intake'; watched.mkdir(parents=True)
    w = HotFolderWatcher(watched, 1, lambda _: True, decodable=lambda p: True)
    w._thread = type("T", (), {"is_alive": lambda self: True})()      # running
    assert check(hotfolder=w, export_dir=tmp_path) == {} or set(check(hotfolder=w, export_dir=tmp_path)) <= {"memory", "disk"}
    watched.rmdir()                                                    # Lucid dropped
    probs = check(hotfolder=w, export_dir=tmp_path / 'lucid' / 'export')
    assert "watch-unreadable" in probs and "export-missing" in probs
    w._thread = type("T", (), {"is_alive": lambda self: False})()     # thread died, nobody stopped it
    assert "watch-dead" in check(hotfolder=w)
    w._stop.set()                                                      # stopped on purpose: not a problem
    assert "watch-dead" not in check(hotfolder=w)


def test_idle_stop_is_deliberate_and_reported_once(tmp_path):
    stopped = []
    w = HotFolderWatcher(tmp_path, 1, lambda _: True, scan_interval=0.01, stop_after_idle=0.0,
                         on_stopped=lambda: stopped.append(1))
    w.start(); w._thread.join(timeout=5)
    assert w._stop.is_set() and stopped == [1]


def test_watch_resumes_into_the_same_run_after_a_restart(shoot, tmp_path):
    c = shoot.client
    folder = tmp_path / 'intake'; folder.mkdir()                 # empty: no real ingest kicks off
    rid = c.post('/api/hotfolder/start', json={'path': str(folder), 'name': 'live'}).json['run_id']
    wf = shoot.cfg.data_root / 'active_watch.json'
    assert json.loads(wf.read_text())['run_id'] == rid
    # the run already holds two photos (as its last ingest recorded), then the server dies
    run_dir = shoot.cfg.runs_dir / shoot.db.get(rid)['folder']; run_dir.mkdir(parents=True, exist_ok=True)
    in_run = [str(folder / 'a.jpg'), str(folder / 'b.jpg')]
    (run_dir / 'validated.json').write_text(json.dumps({p: [1, 1] for p in in_run}))
    shoot.db.update(rid, status='ready')
    # "restart": a new app on the same data folder; the old watcher is abandoned, as in a crash
    app2 = create_app(shoot.cfg)
    assert app2.extensions['image_filterer_resume_watch']() == rid
    c2 = app2.test_client()
    st = c2.get('/api/hotfolder').json
    assert st['watching'] and st['run_id'] == rid
    assert "2 photos already in" in shoot.db.get(rid)['message']   # known: not re-added as new
    assert c2.post('/api/hotfolder/stop').json['ok']
    for _ in range(100):
        if not wf.exists():
            break
        time.sleep(0.05)
    assert not wf.exists()                                        # stopped on purpose: never resumed
    assert app2.extensions['image_filterer_resume_watch']() is None
    c.post('/api/hotfolder/stop')
