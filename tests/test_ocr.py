"""Text on screen: matching rules, auto-export folders, search order, cache."""
import shutil
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image

from image_filterer.auto_export import AutoExportSettings, ShotPrompt, plan
from image_filterer.cache_io import FeatureCache, Sha1Memo
from image_filterer.config import OcrConfig
from image_filterer.ocr import CACHE_KIND, OcrService
from image_filterer.textmatch import match_scores, query_terms, snippet
from test_auto_export import ONLY_BEST, frames
from test_regressions import shoot  # noqa: F401 — shared fixture


def hit(text, query):
    return float(match_scores([text], query)[0]) > 0


def test_matching_forgives_ocr_slips_but_not_different_words():
    assert hit("NVIDIA Blackweli", "blackwell")          # one wrong letter in a long word
    assert hit("DSXFlex DSX OS", "dsx flex")              # words run together on screen
    assert hit("Full-Stack AI Infrastructure Company", "infrastructure company")
    assert not hit("Dell Pro Max", "bell")                # short words must be exact
    assert not hit("NVIDIA Vera Rubin", "vera blackwell")  # every word must be there
    assert not hit("SIEMENS Energy", "jensen")             # "siemensenergy" holds "mensen": not a word start
    assert hit("Jenson Huang", "jensen")


def test_snippet_points_at_the_match_even_when_misread():
    text = "NVIDIA DSX AI Factory Ecosystem " * 3 + "Grace Blackweli NVL72"
    assert "Blackweli" in snippet(text, "blackwell", width=30)
    slide = "Vera BlueField-4 STX Storage Tray " + "Full-Stack AI Infrastructure Company " * 2 + "Vera Rubin NVL72"
    assert "Vera Rubin" in snippet(slide, "vera rubin", width=30)


def test_short_words_must_stand_alone():
    assert not hit("as she said", "ai")
    assert hit("Full-Stack AI Factory", "ai")
    assert hit("Vera Rubin NVL72", "nvl")                  # letters may touch digits
    assert hit("Groq3 LPX", "groq")


def test_query_terms_drop_filler_and_keep_phrases():
    assert query_terms("the slide that says Vera Rubin") == ["vera", "rubin"]
    assert query_terms('"DSX Flex" on screen') == ["dsxflex"]
    assert match_scores([None, ""], "vera").tolist() == [0.0, 0.0]


def test_text_folder_sends_every_frame_showing_the_words_and_waits_for_unread():
    df = frames(6)
    texts = ["NVIDIA Vera Rubin", "", "Groq 3 LPX", None, "vera rubin nvl72", None]
    s = AutoExportSettings(enabled=True, outputs={**ONLY_BEST, "Best": False}, min_per_folder=0,
                           variety="off", prompts=[ShotPrompt("Rubin", "Vera Rubin", kind="text")])
    res = plan(df, s, {}, {}, now=1e9, final=True, ocr_texts=texts)
    assert sorted(e.path for e in res.exports) == ["/p/000.jpg", "/p/004.jpg"]
    assert res.matches == {"Rubin": 2} and res.unread == {"Rubin": 2}
    assert all(e.reason.startswith("text on screen") for e in res.exports)
    # No text read at all (OCR off) → the folder simply gets nothing.
    assert plan(df, s, {}, {}, now=1e9, final=True).exports == []


def test_text_kind_round_trips_and_defaults_to_describe():
    s = AutoExportSettings.from_dict({"prompts": [{"text": "Vera Rubin", "kind": "text"},
                                                  {"text": "a robot", "kind": "bogus"}]})
    assert [p.kind for p in s.prompts] == ["text", "describe"]


def test_search_puts_frames_showing_the_text_first(shoot, monkeypatch):
    import image_filterer.encoders as enc
    monkeypatch.setattr(enc, "build_text_encoder", lambda ctx: SimpleNamespace(
        name=ctx.strip().lower(), embed_texts=lambda _: np.array([[1.0, 0.0]])))
    monkeypatch.setattr(enc, "context_encoder_has_text_tower", lambda _: True)
    q = {"run_id": shoot.id, "q": "vera rubin", "unfurl": "1"}
    r = shoot.client.get("/api/search", query_string=q).json
    assert [b["rep_filename"] for b in r["bursts"]] == ["best.jpg", "early.jpg"]   # visual order
    ocr = shoot.app.extensions["image_filterer_ocr"]
    ocr._text[str(shoot.paths[1])] = "Vera Rubin NVL72"     # the visually worse frame shows the words
    r = shoot.client.get("/api/search", query_string=q).json
    assert [b["rep_filename"] for b in r["bursts"]] == ["early.jpg", "best.jpg"]
    assert "Vera Rubin" in r["bursts"][0]["text_match"] and r["bursts"][1]["text_match"] == ""
    assert r["text"]["matches"] == 1 and r["text"]["read"] == 1 and r["text"]["total"] == 2


def test_text_is_cached_by_content_and_shared_between_copies(tmp_path):
    a = tmp_path / "a.jpg"
    Image.new("RGB", (16, 16), "red").save(a)
    b = tmp_path / "copy" / "b.jpg"
    b.parent.mkdir()
    shutil.copy2(a, b)
    cache = tmp_path / "cache"
    sha1 = Sha1Memo(cache / "sha1_memo.db").sha1(a)
    FeatureCache(cache).save_meta(sha1, CACHE_KIND, {"lines": [["Vera Rubin", 0.98], ["NVL72", 0.9]]})
    # Everything is already read, so this never needs a GPU worker.
    svc = OcrService(OcrConfig(workers=1), cache, tmp_path / "models", engine="tensorrt")
    svc.activate()
    svc.submit(1, [str(a), str(b)])
    deadline = time.time() + 10
    while None in svc.texts([str(a), str(b)]) and time.time() < deadline:
        time.sleep(0.05)
    assert svc.texts([str(a), str(b)]) == ["Vera Rubin\nNVL72"] * 2
    assert svc.status()["workers"] == 0
    svc.stop()


def test_ocr_can_be_turned_off(tmp_path):
    svc = OcrService(OcrConfig(workers=0), tmp_path / "cache", tmp_path / "models")
    svc.activate()
    svc.submit(1, ["/nope.jpg"])
    assert not svc.enabled and "turned off" in svc.error and svc.texts(["/nope.jpg"]) == [None]


def test_no_new_photos_go_to_the_gpu_while_an_ingest_runs(tmp_path):
    import queue
    from image_filterer.ocr import _Worker
    busy = {"on": True}
    svc = OcrService(OcrConfig(workers=1), tmp_path / "cache", tmp_path / "models",
                     is_busy=lambda: busy["on"], engine="tensorrt")
    w = _Worker(0, proc=None, tasks=queue.Queue())
    w.ready = True
    svc._workers.append(w)
    svc._queue.update({"sha-a": "/a.jpg", "sha-b": "/b.jpg"})
    svc._dispatch()
    assert w.tasks.empty() and svc.status()["paused"]
    busy["on"] = False
    svc._dispatch()
    assert [w.tasks.get_nowait() for _ in range(2)] == [("sha-a", "/a.jpg"), ("sha-b", "/b.jpg")]


def test_highlight_prefers_lines_holding_the_whole_query():
    from image_filterer.textmatch import line_matches
    lines = ["Vera BlueField-4", "Vera Rubin NVL72", "Groq 3 LPX"]
    assert line_matches(lines, "vera rubin") == [False, True, False]
    assert line_matches(["NVIDIA Vera", "Rubin GPU"], "vera rubin") == [True, True]   # split over two lines
    assert line_matches(lines, "") == [False, False, False]


def test_boxes_are_fractions_of_the_photo():
    from image_filterer.ocr import read_lines
    box = np.array([[200, 50], [600, 50], [600, 100], [200, 100]], dtype=np.float32)
    reader = lambda img: SimpleNamespace(txts=("Vera Rubin", "blur"), scores=(0.97, 0.3), boxes=np.stack([box, box]))
    assert read_lines(reader, np.zeros((500, 1000, 3), np.uint8), 0.6) == [
        ["Vera Rubin", 0.97, [[0.2, 0.1], [0.6, 0.1], [0.6, 0.2], [0.2, 0.2]]]]


def test_portrait_photos_are_read_and_shown_upright(shoot, tmp_path):
    from image_filterer.ocr import load_for_ocr
    im = Image.new("RGB", (60, 40), "white")
    exif = im.getexif(); exif[0x0112] = 6            # stored sideways, "rotate 90° clockwise"
    p = tmp_path / "portrait.jpg"
    im.save(p, exif=exif)
    assert load_for_ocr(p, 2048).shape[:2] == (60, 40)          # tall, as the camera meant it
    shoot.paths[0].unlink(); im.save(shoot.paths[0], exif=exif)
    r = shoot.client.get("/thumb", query_string={"run_id": shoot.id, "path": str(shoot.paths[0]), "w": 64})
    import io
    assert Image.open(io.BytesIO(r.data)).size[1] > Image.open(io.BytesIO(r.data)).size[0]


def test_viewer_gets_lines_with_boxes_and_matches(shoot):
    ocr = shoot.app.extensions["image_filterer_ocr"]
    p = str(shoot.paths[1])
    q = {"run_id": shoot.id, "path": p, "q": "vera rubin"}
    assert shoot.client.get("/api/ocr/lines", query_string=q).json["read"] is False
    quad = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.2], [0.1, 0.2]]
    ocr._lines[p] = [["Vera BlueField-4", 0.9, quad], ["Vera Rubin", 0.95, quad]]
    d = shoot.client.get("/api/ocr/lines", query_string=q).json
    assert d["read"] and [(l["text"], l["match"]) for l in d["lines"]] == [("Vera BlueField-4", False), ("Vera Rubin", True)]
    assert d["lines"][1]["box"] == quad
    # Only this run's photos.
    q["path"] = "/etc/passwd"
    assert shoot.client.get("/api/ocr/lines", query_string=q).status_code in (403, 404)


def test_status_says_when_text_is_still_being_read(shoot):
    ocr = shoot.app.extensions["image_filterer_ocr"]
    assert shoot.client.get("/api/active").json["text"] is None
    ocr._queue.update({"sha-a": "/a.jpg"})
    assert shoot.client.get("/api/active").json["text"] == {"left": 1, "paused": False}
