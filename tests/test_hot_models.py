from types import SimpleNamespace


class _FakeExtractor:
    def __init__(self):
        self.calls = 0
        self.closes = 0

    def extract_paths(self, paths, **_kwargs):
        self.calls += 1
        return list(paths)

    def close(self):
        self.closes += 1


def test_external_extractor_stays_hot_across_batches(tmp_path):
    from image_filterer.common import load_or_extract

    cfg = SimpleNamespace(cache_dir=tmp_path, features=object())
    extractor = _FakeExtractor()
    assert load_or_extract(cfg, [tmp_path / "a.jpg"], extractor=extractor)
    assert load_or_extract(cfg, [tmp_path / "b.jpg"], extractor=extractor)
    assert extractor.calls == 2
    assert extractor.closes == 0


def test_owned_extractor_is_still_closed(tmp_path, monkeypatch):
    import image_filterer.common as common

    cfg = SimpleNamespace(cache_dir=tmp_path, features=object())
    extractor = _FakeExtractor()
    monkeypatch.setattr(common, "FeatureExtractor", lambda *_args: extractor)
    assert common.load_or_extract(cfg, [tmp_path / "a.jpg"])
    assert extractor.calls == 1
    assert extractor.closes == 1
