from __future__ import annotations

import pytest

from bfrb_sensors.data import download as download_module
from bfrb_sensors.data.download import download_data, ensure_prepared_data, ensure_raw_data


class _FakeRepo:
    reproduce_calls: list = []
    pull_calls: list = []
    add_calls: list = []
    push_calls: list = []
    # When set, pull() creates this path to simulate a successful remote fetch.
    pull_materializes: object = None

    def __init__(self, root: str) -> None:
        self.root = root

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def reproduce(self, targets=None, **kwargs):
        _FakeRepo.reproduce_calls.append(targets)

    def pull(self, targets=None, remote=None, **kwargs):
        _FakeRepo.pull_calls.append({"targets": targets, "remote": remote})
        if _FakeRepo.pull_materializes is not None:
            from pathlib import Path as _Path

            _Path(_FakeRepo.pull_materializes).write_text("data")

    def add(self, path, **kwargs):
        _FakeRepo.add_calls.append(path)

    def push(self, targets=None, remote=None, **kwargs):
        _FakeRepo.push_calls.append({"targets": targets, "remote": remote})


@pytest.fixture(autouse=True)
def _reset_calls():
    _FakeRepo.reproduce_calls = []
    _FakeRepo.pull_calls = []
    _FakeRepo.add_calls = []
    _FakeRepo.push_calls = []
    _FakeRepo.pull_materializes = None
    yield


def test_download_data_forwards_targets(tmp_path, monkeypatch):
    monkeypatch.setattr(download_module, "DvcRepo", _FakeRepo)
    download_data(tmp_path, remote="bfrb-data", targets=["data/raw/train.csv"])
    assert _FakeRepo.pull_calls == [{"targets": ["data/raw/train.csv"], "remote": "bfrb-data"}]


def test_download_data_full_pull_when_no_targets(tmp_path, monkeypatch):
    monkeypatch.setattr(download_module, "DvcRepo", _FakeRepo)
    download_data(tmp_path, remote="bfrb-data")
    assert _FakeRepo.pull_calls == [{"targets": None, "remote": "bfrb-data"}]


def _make_prepared(tmp_path, *, index=True, splits=True):
    if index:
        (tmp_path / "index.parquet").write_text("x")
    if splits:
        (tmp_path / "splits.json").write_text("x")


def test_skips_repro_when_all_present(tmp_path, monkeypatch):
    monkeypatch.setattr(download_module, "DvcRepo", _FakeRepo)
    _make_prepared(tmp_path)
    ensure_prepared_data(tmp_path, tmp_path)
    assert _FakeRepo.reproduce_calls == []


def test_runs_repro_when_index_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(download_module, "DvcRepo", _FakeRepo)
    _make_prepared(tmp_path, index=False)
    ensure_prepared_data(tmp_path, tmp_path)
    assert _FakeRepo.reproduce_calls == [["prepare", "splits"]]


def test_runs_repro_when_splits_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(download_module, "DvcRepo", _FakeRepo)
    _make_prepared(tmp_path, splits=False)
    ensure_prepared_data(tmp_path, tmp_path)
    assert _FakeRepo.reproduce_calls == [["prepare", "splits"]]


def test_ensure_raw_noop_when_present(tmp_path, monkeypatch):
    monkeypatch.setattr(download_module, "DvcRepo", _FakeRepo)
    raw = tmp_path / "train.csv"
    raw.write_text("x")
    ensure_raw_data(tmp_path, raw, "comp")
    assert _FakeRepo.pull_calls == []
    assert _FakeRepo.add_calls == []


def test_ensure_raw_pulls_from_remote(tmp_path, monkeypatch):
    monkeypatch.setattr(download_module, "DvcRepo", _FakeRepo)
    raw = tmp_path / "train.csv"
    _FakeRepo.pull_materializes = raw  # remote has it -> pull succeeds
    ensure_raw_data(tmp_path, raw, "comp")
    assert _FakeRepo.pull_calls == [{"targets": [str(raw)], "remote": "bfrb-data"}]
    assert _FakeRepo.add_calls == []  # no Kaggle fallback


def test_ensure_raw_falls_back_to_download_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(download_module, "DvcRepo", _FakeRepo)
    raw = tmp_path / "train.csv"

    calls = {}

    def _fake_fetch(url, raw_dir):
        calls["args"] = (url, raw_dir)
        path = raw_dir / "train.csv"
        path.write_text("downloaded")
        return path

    monkeypatch.setattr("bfrb_sensors.data.fetch_raw.fetch_raw_dataset", _fake_fetch)

    ensure_raw_data(tmp_path, raw, "http://example.test/d.zip")

    assert calls["args"] == ("http://example.test/d.zip", raw.parent)
    assert _FakeRepo.add_calls == [str(raw)]
    assert _FakeRepo.push_calls == [{"targets": [str(raw)], "remote": "bfrb-data"}]
