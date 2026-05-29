from __future__ import annotations

import pytest

from bfrb_sensors.data import download as download_module
from bfrb_sensors.data.download import download_data, ensure_prepared_data


class _FakeRepo:
    reproduce_calls: list = []
    pull_calls: list = []

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


@pytest.fixture(autouse=True)
def _reset_calls():
    _FakeRepo.reproduce_calls = []
    _FakeRepo.pull_calls = []
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
