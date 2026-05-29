from __future__ import annotations

import pytest

from bfrb_sensors.data import download as download_module
from bfrb_sensors.data.download import ensure_prepared_data


class _FakeRepo:
    reproduce_calls: list = []

    def __init__(self, root: str) -> None:
        self.root = root

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def reproduce(self, targets=None, **kwargs):
        _FakeRepo.reproduce_calls.append(targets)


@pytest.fixture(autouse=True)
def _reset_calls():
    _FakeRepo.reproduce_calls = []
    yield


def _make_prepared(tmp_path, *, index=True, splits=True, demographics=True):
    if index:
        (tmp_path / "index.parquet").write_text("x")
    if splits:
        (tmp_path / "splits.json").write_text("x")
    if demographics:
        (tmp_path / "demographics.parquet").write_text("x")


def test_skips_repro_when_all_present(tmp_path, monkeypatch):
    monkeypatch.setattr(download_module, "DvcRepo", _FakeRepo)
    _make_prepared(tmp_path)
    ensure_prepared_data(tmp_path, tmp_path, require_demographics=True)
    assert _FakeRepo.reproduce_calls == []


def test_runs_repro_when_index_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(download_module, "DvcRepo", _FakeRepo)
    _make_prepared(tmp_path, index=False)
    ensure_prepared_data(tmp_path, tmp_path, require_demographics=True)
    assert _FakeRepo.reproduce_calls == [["prepare", "splits"]]


def test_demographics_not_required_when_flag_false(tmp_path, monkeypatch):
    monkeypatch.setattr(download_module, "DvcRepo", _FakeRepo)
    _make_prepared(tmp_path, demographics=False)  # only demographics missing
    ensure_prepared_data(tmp_path, tmp_path, require_demographics=False)
    assert _FakeRepo.reproduce_calls == []  # demographics not required -> no repro
