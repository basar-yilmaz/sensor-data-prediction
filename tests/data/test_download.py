from __future__ import annotations

import pytest

from bfrb_sensors.data import download as download_module
from bfrb_sensors.data.download import (
    ensure_prepared_data,
    ensure_raw_data,
    pull_dvc_data,
)


class _FakeRepo:
    reproduce_calls: list = []
    pull_calls: list = []
    add_calls: list = []
    push_calls: list = []
    # When set, pull() creates this path to simulate a successful remote fetch.
    pull_materializes: object = None
    pull_materializes_many: list[object] = []
    pull_error: Exception | None = None

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
        if _FakeRepo.pull_error is not None:
            raise _FakeRepo.pull_error
        if _FakeRepo.pull_materializes is not None:
            from pathlib import Path as _Path

            _Path(_FakeRepo.pull_materializes).write_text("data")
        for path in _FakeRepo.pull_materializes_many:
            from pathlib import Path as _Path

            path = _Path(path)
            if path.suffix:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("data")
            else:
                path.mkdir(parents=True, exist_ok=True)

    def add(self, path, **kwargs):
        _FakeRepo.add_calls.append(path)

    def push(self, targets=None, remote=None, **kwargs):
        _FakeRepo.push_calls.append({"targets": targets, "remote": remote})


@pytest.fixture(autouse=True)
def _reset_calls(monkeypatch):
    _FakeRepo.reproduce_calls = []
    _FakeRepo.pull_calls = []
    _FakeRepo.add_calls = []
    _FakeRepo.push_calls = []
    _FakeRepo.pull_materializes = None
    _FakeRepo.pull_materializes_many = []
    _FakeRepo.pull_error = None

    yield


def test_pull_dvc_data_forwards_targets(tmp_path, monkeypatch):
    monkeypatch.setattr(download_module, "DvcRepo", _FakeRepo)
    pull_dvc_data(tmp_path, remote="bfrb-data", targets=["data/raw/train.csv"])
    assert _FakeRepo.pull_calls == [{"targets": ["data/raw/train.csv"], "remote": "bfrb-data"}]


def test_pull_dvc_data_full_pull_when_no_targets(tmp_path, monkeypatch):
    monkeypatch.setattr(download_module, "DvcRepo", _FakeRepo)
    pull_dvc_data(tmp_path, remote="bfrb-data")
    assert _FakeRepo.pull_calls == [{"targets": None, "remote": "bfrb-data"}]


def test_pull_dvc_data_logs_failure_without_traceback(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(download_module, "DvcRepo", _FakeRepo)
    _FakeRepo.pull_error = RuntimeError("missing cache")

    with pytest.raises(RuntimeError, match="missing cache"):
        pull_dvc_data(tmp_path, remote="bfrb-data", targets=["data/raw/train.csv"])

    records = [record for record in caplog.records if record.name == download_module.__name__]
    errors = [record for record in records if record.levelname == "ERROR"]
    assert len(errors) == 1
    assert errors[0].exc_info is None
    assert "Failed to pull from DVC remote" in errors[0].message


def _make_prepared(tmp_path, *, index=True, splits=True, label_encoder=True, sequences=True):
    if index:
        (tmp_path / "index.parquet").write_text("x")
    if splits:
        (tmp_path / "splits.json").write_text("x")
    if label_encoder:
        (tmp_path / "label_encoder.json").write_text("x")
    if sequences:
        (tmp_path / "sequences").mkdir()


def test_skips_repro_when_all_present(tmp_path, monkeypatch):
    monkeypatch.setattr(download_module, "DvcRepo", _FakeRepo)
    _make_prepared(tmp_path)
    ensure_prepared_data(tmp_path, tmp_path)
    assert _FakeRepo.reproduce_calls == []


def test_runs_repro_when_index_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(download_module, "DvcRepo", _FakeRepo)
    _FakeRepo.pull_error = RuntimeError("missing prepared cache")
    _make_prepared(tmp_path, index=False)
    ensure_prepared_data(tmp_path, tmp_path)
    assert _FakeRepo.reproduce_calls == [["prepare", "splits"]]
    assert _FakeRepo.push_calls == [{"targets": ["prepare", "splits"], "remote": "bfrb-data"}]


def test_runs_repro_when_splits_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(download_module, "DvcRepo", _FakeRepo)
    _FakeRepo.pull_error = RuntimeError("missing prepared cache")
    _make_prepared(tmp_path, splits=False)
    ensure_prepared_data(tmp_path, tmp_path)
    assert _FakeRepo.reproduce_calls == [["prepare", "splits"]]
    assert _FakeRepo.push_calls == [{"targets": ["prepare", "splits"], "remote": "bfrb-data"}]


def test_prepared_repro_skips_push_when_explicitly_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(download_module, "DvcRepo", _FakeRepo)
    _FakeRepo.pull_error = RuntimeError("missing prepared cache")
    _make_prepared(tmp_path, splits=False)

    ensure_prepared_data(tmp_path, tmp_path, push_to_dvc=False)

    assert _FakeRepo.reproduce_calls == [["prepare", "splits"]]
    assert _FakeRepo.push_calls == []


def test_pulls_prepared_from_remote_before_repro(tmp_path, monkeypatch):
    monkeypatch.setattr(download_module, "DvcRepo", _FakeRepo)
    _FakeRepo.pull_materializes_many = [
        tmp_path / "index.parquet",
        tmp_path / "splits.json",
        tmp_path / "label_encoder.json",
        tmp_path / "sequences",
    ]

    ensure_prepared_data(tmp_path, tmp_path)

    assert _FakeRepo.pull_calls == [{"targets": ["prepare", "splits"], "remote": "bfrb-data"}]
    assert _FakeRepo.reproduce_calls == []
    assert _FakeRepo.push_calls == []


def test_missing_sequence_dir_triggers_prepared_fetch_or_repro(tmp_path, monkeypatch):
    monkeypatch.setattr(download_module, "DvcRepo", _FakeRepo)
    _FakeRepo.pull_error = RuntimeError("missing prepared cache")
    _make_prepared(tmp_path, sequences=False)

    ensure_prepared_data(tmp_path, tmp_path)

    assert _FakeRepo.reproduce_calls == [["prepare", "splits"]]
    assert _FakeRepo.push_calls == [{"targets": ["prepare", "splits"], "remote": "bfrb-data"}]


def test_ensure_raw_noop_when_present(tmp_path, monkeypatch):
    monkeypatch.setattr(download_module, "DvcRepo", _FakeRepo)
    raw = tmp_path / "train.csv"
    raw.write_text("x")
    ensure_raw_data(tmp_path, raw)
    assert _FakeRepo.pull_calls == []
    assert _FakeRepo.add_calls == []


def test_ensure_raw_pulls_from_remote(tmp_path, monkeypatch):
    monkeypatch.setattr(download_module, "DvcRepo", _FakeRepo)
    raw = tmp_path / "train.csv"
    _FakeRepo.pull_materializes = raw  # remote has it -> pull succeeds
    ensure_raw_data(tmp_path, raw)
    assert _FakeRepo.pull_calls == [{"targets": [str(raw)], "remote": "bfrb-data"}]
    assert _FakeRepo.add_calls == []


def test_ensure_raw_raises_when_minio_sources_cannot_restore_file(tmp_path, monkeypatch):
    monkeypatch.setattr(download_module, "DvcRepo", _FakeRepo)
    _FakeRepo.pull_error = RuntimeError("missing raw cache")
    raw = tmp_path / "train.csv"

    with pytest.raises(FileNotFoundError, match="DVC remote"):
        ensure_raw_data(tmp_path, raw)

    assert _FakeRepo.add_calls == []
    assert _FakeRepo.push_calls == []
