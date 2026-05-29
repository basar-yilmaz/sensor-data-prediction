from __future__ import annotations

from pathlib import Path

import pytest

from bfrb_sensors.training.train import check_mlflow_server, git_state, make_checkpoint_callback


def test_make_checkpoint_callback_monitors_hierarchical_f1(tmp_path: Path):
    cb = make_checkpoint_callback(tmp_path, monitor="val_hierarchical_f1", mode="max")
    assert cb.monitor == "val_hierarchical_f1"
    assert cb.mode == "max"


def test_check_mlflow_server_raises_clear_error(monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("requests.get", _raise)
    with pytest.raises(RuntimeError, match="docker compose up -d mlflow"):
        check_mlflow_server("http://127.0.0.1:8080")


def test_git_state_returns_sha_and_dirty_flag():
    state = git_state()
    assert "sha" in state
    assert "dirty" in state
