from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from bfrb_sensors.training.train import (
    _datamodule_config,
    _split_artifact_paths,
    _split_hyperparams,
    check_mlflow_server,
    git_state,
    make_checkpoint_callback,
    make_early_stopping_callback,
)


def test_make_early_stopping_callback_monitors_hierarchical_f1():
    cb = make_early_stopping_callback(monitor="val_hierarchical_f1", mode="max", patience=10)
    assert cb.monitor == "val_hierarchical_f1"
    assert cb.mode == "max"
    assert cb.patience == 10


def test_make_checkpoint_callback_monitors_hierarchical_f1(tmp_path: Path):
    cb = make_checkpoint_callback(tmp_path, monitor="val_hierarchical_f1", mode="max")
    assert cb.monitor == "val_hierarchical_f1"
    assert cb.mode == "max"


def test_split_hyperparams_are_prefixed():
    metadata = {
        "version": 1,
        "algorithm": "stratified_train_val_test_split",
        "train_size": 0.8,
        "val_size": 0.1,
        "test_size": 0.1,
        "seed": 42,
        "shuffle": True,
        "stratify_col": "gesture",
        "sequence_count": 360,
        "index_hash": "a" * 64,
    }

    assert _split_hyperparams(metadata) == {
        "split_version": 1,
        "split_algorithm": "stratified_train_val_test_split",
        "split_train_size": 0.8,
        "split_val_size": 0.1,
        "split_test_size": 0.1,
        "split_seed": 42,
        "split_shuffle": True,
        "split_stratify_col": "gesture",
        "split_sequence_count": 360,
        "split_index_hash": "a" * 64,
    }


def test_split_artifact_paths_include_existing_splits_json(tmp_path: Path):
    splits_path = tmp_path / "splits.json"
    splits_path.write_text("{}")
    cfg = {"data": {"datamodule": {"prepared_dir": str(tmp_path)}}}

    assert _split_artifact_paths(cfg) == [splits_path]


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


def test_datamodule_config_skips_raw_tof_when_model_does_not_use_it():
    cfg = OmegaConf.create(
        {
            "data": {
                "datamodule": {
                    "prepared_dir": "data/prepared",
                    "artifacts_dir": "artifacts",
                    "p_thm": 0.5,
                    "p_tof": 0.5,
                    "pin_memory": True,
                    "persistent_workers": True,
                }
            },
            "model": {"use_tof_raw": False},
            "training": {"batch_size": 64, "num_workers": 4},
        }
    )

    assert _datamodule_config(cfg).load_tof_raw is False
