"""Verify the Hydra config tree composes without errors."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir


def test_config_composes():
    config_dir = (Path(__file__).resolve().parents[2] / "configs").resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config")

    assert cfg.seed == 42
    assert cfg.data.prepare.min_length == 10
    assert cfg.data.prepare.raw_csv == "data/raw/train.csv"
    assert cfg.data.prepare.subject_id_col == "subject"
    assert cfg.data.prepare.step_col == "sequence_counter"
    assert cfg.data.prepare.orientation_col == "orientation"
    assert cfg.data.prepare.sequence_type_col == "sequence_type"
    assert cfg.data.prepare.expected_n_classes == 18
    assert cfg.data.prepare.gravity == 9.81
    assert cfg.data.prepare.sample_hz == 200
    assert float(cfg.data.prepare.quaternion_eps) == 1e-8
    assert cfg.data.prepare.tof_missing_sentinel == -1
    assert cfg.data.splits.n_folds == 5
    assert cfg.data.splits.prepared_dir == "data/prepared"
    assert cfg.data.splits.force is False
    assert cfg.data.datamodule.batch_size == 32
    assert cfg.data.datamodule.artifacts_dir == "artifacts"
    assert cfg.model.name == "baseline_mlp"
    assert cfg.model.input_dim == 39
    assert cfg.training.devices == 1
    assert cfg.training.monitor == "val_hierarchical_f1"
    assert cfg.training.monitor_mode == "max"
    assert cfg.training.class_weighting == "none"
    assert cfg.mlflow.tracking_uri == "http://127.0.0.1:8080"
    assert cfg.data.auto_prepare is True


def test_config_selects_temporal_model():
    config_dir = (Path(__file__).resolve().parents[2] / "configs").resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=["model=temporal_conv_gru"])

    assert cfg.model.name == "temporal_conv_gru"
    assert cfg.model.num_conv_blocks == 2
    assert cfg.model.gru_layers == 1
    assert cfg.model.input_dim == 39
    assert cfg.model.use_demographics is False
    assert cfg.model.meta_embed_dim == 16


def test_config_selects_temporal_tof_model():
    config_dir = (Path(__file__).resolve().parents[2] / "configs").resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=["model=temporal_conv_gru_tof"])

    assert cfg.model.name == "temporal_conv_gru"
    assert cfg.model.use_tof_raw is True
    assert cfg.model.tof_embed_dim == 32


@pytest.mark.parametrize(
    ("experiment", "aux_binary", "aux_weight"),
    [
        ("tof_no_demo", False, 0.0),
        ("tof_no_demo_aux_01", True, 0.1),
        ("tof_no_demo_aux_02", True, 0.2),
        ("tof_no_demo_aux_03", True, 0.3),
        ("tof_no_demo_aux_05", True, 0.5),
    ],
)
def test_config_selects_named_training_experiments(
    experiment: str, aux_binary: bool, aux_weight: float
):
    config_dir = (Path(__file__).resolve().parents[2] / "configs").resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=[f"+experiment={experiment}"])

    assert cfg.model.name == "temporal_conv_gru"
    assert cfg.model.use_tof_raw is True
    assert cfg.model.use_demographics is False
    assert cfg.model.aux_binary is aux_binary
    assert cfg.training.class_weighting == "sqrt_inv_freq"
    assert float(cfg.training.aux_binary_weight) == aux_weight
    assert cfg.training.max_epochs == 30
    assert cfg.training.batch_size == 64
