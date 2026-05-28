"""Verify the Hydra config tree composes without errors."""

from __future__ import annotations

from pathlib import Path

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
    assert cfg.data.datamodule.batch_size == 32
    assert cfg.data.datamodule.artifacts_dir == "artifacts"
