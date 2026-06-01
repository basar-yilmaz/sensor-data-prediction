"""Tests for the Lightning BFRBDataModule."""

from __future__ import annotations

import pickle
from pathlib import Path

import torch

from bfrb_sensors.data.datamodule import BFRBDataModule, DataModuleConfig
from bfrb_sensors.data.prepare import PrepareConfig, prepare
from bfrb_sensors.data.splits import SplitsConfig, make_splits
from tests.data.conftest import SyntheticSequenceSpec


def _prepared_fixture(tmp_path: Path, synthetic_raw_csv) -> tuple[Path, Path]:
    specs = []
    for cls in range(3):
        for i in range(10):
            idx = cls * 10 + i
            specs.append(
                SyntheticSequenceSpec(
                    f"s{idx:04d}", f"p{idx:02d}", f"gesture_{cls:02d}", length=12 + i
                )
            )
    raw_csv = synthetic_raw_csv(specs)
    prepared_dir = tmp_path / "prepared"
    artifacts_dir = tmp_path / "artifacts"

    prepare(PrepareConfig(raw_csv=raw_csv, prepared_dir=prepared_dir, min_length=1))
    make_splits(SplitsConfig(prepared_dir=prepared_dir))
    return prepared_dir, artifacts_dir


def test_datamodule_prepare_setup_and_train_batch(tmp_path: Path, synthetic_raw_csv):
    prepared_dir, artifacts_dir = _prepared_fixture(tmp_path, synthetic_raw_csv)
    dm = BFRBDataModule(
        DataModuleConfig(
            prepared_dir=prepared_dir,
            artifacts_dir=artifacts_dir,
            batch_size=2,
            num_workers=0,
            p_thm=0.0,
            p_tof=0.0,
        )
    )

    dm.prepare_data()
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))

    assert (artifacts_dir / "scaler.joblib").exists()
    assert len(dm.train_dataset) > 0
    assert len(dm.val_dataset) > 0
    assert len(dm.test_dataset) > 0
    test_batch = next(iter(dm.test_dataloader()))
    assert test_batch["label"].shape[0] == test_batch["imu"].shape[0]
    assert batch["imu"].shape[0] == 2
    assert batch["thm"].shape[0] == 2
    assert batch["tof"].shape[0] == 2
    assert batch["attention_mask"].dtype == torch.bool
    assert torch.isfinite(batch["imu"]).all()
    assert torch.isfinite(batch["thm"]).all()
    assert torch.isfinite(batch["tof"]).all()


def test_val_dataloader_does_not_apply_modality_dropout(tmp_path: Path, synthetic_raw_csv):
    prepared_dir, artifacts_dir = _prepared_fixture(tmp_path, synthetic_raw_csv)
    dm = BFRBDataModule(
        DataModuleConfig(
            prepared_dir=prepared_dir,
            artifacts_dir=artifacts_dir,
            batch_size=3,
            num_workers=0,
            p_thm=1.0,
            p_tof=1.0,
        )
    )

    dm.prepare_data()
    dm.setup("fit")
    batch = next(iter(dm.val_dataloader()))

    assert batch["has_thm"].all()
    assert batch["has_tof"].all()
    assert torch.count_nonzero(batch["thm"]) > 0
    assert torch.count_nonzero(batch["tof"]) > 0


def test_datamodule_can_skip_raw_tof_loading(tmp_path: Path, synthetic_raw_csv):
    prepared_dir, artifacts_dir = _prepared_fixture(tmp_path, synthetic_raw_csv)
    dm = BFRBDataModule(
        DataModuleConfig(
            prepared_dir=prepared_dir,
            artifacts_dir=artifacts_dir,
            batch_size=2,
            num_workers=0,
            p_thm=0.0,
            p_tof=1.0,
            load_tof_raw=False,
        )
    )

    dm.prepare_data()
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))

    assert "tof" not in batch
    assert "tof_stats" in batch
    assert not batch["has_tof"].any()
    assert torch.count_nonzero(batch["tof_stats"]) == 0


def test_train_collate_function_is_pickleable(tmp_path: Path, synthetic_raw_csv):
    prepared_dir, artifacts_dir = _prepared_fixture(tmp_path, synthetic_raw_csv)
    dm = BFRBDataModule(
        DataModuleConfig(
            prepared_dir=prepared_dir,
            artifacts_dir=artifacts_dir,
            batch_size=2,
            num_workers=0,
        )
    )

    dm.prepare_data()
    dm.setup("fit")

    pickle.loads(pickle.dumps(dm.train_dataloader().collate_fn))
