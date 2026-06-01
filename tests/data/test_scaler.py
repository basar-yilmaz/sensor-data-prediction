"""Tests for the training-split StandardScaler fitter."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from bfrb_sensors.data.scaler import ScalerConfig, fit_scaler, load_scaler


def _write_synthetic_sequence(prepared_dir: Path, seq_id: str, length: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    imu = rng.standard_normal((length, 7)).astype(np.float32) * 2.0 + 1.0
    thm = rng.standard_normal((length, 5)).astype(np.float32) * 0.5 + 30.0
    tof = rng.uniform(0, 255, size=(length, 5, 8, 8)).astype(np.float32)
    table = pa.table(
        {
            "imu": [imu.tolist()],
            "thm": [thm.tolist()],
            "tof": [tof.tolist()],
        }
    )
    sequences_dir = prepared_dir / "sequences"
    sequences_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, sequences_dir / f"{seq_id}.parquet")


def test_fit_scaler_saves_artifact(tmp_path: Path):
    prepared_dir = tmp_path / "prepared"
    artifacts_dir = tmp_path / "artifacts"
    for i, length in enumerate([15, 20, 18]):
        _write_synthetic_sequence(prepared_dir, f"s{i}", length, seed=i)
    train_ids = ["s0", "s1", "s2"]

    cfg = ScalerConfig(prepared_dir=prepared_dir, artifacts_dir=artifacts_dir)
    path = fit_scaler(cfg, train_ids)

    assert path.exists()
    assert path == artifacts_dir / "scaler.joblib"

    loaded = joblib.load(path)
    assert "imu_mean" in loaded
    assert loaded["imu_mean"].shape == (7,)
    assert loaded["imu_std"].shape == (7,)
    assert loaded["thm_mean"].shape == (5,)
    assert loaded["thm_std"].shape == (5,)


def test_fit_scaler_is_data_driven(tmp_path: Path):
    prepared_dir = tmp_path / "prepared"
    artifacts_dir = tmp_path / "artifacts"
    for i in range(4):
        _write_synthetic_sequence(prepared_dir, f"s{i}", length=20, seed=i)

    cfg = ScalerConfig(prepared_dir=prepared_dir, artifacts_dir=artifacts_dir)
    path = fit_scaler(cfg, ["s0", "s1", "s2", "s3"])
    scaler = joblib.load(path)

    # IMU was scaled by 2.0 + offset 1.0 — std should be roughly 2.0
    assert np.all(scaler["imu_std"] > 0.5)
    # THM offset around 30.0
    assert np.all(np.abs(scaler["thm_mean"] - 30.0) < 1.0)


def test_load_scaler_round_trip(tmp_path: Path):
    prepared_dir = tmp_path / "prepared"
    artifacts_dir = tmp_path / "artifacts"
    for i in range(2):
        _write_synthetic_sequence(prepared_dir, f"s{i}", length=20, seed=i)

    cfg = ScalerConfig(prepared_dir=prepared_dir, artifacts_dir=artifacts_dir)
    path = fit_scaler(cfg, ["s0", "s1"])
    scaler = load_scaler(path)

    assert scaler["imu_mean"].shape == (7,)
