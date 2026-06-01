"""Shared pytest fixtures for the deploy service tests."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_CSV = REPO_ROOT / "deploy" / "sample_data" / "demo_sequence.csv"
TRAIN_CSV = REPO_ROOT / "data" / "raw" / "train.csv"


@pytest.fixture
def sample_csv_bytes() -> bytes:
    if not SAMPLE_CSV.exists():
        pytest.skip(f"sample CSV not found at {SAMPLE_CSV}")
    return SAMPLE_CSV.read_bytes()


@pytest.fixture
def raw_csv_bytes() -> bytes:
    if not TRAIN_CSV.exists():
        pytest.skip(f"raw CSV not found at {TRAIN_CSV}")
    return TRAIN_CSV.read_bytes()


@pytest.fixture
def synthetic_scaler() -> dict[str, np.ndarray]:
    """Tiny scaler with sane mean/std for the 7 IMU + 5 THM channels."""
    return {
        "n_timesteps": 100,
        "imu_mean": np.zeros(7, dtype=np.float32),
        "imu_std": np.ones(7, dtype=np.float32),
        "thm_mean": np.zeros(5, dtype=np.float32),
        "thm_std": np.ones(5, dtype=np.float32),
    }


@pytest.fixture
def tmp_scaler_path(tmp_path, synthetic_scaler) -> Path:
    path = tmp_path / "scaler.joblib"
    joblib.dump(synthetic_scaler, path)
    return path
