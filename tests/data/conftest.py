"""Test fixtures for the BFRB data pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# CMI competition channel names. ToF is 5 sensors x 64 pixels = 320 channels.
IMU_COLS = ["acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z"]
THM_COLS = [f"thm_{i}" for i in range(1, 6)]
TOF_COLS = [f"tof_{s}_v{p}" for s in range(1, 6) for p in range(64)]
N_TOF = len(TOF_COLS)  # 320
ALL_GESTURES = [f"gesture_{i:02d}" for i in range(18)]


@dataclass(frozen=True)
class SyntheticSequenceSpec:
    sequence_id: str
    subject_id: str
    gesture: str
    length: int
    has_thm: bool = True
    has_tof: bool = True
    nan_frac_imu: float = 0.0


def _make_sequence(spec: SyntheticSequenceSpec, rng: np.random.Generator) -> pd.DataFrame:
    rows = spec.length
    data: dict[str, np.ndarray] = {}
    data["sequence_id"] = np.array([spec.sequence_id] * rows)
    data["subject_id"] = np.array([spec.subject_id] * rows)
    data["gesture"] = np.array([spec.gesture] * rows)
    data["step"] = np.arange(rows, dtype=np.int64)

    imu = rng.standard_normal((rows, len(IMU_COLS))).astype(np.float64)
    if spec.nan_frac_imu > 0:
        mask = rng.random((rows, len(IMU_COLS))) < spec.nan_frac_imu
        imu[mask] = np.nan
    for col_idx, col in enumerate(IMU_COLS):
        data[col] = imu[:, col_idx]

    thm = rng.standard_normal((rows, len(THM_COLS))).astype(np.float64) * 0.5 + 30.0
    if not spec.has_thm:
        thm[:] = np.nan
    for col_idx, col in enumerate(THM_COLS):
        data[col] = thm[:, col_idx]

    tof = rng.uniform(0, 255, size=(rows, N_TOF)).astype(np.float64)
    if not spec.has_tof:
        tof[:] = np.nan
    for col_idx, col in enumerate(TOF_COLS):
        data[col] = tof[:, col_idx]

    return pd.DataFrame(data)


@pytest.fixture
def synthetic_raw_csv(tmp_path: Path):
    """Returns a builder that writes a CMI-shaped CSV at tmp_path/raw/train.csv."""

    def _build(specs: list[SyntheticSequenceSpec], seed: int = 0) -> Path:
        rng = np.random.default_rng(seed)
        frames = [_make_sequence(spec, rng) for spec in specs]
        df = pd.concat(frames, ignore_index=True)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        csv_path = raw_dir / "train.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    return _build


@pytest.fixture
def default_specs() -> list[SyntheticSequenceSpec]:
    """A small but representative set of sequences covering edge cases."""
    return [
        SyntheticSequenceSpec("s0001", "p01", "gesture_00", length=20),
        SyntheticSequenceSpec("s0002", "p01", "gesture_01", length=15),
        SyntheticSequenceSpec("s0003", "p02", "gesture_02", length=25, has_tof=False),
        SyntheticSequenceSpec("s0004", "p02", "gesture_00", length=18, has_thm=False),
        SyntheticSequenceSpec("s0005", "p03", "gesture_01", length=22, nan_frac_imu=0.1),
    ]
