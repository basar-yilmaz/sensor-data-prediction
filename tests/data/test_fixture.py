"""Verify the synthetic CSV fixture produces well-formed data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tests.data.conftest import IMU_COLS, N_TOF, THM_COLS, SyntheticSequenceSpec


def test_synthetic_csv_has_expected_columns(synthetic_raw_csv, default_specs):
    csv_path = synthetic_raw_csv(default_specs)
    df = pd.read_csv(csv_path)

    expected_meta = {
        "sequence_id",
        "subject",
        "gesture",
        "sequence_counter",
        "orientation",
        "sequence_type",
    }
    assert expected_meta.issubset(df.columns)
    for col in IMU_COLS + THM_COLS:
        assert col in df.columns
    tof_cols = [c for c in df.columns if c.startswith("tof_")]
    assert len(tof_cols) == N_TOF


def test_synthetic_csv_respects_missing_modalities(synthetic_raw_csv):
    specs = [
        SyntheticSequenceSpec("s_a", "p_a", "gesture_00", length=10, has_tof=False),
        SyntheticSequenceSpec("s_b", "p_b", "gesture_00", length=10, has_thm=False),
    ]
    csv_path = synthetic_raw_csv(specs)
    df = pd.read_csv(csv_path)

    seq_a = df[df["sequence_id"] == "s_a"]
    seq_b = df[df["sequence_id"] == "s_b"]
    assert seq_a.filter(like="tof_").isna().all().all()
    assert seq_b.filter(like="thm_").isna().all().all()


def test_synthetic_quaternions_are_unit_norm(synthetic_raw_csv):
    spec = SyntheticSequenceSpec("q1", "p01", "gesture_00", length=8)
    csv_path = synthetic_raw_csv([spec])
    df = pd.read_csv(csv_path)
    quat = df[["rot_x", "rot_y", "rot_z", "rot_w"]].to_numpy()
    norms = np.linalg.norm(quat, axis=1)
    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-6)
