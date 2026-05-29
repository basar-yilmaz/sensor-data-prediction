"""Integration tests for the prepare stage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from bfrb_sensors.data.prepare import PrepareConfig, prepare
from tests.data.conftest import SyntheticSequenceSpec


def _run_prepare(raw_csv: Path, tmp_path: Path) -> Path:
    prepared_dir = tmp_path / "prepared"
    cfg = PrepareConfig(
        raw_csv=raw_csv,
        prepared_dir=prepared_dir,
        min_length=5,
        nan_threshold=0.5,
        verbose=False,
    )
    prepare(cfg)
    return prepared_dir


def test_prepare_writes_index_and_sequence_files(synthetic_raw_csv, default_specs, tmp_path):
    raw_csv = synthetic_raw_csv(default_specs)
    prepared_dir = _run_prepare(raw_csv, tmp_path)

    index_path = prepared_dir / "index.parquet"
    assert index_path.exists()
    index = pq.read_table(index_path).to_pandas()

    expected_ids = {spec.sequence_id for spec in default_specs}
    assert set(index["sequence_id"]) == expected_ids

    for seq_id in expected_ids:
        seq_path = prepared_dir / "sequences" / f"{seq_id}.parquet"
        assert seq_path.exists()


def test_prepare_reshapes_tof_correctly(synthetic_raw_csv, tmp_path):
    spec = SyntheticSequenceSpec("only", "p01", "gesture_00", length=12)
    raw_csv = synthetic_raw_csv([spec])
    prepared_dir = _run_prepare(raw_csv, tmp_path)

    seq_table = pq.read_table(prepared_dir / "sequences" / "only.parquet")
    payload = seq_table.to_pydict()
    imu = np.asarray(payload["imu"][0])
    thm = np.asarray(payload["thm"][0])
    tof = np.asarray(payload["tof"][0])

    assert imu.shape == (12, 7)
    assert thm.shape == (12, 5)
    assert tof.shape == (12, 5, 8, 8)


def test_prepare_marks_missing_modalities(synthetic_raw_csv, tmp_path):
    specs = [
        SyntheticSequenceSpec("with_all", "p01", "gesture_00", length=12),
        SyntheticSequenceSpec("no_tof", "p01", "gesture_01", length=12, has_tof=False),
        SyntheticSequenceSpec("no_thm", "p02", "gesture_02", length=12, has_thm=False),
    ]
    raw_csv = synthetic_raw_csv(specs)
    prepared_dir = _run_prepare(raw_csv, tmp_path)
    index = pq.read_table(prepared_dir / "index.parquet").to_pandas()

    by_id = index.set_index("sequence_id")
    assert bool(by_id.loc["with_all", "has_thm"]) is True
    assert bool(by_id.loc["with_all", "has_tof"]) is True
    assert bool(by_id.loc["no_tof", "has_tof"]) is False
    assert bool(by_id.loc["no_tof", "has_thm"]) is True
    assert bool(by_id.loc["no_thm", "has_thm"]) is False
    assert bool(by_id.loc["no_thm", "has_tof"]) is True


def test_prepare_drops_short_sequences(synthetic_raw_csv, tmp_path):
    specs = [
        SyntheticSequenceSpec("ok", "p01", "gesture_00", length=10),
        SyntheticSequenceSpec("too_short", "p01", "gesture_01", length=3),
    ]
    raw_csv = synthetic_raw_csv(specs)
    prepared_dir = _run_prepare(raw_csv, tmp_path)
    index = pq.read_table(prepared_dir / "index.parquet").to_pandas()

    assert "ok" in set(index["sequence_id"])
    assert "too_short" not in set(index["sequence_id"])


def test_prepare_writes_label_encoder(synthetic_raw_csv, default_specs, tmp_path):
    raw_csv = synthetic_raw_csv(default_specs)
    prepared_dir = _run_prepare(raw_csv, tmp_path)
    encoder_path = prepared_dir / "label_encoder.json"
    assert encoder_path.exists()


def test_prepare_index_has_orientation_and_sequence_type(synthetic_raw_csv, tmp_path):
    specs = [
        SyntheticSequenceSpec(
            "s1",
            "p01",
            "gesture_00",
            length=12,
            orientation="Lie on Back",
            sequence_type="Target",
        ),
        SyntheticSequenceSpec(
            "s2",
            "p02",
            "gesture_01",
            length=12,
            orientation="Seated Straight",
            sequence_type="Non-Target",
        ),
    ]
    raw_csv = synthetic_raw_csv(specs)
    prepared_dir = _run_prepare(raw_csv, tmp_path)
    index = pq.read_table(prepared_dir / "index.parquet").to_pandas().set_index("sequence_id")

    assert index.loc["s1", "orientation"] == "Lie on Back"
    assert index.loc["s1", "sequence_type"] == "Target"
    assert index.loc["s2", "orientation"] == "Seated Straight"
    assert index.loc["s2", "sequence_type"] == "Non-Target"


def test_prepare_rejects_wrong_class_count(synthetic_raw_csv, tmp_path):
    import pytest

    specs = [
        SyntheticSequenceSpec("s1", "p01", "gesture_00", length=12),
        SyntheticSequenceSpec("s2", "p01", "gesture_01", length=12),
    ]
    raw_csv = synthetic_raw_csv(specs)
    prepared_dir = tmp_path / "prepared"
    cfg = PrepareConfig(
        raw_csv=raw_csv,
        prepared_dir=prepared_dir,
        min_length=5,
        nan_threshold=0.5,
        expected_n_classes=18,  # fixture only has 2 -> must raise
    )
    with pytest.raises(ValueError, match="expected 18 gesture classes"):
        prepare(cfg)


def test_prepare_writes_imu_derived_and_tof_stats(synthetic_raw_csv, tmp_path):
    spec = SyntheticSequenceSpec("only", "p01", "gesture_00", length=12)
    raw_csv = synthetic_raw_csv([spec])
    prepared_dir = _run_prepare(raw_csv, tmp_path)
    payload = pq.read_table(prepared_dir / "sequences" / "only.parquet").to_pydict()

    imu_derived = np.asarray(payload["imu_derived"][0])
    tof_stats = np.asarray(payload["tof_stats"][0])
    assert imu_derived.shape == (12, 7)
    assert tof_stats.shape == (12, 20)
    assert np.isfinite(imu_derived).all()
    assert np.isfinite(tof_stats).all()


def test_prepare_leaves_raw_modalities_unchanged(synthetic_raw_csv, tmp_path):
    spec = SyntheticSequenceSpec("only", "p01", "gesture_00", length=12)
    raw_csv = synthetic_raw_csv([spec])
    prepared_dir = _run_prepare(raw_csv, tmp_path)
    payload = pq.read_table(prepared_dir / "sequences" / "only.parquet").to_pydict()

    imu = np.asarray(payload["imu"][0])
    thm = np.asarray(payload["thm"][0])
    tof = np.asarray(payload["tof"][0])
    assert imu.shape == (12, 7)
    assert thm.shape == (12, 5)
    assert tof.shape == (12, 5, 8, 8)
