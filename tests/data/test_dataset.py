"""Tests for BFRBDataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from bfrb_sensors.data.dataset import BFRBDataset
from bfrb_sensors.data.label_encoder import build_label_encoder


def _write_seq(
    prepared_dir: Path,
    seq_id: str,
    length: int,
    gesture: str,
    has_thm: bool = True,
    has_tof: bool = True,
    seed: int = 0,
    include_derived: bool = True,
    tof_stats_width: int = 20,
) -> None:
    rng = np.random.default_rng(seed)
    imu = rng.standard_normal((length, 7)).astype(np.float32)
    imu_derived = rng.standard_normal((length, 7)).astype(np.float32)
    thm = (rng.standard_normal((length, 5)) * 0.5 + 30.0).astype(np.float32)
    tof = rng.uniform(0, 255, size=(length, 5, 8, 8)).astype(np.float32)
    tof_stats = rng.standard_normal((length, tof_stats_width)).astype(np.float32)
    if not has_thm:
        thm[:] = 0.0
    if not has_tof:
        tof[:] = 0.0

    sequences_dir = prepared_dir / "sequences"
    sequences_dir.mkdir(parents=True, exist_ok=True)
    columns = {
        "imu": [imu.tolist()],
        "thm": [thm.tolist()],
        "tof": [tof.tolist()],
    }
    if include_derived:
        columns["imu_derived"] = [imu_derived.tolist()]
        columns["tof_stats"] = [tof_stats.tolist()]
    table = pa.table(columns)
    pq.write_table(table, sequences_dir / f"{seq_id}.parquet")


def _make_index(prepared_dir: Path, rows: list[dict]) -> None:
    import pandas as pd

    pd.DataFrame(rows).to_parquet(prepared_dir / "index.parquet", index=False)


def _identity_scaler() -> dict:
    return {
        "fold_idx": 0,
        "n_timesteps": 0,
        "imu_mean": np.zeros(7, dtype=np.float32),
        "imu_std": np.ones(7, dtype=np.float32),
        "thm_mean": np.zeros(5, dtype=np.float32),
        "thm_std": np.ones(5, dtype=np.float32),
    }


def test_dataset_returns_expected_keys_and_shapes(tmp_path: Path):
    prepared_dir = tmp_path / "prepared"
    _write_seq(prepared_dir, "s0", length=12, gesture="g_a")
    _make_index(
        prepared_dir,
        [
            {
                "sequence_id": "s0",
                "subject_id": "p1",
                "gesture": "g_a",
                "length": 12,
                "has_thm": True,
                "has_tof": True,
                "frac_nan_imu": 0.0,
                "frac_nan_thm": 0.0,
                "frac_nan_tof": 0.0,
            },
        ],
    )
    encoder = build_label_encoder(["g_a", "g_b"])

    ds = BFRBDataset(
        prepared_dir=prepared_dir,
        sequence_ids=["s0"],
        scaler=_identity_scaler(),
        label_encoder=encoder,
    )
    assert len(ds) == 1
    sample = ds[0]

    expected_keys = {
        "imu",
        "imu_derived",
        "thm",
        "tof",
        "tof_stats",
        "label",
        "has_thm",
        "has_tof",
        "length",
    }
    assert set(sample.keys()) == expected_keys
    assert sample["imu"].shape == (12, 7)
    assert sample["imu_derived"].shape == (12, 7)
    assert sample["thm"].shape == (12, 5)
    assert sample["tof"].shape == (12, 5, 8, 8)
    assert sample["tof_stats"].shape == (12, 20)
    assert sample["imu"].dtype == torch.float32
    assert sample["imu_derived"].dtype == torch.float32
    assert sample["thm"].dtype == torch.float32
    assert sample["tof"].dtype == torch.float32
    assert sample["tof_stats"].dtype == torch.float32
    assert sample["label"].dtype == torch.long
    assert sample["has_thm"].dtype == torch.bool
    assert sample["has_tof"].dtype == torch.bool
    assert sample["length"].dtype == torch.long


def test_dataset_applies_scaler(tmp_path: Path):
    prepared_dir = tmp_path / "prepared"
    _write_seq(prepared_dir, "s0", length=10, gesture="g_a", seed=1)
    _make_index(
        prepared_dir,
        [
            {
                "sequence_id": "s0",
                "subject_id": "p1",
                "gesture": "g_a",
                "length": 10,
                "has_thm": True,
                "has_tof": True,
                "frac_nan_imu": 0.0,
                "frac_nan_thm": 0.0,
                "frac_nan_tof": 0.0,
            },
        ],
    )
    encoder = build_label_encoder(["g_a"])

    scaler = _identity_scaler()
    scaler["imu_mean"] = np.full(7, 5.0, dtype=np.float32)
    scaler["imu_std"] = np.full(7, 2.0, dtype=np.float32)

    ds = BFRBDataset(
        prepared_dir=prepared_dir, sequence_ids=["s0"], scaler=scaler, label_encoder=encoder
    )
    raw_sample = pq.read_table(prepared_dir / "sequences" / "s0.parquet").to_pydict()
    raw_imu = np.asarray(raw_sample["imu"][0])
    expected = (raw_imu - 5.0) / 2.0

    np.testing.assert_allclose(ds[0]["imu"].numpy(), expected, atol=1e-5)


def test_dataset_requires_prepared_derived_columns(tmp_path: Path):
    prepared_dir = tmp_path / "prepared"
    _write_seq(prepared_dir, "s0", length=10, gesture="g_a", include_derived=False)
    _make_index(
        prepared_dir,
        [
            {
                "sequence_id": "s0",
                "subject_id": "p1",
                "gesture": "g_a",
                "length": 10,
                "has_thm": True,
                "has_tof": True,
                "frac_nan_imu": 0.0,
                "frac_nan_thm": 0.0,
                "frac_nan_tof": 0.0,
            },
        ],
    )
    ds = BFRBDataset(
        prepared_dir=prepared_dir,
        sequence_ids=["s0"],
        scaler=_identity_scaler(),
        label_encoder=build_label_encoder(["g_a"]),
    )

    with pytest.raises(
        KeyError, match="imu_derived/tof_stats.*uv run bfrb download|uv run dvc repro prepare"
    ):
        ds[0]


def test_dataset_validates_derived_shapes_with_sequence_id(tmp_path: Path):
    prepared_dir = tmp_path / "prepared"
    _write_seq(prepared_dir, "s_bad", length=10, gesture="g_a", tof_stats_width=19)
    _make_index(
        prepared_dir,
        [
            {
                "sequence_id": "s_bad",
                "subject_id": "p1",
                "gesture": "g_a",
                "length": 10,
                "has_thm": True,
                "has_tof": True,
                "frac_nan_imu": 0.0,
                "frac_nan_thm": 0.0,
                "frac_nan_tof": 0.0,
            },
        ],
    )
    ds = BFRBDataset(
        prepared_dir=prepared_dir,
        sequence_ids=["s_bad"],
        scaler=_identity_scaler(),
        label_encoder=build_label_encoder(["g_a"]),
    )

    with pytest.raises(ValueError, match="s_bad.*tof_stats"):
        ds[0]


def test_dataset_rejects_invalid_derived_rank(tmp_path: Path):
    prepared_dir = tmp_path / "prepared"
    _write_seq(prepared_dir, "s_bad_rank", length=10, gesture="g_a")
    payload = pq.read_table(prepared_dir / "sequences" / "s_bad_rank.parquet").to_pydict()
    table = pa.table(
        {
            "imu": [np.asarray(payload["imu"][0], dtype=np.float32).tolist()],
            "imu_derived": [np.ones(7, dtype=np.float32).tolist()],
            "thm": [np.asarray(payload["thm"][0], dtype=np.float32).tolist()],
            "tof": [np.asarray(payload["tof"][0], dtype=np.float32).tolist()],
            "tof_stats": [np.asarray(payload["tof_stats"][0], dtype=np.float32).tolist()],
        }
    )
    pq.write_table(table, prepared_dir / "sequences" / "s_bad_rank.parquet")
    _make_index(
        prepared_dir,
        [
            {
                "sequence_id": "s_bad_rank",
                "subject_id": "p1",
                "gesture": "g_a",
                "length": 10,
                "has_thm": True,
                "has_tof": True,
                "frac_nan_imu": 0.0,
                "frac_nan_thm": 0.0,
                "frac_nan_tof": 0.0,
            },
        ],
    )
    ds = BFRBDataset(
        prepared_dir=prepared_dir,
        sequence_ids=["s_bad_rank"],
        scaler=_identity_scaler(),
        label_encoder=build_label_encoder(["g_a"]),
    )

    with pytest.raises(ValueError, match="s_bad_rank.*imu_derived"):
        ds[0]


def test_dataset_rejects_mismatched_feature_lengths(tmp_path: Path):
    prepared_dir = tmp_path / "prepared"
    _write_seq(prepared_dir, "s_bad_length", length=10, gesture="g_a")
    payload = pq.read_table(prepared_dir / "sequences" / "s_bad_length.parquet").to_pydict()
    table = pa.table(
        {
            "imu": [np.asarray(payload["imu"][0], dtype=np.float32).tolist()],
            "imu_derived": [np.asarray(payload["imu_derived"][0], dtype=np.float32).tolist()],
            "thm": [np.asarray(payload["thm"][0], dtype=np.float32).tolist()],
            "tof": [np.asarray(payload["tof"][0], dtype=np.float32).tolist()],
            "tof_stats": [np.asarray(payload["tof_stats"][0], dtype=np.float32)[:9].tolist()],
        }
    )
    pq.write_table(table, prepared_dir / "sequences" / "s_bad_length.parquet")
    _make_index(
        prepared_dir,
        [
            {
                "sequence_id": "s_bad_length",
                "subject_id": "p1",
                "gesture": "g_a",
                "length": 10,
                "has_thm": True,
                "has_tof": True,
                "frac_nan_imu": 0.0,
                "frac_nan_thm": 0.0,
                "frac_nan_tof": 0.0,
            },
        ],
    )
    ds = BFRBDataset(
        prepared_dir=prepared_dir,
        sequence_ids=["s_bad_length"],
        scaler=_identity_scaler(),
        label_encoder=build_label_encoder(["g_a"]),
    )

    with pytest.raises(ValueError, match="s_bad_length.*tof_stats"):
        ds[0]


def test_dataset_applies_thm_scaler_even_when_unavailable(tmp_path: Path):
    prepared_dir = tmp_path / "prepared"
    _write_seq(prepared_dir, "s_no_thm", length=10, gesture="g_a", has_thm=False, seed=2)
    _make_index(
        prepared_dir,
        [
            {
                "sequence_id": "s_no_thm",
                "subject_id": "p1",
                "gesture": "g_a",
                "length": 10,
                "has_thm": False,
                "has_tof": True,
                "frac_nan_imu": 0.0,
                "frac_nan_thm": 1.0,
                "frac_nan_tof": 0.0,
            },
        ],
    )
    encoder = build_label_encoder(["g_a"])

    scaler = _identity_scaler()
    scaler["thm_mean"] = np.full(5, 10.0, dtype=np.float32)
    scaler["thm_std"] = np.full(5, 2.0, dtype=np.float32)

    ds = BFRBDataset(
        prepared_dir=prepared_dir,
        sequence_ids=["s_no_thm"],
        scaler=scaler,
        label_encoder=encoder,
    )
    raw_sample = pq.read_table(prepared_dir / "sequences" / "s_no_thm.parquet").to_pydict()
    raw_thm = np.asarray(raw_sample["thm"][0])
    expected = (raw_thm - 10.0) / 2.0

    np.testing.assert_allclose(ds[0]["thm"].numpy(), expected, atol=1e-5)


def test_dataset_respects_availability_flags(tmp_path: Path):
    prepared_dir = tmp_path / "prepared"
    _write_seq(prepared_dir, "s_no_tof", length=10, gesture="g_a", has_tof=False)
    _make_index(
        prepared_dir,
        [
            {
                "sequence_id": "s_no_tof",
                "subject_id": "p1",
                "gesture": "g_a",
                "length": 10,
                "has_thm": True,
                "has_tof": False,
                "frac_nan_imu": 0.0,
                "frac_nan_thm": 0.0,
                "frac_nan_tof": 1.0,
            },
        ],
    )
    encoder = build_label_encoder(["g_a"])

    ds = BFRBDataset(
        prepared_dir=prepared_dir,
        sequence_ids=["s_no_tof"],
        scaler=_identity_scaler(),
        label_encoder=encoder,
    )
    sample = ds[0]
    assert bool(sample["has_tof"]) is False
    assert bool(sample["has_thm"]) is True


def test_dataset_preserves_tof_values_when_flagged_unavailable(tmp_path: Path):
    prepared_dir = tmp_path / "prepared"
    _write_seq(prepared_dir, "s_no_tof", length=10, gesture="g_a", has_tof=True, seed=3)
    _make_index(
        prepared_dir,
        [
            {
                "sequence_id": "s_no_tof",
                "subject_id": "p1",
                "gesture": "g_a",
                "length": 10,
                "has_thm": True,
                "has_tof": False,
                "frac_nan_imu": 0.0,
                "frac_nan_thm": 0.0,
                "frac_nan_tof": 1.0,
            },
        ],
    )
    encoder = build_label_encoder(["g_a"])

    ds = BFRBDataset(
        prepared_dir=prepared_dir,
        sequence_ids=["s_no_tof"],
        scaler=_identity_scaler(),
        label_encoder=encoder,
    )
    raw_sample = pq.read_table(prepared_dir / "sequences" / "s_no_tof.parquet").to_pydict()
    raw_tof = np.asarray(raw_sample["tof"][0])

    assert bool(ds[0]["has_tof"]) is False
    np.testing.assert_allclose(ds[0]["tof"].numpy(), raw_tof, atol=1e-5)


def test_dataset_applies_transform_to_sample(tmp_path: Path):
    prepared_dir = tmp_path / "prepared"
    _write_seq(prepared_dir, "s0", length=10, gesture="g_a")
    _make_index(
        prepared_dir,
        [
            {
                "sequence_id": "s0",
                "subject_id": "p1",
                "gesture": "g_a",
                "length": 10,
                "has_thm": True,
                "has_tof": True,
                "frac_nan_imu": 0.0,
                "frac_nan_thm": 0.0,
                "frac_nan_tof": 0.0,
            },
        ],
    )
    encoder = build_label_encoder(["g_a"])

    def transform(sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        sample["length"] = sample["length"] + 1
        return sample

    ds = BFRBDataset(
        prepared_dir=prepared_dir,
        sequence_ids=["s0"],
        scaler=_identity_scaler(),
        label_encoder=encoder,
        transform=transform,
    )

    assert int(ds[0]["length"]) == 11


def test_dataset_encodes_label(tmp_path: Path):
    prepared_dir = tmp_path / "prepared"
    _write_seq(prepared_dir, "s0", length=10, gesture="g_b")
    _make_index(
        prepared_dir,
        [
            {
                "sequence_id": "s0",
                "subject_id": "p1",
                "gesture": "g_b",
                "length": 10,
                "has_thm": True,
                "has_tof": True,
                "frac_nan_imu": 0.0,
                "frac_nan_thm": 0.0,
                "frac_nan_tof": 0.0,
            },
        ],
    )
    encoder = build_label_encoder(["g_a", "g_b", "g_c"])

    ds = BFRBDataset(
        prepared_dir=prepared_dir,
        sequence_ids=["s0"],
        scaler=_identity_scaler(),
        label_encoder=encoder,
    )
    assert int(ds[0]["label"]) == encoder.encode("g_b")
