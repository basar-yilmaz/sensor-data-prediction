"""Tests for the deploy-side CSV -> tensor preprocessing pipeline."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from app.preprocessing import (
    PreprocessingError,
    _tof_per_sensor_stats,
    featurize,
    featurize_and_collate,
    parse_csv,
)


def _make_csv(
    n_rows: int = 20,
    sequence_id: str = "SEQ_TEST",
    missing_thm: bool = False,
    missing_tof: bool = False,
) -> bytes:
    rng = np.random.default_rng(0)
    data = {
        "row_id": np.arange(n_rows),
        "sequence_type": "Target",
        "sequence_id": sequence_id,
        "sequence_counter": np.arange(n_rows),
        "subject": "SUBJ_X",
        "orientation": "lr",
        "behavior": "normal",
        "phase": "Gesture",
        "gesture": "Forehead - scratch",
    }
    data.update({f"acc_{axis}": rng.normal(0, 1, n_rows) for axis in "xyz"})
    data.update({f"rot_{axis}": rng.normal(0, 0.1, n_rows) for axis in "wxyz"})
    if missing_thm:
        data.update({f"thm_{i}": np.full(n_rows, np.nan) for i in range(1, 6)})
    else:
        data.update({f"thm_{i}": rng.normal(25, 1, n_rows) for i in range(1, 6)})
    if missing_tof:
        data.update(
            {f"tof_{s}_v{p}": np.full(n_rows, -1.0) for s in range(1, 6) for p in range(64)}
        )
    else:
        data.update(
            {f"tof_{s}_v{p}": rng.integers(20, 200, n_rows) for s in range(1, 6) for p in range(64)}
        )
    return pd.DataFrame(data).to_csv(index=False).encode("utf-8")


def test_parse_csv_returns_one_sequence_per_group():
    csv_bytes = (
        _make_csv(n_rows=20, sequence_id="SEQ_A")
        + _make_csv(n_rows=20, sequence_id="SEQ_B").decode().encode()
    )
    # The above concatenation mixes CSV strings; build cleanly instead.
    buf = io.StringIO()
    a = pd.read_csv(io.BytesIO(_make_csv(n_rows=20, sequence_id="SEQ_A")))
    b = pd.read_csv(io.BytesIO(_make_csv(n_rows=20, sequence_id="SEQ_B")))
    pd.concat([a, b]).to_csv(buf, index=False)
    csv_bytes = buf.getvalue().encode()

    sequences = parse_csv(csv_bytes)
    assert {s.sequence_id for s in sequences} == {"SEQ_A", "SEQ_B"}
    for seq in sequences:
        assert seq.imu.shape == (20, 7)
        assert seq.thm.shape == (20, 5)
        assert seq.tof.shape == (20, 5, 8, 8)


def test_parse_csv_detects_missing_thm_and_tof():
    csv = _make_csv(missing_thm=True, missing_tof=True)
    sequences = parse_csv(csv)
    assert len(sequences) == 1
    seq = sequences[0]
    assert seq.has_thm is False
    assert seq.has_tof is False
    # has_thm=False stores zeros for THM so the shape is preserved.
    assert np.all(seq.thm == 0.0)
    assert np.all(seq.tof == 0.0)


def test_parse_csv_raises_on_missing_columns():
    df = pd.DataFrame({"foo": [1, 2, 3]})
    csv = df.to_csv(index=False).encode("utf-8")
    with pytest.raises(PreprocessingError) as exc:
        parse_csv(csv)
    assert exc.value.code == "missing_columns"
    assert exc.value.details and "missing" in exc.value.details


def test_parse_csv_raises_on_empty_file():
    with pytest.raises(PreprocessingError) as exc:
        parse_csv(b"")
    assert exc.value.code == "empty_file"


def test_parse_csv_raises_when_all_sequences_below_min_length():
    csv = _make_csv(n_rows=5)
    with pytest.raises(PreprocessingError) as exc:
        parse_csv(csv, min_seq_length=10)
    assert exc.value.code == "no_sequences"


def test_tof_per_sensor_stats_handles_sentinel():
    tof = np.full((4, 5, 8, 8), -1.0, dtype=np.float64)
    stats = _tof_per_sensor_stats(tof, sentinel=-1.0)
    assert stats.shape == (4, 20)
    # All pixels missing -> per-sensor means are NaN; sum 20 = 5 sensors * 4 stats
    assert np.isnan(stats).any()


def test_featurize_runs_end_to_end(synthetic_scaler):
    sequences = parse_csv(_make_csv(n_rows=30))
    f = featurize(sequences[0], synthetic_scaler)
    assert f.imu.shape == (30, 7)
    assert f.imu_derived.shape == (30, 7)
    assert f.thm.shape == (30, 5)
    assert f.tof_stats.shape == (30, 20)
    assert f.tof is not None and f.tof.shape == (30, 5, 8, 8)


def test_collate_produces_model_ready_batch(synthetic_scaler):
    seqs = parse_csv(_make_csv(n_rows=20, sequence_id="A")) + parse_csv(
        _make_csv(n_rows=35, sequence_id="B")
    )
    batch = featurize_and_collate(seqs, synthetic_scaler)
    assert batch["imu"].shape[0] == 2
    assert batch["imu"].shape[1] == 35
    assert batch["imu_derived"].shape == (2, 35, 7)
    assert batch["thm"].shape == (2, 35, 5)
    assert batch["tof_stats"].shape == (2, 35, 20)
    assert batch["tof"].shape == (2, 35, 5, 8, 8)
    assert batch["attention_mask"].shape == (2, 35)
    assert batch["attention_mask"][0, 20:].sum().item() == 0
    assert batch["attention_mask"][0, :20].sum().item() == 20
    assert batch["length"].tolist() == [20, 35]


def test_collate_truncates_to_max_seq_length(synthetic_scaler):
    seqs = parse_csv(_make_csv(n_rows=50))
    batch = featurize_and_collate(seqs, synthetic_scaler, max_seq_length=10)
    assert batch["imu"].shape[1] == 10
    assert int(batch["length"][0].item()) == 10
