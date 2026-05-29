from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bfrb_sensors.data.demographics import (
    BINARY_COLUMNS,
    CONTINUOUS_COLUMNS,
    DEMOGRAPHICS_DIM,
    OUTPUT_COLUMNS,
    DemographicsLookup,
    fit_demographics_stats,
    load_demographics_stats,
    write_demographics_parquet,
)


def _demographics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject": "S1",
                "adult_child": 1,
                "age": 40,
                "sex": 1,
                "handedness": 1,
                "height_cm": 170.0,
                "shoulder_to_wrist_cm": 50,
                "elbow_to_wrist_cm": 25.0,
            },
            {
                "subject": "S2",
                "adult_child": 0,
                "age": 12,
                "sex": 0,
                "handedness": 1,
                "height_cm": 150.0,
                "shoulder_to_wrist_cm": 45,
                "elbow_to_wrist_cm": 22.0,
            },
        ]
    )


def _index() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"sequence_id": "q1", "subject_id": "S1"},
            {"sequence_id": "q2", "subject_id": "S2"},
        ]
    )


def test_column_contract_shapes():
    assert BINARY_COLUMNS == ("adult_child", "sex", "handedness")
    assert CONTINUOUS_COLUMNS == ("age", "height_cm", "shoulder_to_wrist_cm", "elbow_to_wrist_cm")
    assert len(OUTPUT_COLUMNS) == 7


def test_write_demographics_parquet(tmp_path):
    csv = tmp_path / "demo.csv"
    _demographics().to_csv(csv, index=False)
    out = tmp_path / "demographics.parquet"

    write_demographics_parquet(csv, _index(), out)

    written = pd.read_parquet(out)
    assert set(written["subject_id"]) == {"S1", "S2"}
    assert list(written.columns) == ["subject_id", *BINARY_COLUMNS, *CONTINUOUS_COLUMNS]


def test_missing_subject_is_hard_error(tmp_path):
    csv = tmp_path / "demo.csv"
    _demographics().iloc[:1].to_csv(csv, index=False)  # drop S2
    out = tmp_path / "demographics.parquet"

    with pytest.raises(ValueError, match="without demographics"):
        write_demographics_parquet(csv, _index(), out)


def test_bad_binary_domain_is_hard_error(tmp_path):
    bad = _demographics()
    bad.loc[0, "sex"] = 2
    csv = tmp_path / "demo.csv"
    bad.to_csv(csv, index=False)
    out = tmp_path / "demographics.parquet"

    with pytest.raises(ValueError, match="binary column"):
        write_demographics_parquet(csv, _index(), out)


def test_null_binary_value_is_hard_error(tmp_path):
    bad = _demographics()
    bad.loc[0, "handedness"] = None
    csv = tmp_path / "demo.csv"
    bad.to_csv(csv, index=False)
    out = tmp_path / "demographics.parquet"

    with pytest.raises(ValueError, match="null values"):
        write_demographics_parquet(csv, _index(), out)


def test_extra_demographics_subject_is_allowed(tmp_path):
    extra = pd.concat(
        [
            _demographics(),
            pd.DataFrame(
                [
                    {
                        "subject": "S99",
                        "adult_child": 1,
                        "age": 30,
                        "sex": 1,
                        "handedness": 0,
                        "height_cm": 160.0,
                        "shoulder_to_wrist_cm": 48,
                        "elbow_to_wrist_cm": 24.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    csv = tmp_path / "demo.csv"
    extra.to_csv(csv, index=False)
    out = tmp_path / "demographics.parquet"

    write_demographics_parquet(csv, _index(), out)  # must not raise
    written = pd.read_parquet(out)
    assert set(written["subject_id"]) == {"S1", "S2"}  # extra row dropped


def _written_parquet(tmp_path):
    csv = tmp_path / "demo.csv"
    _demographics().to_csv(csv, index=False)
    out = tmp_path / "demographics.parquet"
    write_demographics_parquet(csv, _index(), out)
    return out


def test_stats_fit_over_unique_subjects_not_sequences(tmp_path):
    parquet = _written_parquet(tmp_path)
    # S1 appears in many sequences, S2 in one; stats must NOT be skewed toward S1.
    index = pd.DataFrame(
        [{"sequence_id": f"q{i}", "subject_id": "S1"} for i in range(9)]
        + [{"sequence_id": "qz", "subject_id": "S2"}]
    )
    train_ids = index["sequence_id"].tolist()

    path = fit_demographics_stats(parquet, index, train_ids, fold_idx=0, artifacts_dir=tmp_path)
    stats = load_demographics_stats(path)

    # age mean over unique subjects {S1:40, S2:12} == 26, not weighted toward 40
    age_idx = stats["continuous_columns"].index("age")
    assert np.isclose(stats["mean"][age_idx], 26.0)
    assert stats["output_columns"] == list(OUTPUT_COLUMNS)


def test_lookup_returns_normalized_vector(tmp_path):
    parquet = _written_parquet(tmp_path)
    index = _index()
    path = fit_demographics_stats(
        parquet, index, index["sequence_id"].tolist(), fold_idx=0, artifacts_dir=tmp_path
    )
    stats = load_demographics_stats(path)
    lookup = DemographicsLookup(parquet, stats)

    vec = lookup.vector("S1")
    assert vec.shape == (DEMOGRAPHICS_DIM,)
    assert vec.dtype == np.float32
    # binary cols pass through unscaled: adult_child=1, sex=1, handedness=1
    assert vec[0] == 1.0 and vec[1] == 1.0 and vec[2] == 1.0
    # continuous cols are z-scored; mean over {S1,S2} so S1 age (40>26) is positive
    age_z = vec[3]
    assert age_z > 0
