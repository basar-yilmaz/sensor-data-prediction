"""Tests for the fixed train/validation/test splits stage."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bfrb_sensors.data.splits import SplitsConfig, load_split_file, load_splits, make_splits


def _build_index(tmp_path: Path, n_subjects: int = 20, n_classes: int = 3, reps: int = 2) -> Path:
    """Synthetic index with multiple sequences per subject across all gestures.

    Each subject contributes ``n_classes * reps`` sequences, so subject grouping
    is non-trivial (a subject spans several sequences and gestures).
    """
    rows = []
    seq_idx = 0
    for subject in range(n_subjects):
        for cls in range(n_classes):
            for _ in range(reps):
                rows.append(
                    {
                        "sequence_id": f"s{seq_idx:04d}",
                        "subject_id": f"p{subject:02d}",
                        "gesture": f"gesture_{cls:02d}",
                        "length": 20,
                        "has_thm": True,
                        "has_tof": True,
                        "frac_nan_imu": 0.0,
                        "frac_nan_thm": 0.0,
                        "frac_nan_tof": 0.0,
                    }
                )
                seq_idx += 1
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(prepared_dir / "index.parquet", index=False)
    return prepared_dir


def _subjects(index: pd.DataFrame, sequence_ids: list[str]) -> set[str]:
    return set(index[index["sequence_id"].isin(sequence_ids)]["subject_id"])


def test_splits_cover_all_sequences_once(tmp_path: Path):
    prepared_dir = _build_index(tmp_path)
    make_splits(SplitsConfig(prepared_dir=prepared_dir, seed=42))

    splits = load_splits(prepared_dir)
    index = pd.read_parquet(prepared_dir / "index.parquet")
    all_ids = set(index["sequence_id"])

    assert set(splits) == {"train", "val", "test"}
    assert set(splits["train"]) | set(splits["val"]) | set(splits["test"]) == all_ids
    assert set(splits["train"]).isdisjoint(splits["val"])
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert set(splits["val"]).isdisjoint(splits["test"])


def test_splits_are_subject_disjoint(tmp_path: Path):
    prepared_dir = _build_index(tmp_path)
    make_splits(SplitsConfig(prepared_dir=prepared_dir, seed=42))

    splits = load_splits(prepared_dir)
    index = pd.read_parquet(prepared_dir / "index.parquet")
    train_subj = _subjects(index, splits["train"])
    val_subj = _subjects(index, splits["val"])
    test_subj = _subjects(index, splits["test"])

    assert train_subj.isdisjoint(val_subj)
    assert train_subj.isdisjoint(test_subj)
    assert val_subj.isdisjoint(test_subj)
    assert train_subj | val_subj | test_subj == set(index["subject_id"])


def test_splits_approximate_target_ratios(tmp_path: Path):
    prepared_dir = _build_index(tmp_path)
    make_splits(SplitsConfig(prepared_dir=prepared_dir, seed=42))

    splits = load_splits(prepared_dir)
    total = sum(len(ids) for ids in splits.values())

    # Whole subjects move together, so ratios approximate rather than match exactly.
    assert len(splits["train"]) / total == pytest.approx(0.8, abs=0.1)
    assert len(splits["val"]) / total == pytest.approx(0.1, abs=0.1)
    assert len(splits["test"]) / total == pytest.approx(0.1, abs=0.1)


def test_splits_are_deterministic_when_forced(tmp_path: Path):
    prepared_dir = _build_index(tmp_path)
    cfg = SplitsConfig(prepared_dir=prepared_dir, seed=42, force=True)
    make_splits(cfg)
    first = (prepared_dir / "splits.json").read_text()

    make_splits(cfg)
    second = (prepared_dir / "splits.json").read_text()
    assert first == second


def test_splits_are_independent_of_index_row_order(tmp_path: Path):
    ordered_dir = _build_index(tmp_path / "ordered")
    reversed_dir = _build_index(tmp_path / "reversed")
    reversed_index_path = reversed_dir / "index.parquet"
    index = pd.read_parquet(reversed_index_path)
    index.iloc[::-1].to_parquet(reversed_index_path, index=False)

    make_splits(SplitsConfig(prepared_dir=ordered_dir, seed=42))
    make_splits(SplitsConfig(prepared_dir=reversed_dir, seed=42))

    assert (ordered_dir / "splits.json").read_text() == (reversed_dir / "splits.json").read_text()


def test_splits_write_versioned_metadata_schema(tmp_path: Path):
    prepared_dir = _build_index(tmp_path)

    make_splits(SplitsConfig(prepared_dir=prepared_dir, seed=42))

    payload = json.loads((prepared_dir / "splits.json").read_text())
    assert sorted(payload.keys()) == ["metadata", "splits"]
    assert payload["metadata"] == {
        "version": 1,
        "algorithm": "subject_disjoint_stratified_group_kfold",
        "train_size": 0.8,
        "val_size": 0.1,
        "test_size": 0.1,
        "seed": 42,
        "shuffle": True,
        "stratify_col": "gesture",
        "group_col": "subject_id",
        "sequence_count": 120,
        "index_hash": payload["metadata"]["index_hash"],
    }
    assert len(payload["metadata"]["index_hash"]) == 64
    assert sorted(payload["splits"].keys()) == ["test", "train", "val"]


def test_splits_refuse_to_overwrite_without_force(tmp_path: Path):
    prepared_dir = _build_index(tmp_path)
    cfg = SplitsConfig(prepared_dir=prepared_dir, seed=42)
    make_splits(cfg)

    with pytest.raises(FileExistsError, match="splits.json already exists"):
        make_splits(cfg)


def test_load_split_file_rejects_legacy_schema(tmp_path: Path):
    prepared_dir = _build_index(tmp_path)
    (prepared_dir / "splits.json").write_text(
        json.dumps({"0": {"train": ["s0000"], "val": ["s0001"]}})
    )

    with pytest.raises(ValueError, match="versioned split schema"):
        load_split_file(prepared_dir)


def test_load_split_file_rejects_non_string_sequence_ids(tmp_path: Path):
    prepared_dir = _build_index(tmp_path)
    (prepared_dir / "splits.json").write_text(
        json.dumps(
            {
                "metadata": {"version": 1},
                "splits": {"train": [1], "val": ["s0001"], "test": [None]},
            }
        )
    )

    with pytest.raises(ValueError, match="sequence IDs must be strings"):
        load_split_file(prepared_dir)


def test_split_ratios_must_sum_to_one(tmp_path: Path):
    prepared_dir = _build_index(tmp_path)

    with pytest.raises(ValueError, match="must sum to 1.0"):
        make_splits(SplitsConfig(prepared_dir=prepared_dir, train_size=0.7))
