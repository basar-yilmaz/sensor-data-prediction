"""Tests for the StratifiedGroupKFold splits stage."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bfrb_sensors.data.splits import SplitsConfig, make_splits


def _build_index(tmp_path: Path, n_subjects: int = 20, n_classes: int = 18) -> Path:
    rows = []
    seq_idx = 0
    for subject in range(n_subjects):
        for cls in range(n_classes):
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
    index_path = prepared_dir / "index.parquet"
    pd.DataFrame(rows).to_parquet(index_path, index=False)
    return prepared_dir


def test_splits_are_subject_disjoint(tmp_path: Path):
    prepared_dir = _build_index(tmp_path)
    cfg = SplitsConfig(prepared_dir=prepared_dir, n_folds=5, seed=42)
    make_splits(cfg)

    payload = json.loads((prepared_dir / "splits.json").read_text())
    splits = payload["folds"]
    index = pd.read_parquet(prepared_dir / "index.parquet").set_index("sequence_id")

    for fold_idx, fold in splits.items():
        train_subjects = set(index.loc[fold["train"], "subject_id"])
        val_subjects = set(index.loc[fold["val"], "subject_id"])
        assert train_subjects.isdisjoint(
            val_subjects
        ), f"fold {fold_idx} has overlapping subjects between train and val"


def test_splits_cover_all_sequences(tmp_path: Path):
    prepared_dir = _build_index(tmp_path)
    cfg = SplitsConfig(prepared_dir=prepared_dir, n_folds=5, seed=42)
    make_splits(cfg)

    payload = json.loads((prepared_dir / "splits.json").read_text())
    splits = payload["folds"]
    index = pd.read_parquet(prepared_dir / "index.parquet")
    all_ids = set(index["sequence_id"])

    for fold in splits.values():
        union = set(fold["train"]) | set(fold["val"])
        assert union == all_ids
        assert set(fold["train"]).isdisjoint(set(fold["val"]))


def test_splits_are_deterministic_when_forced(tmp_path: Path):
    prepared_dir = _build_index(tmp_path)
    cfg = SplitsConfig(prepared_dir=prepared_dir, n_folds=5, seed=42, force=True)
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

    make_splits(SplitsConfig(prepared_dir=ordered_dir, n_folds=5, seed=42))
    make_splits(SplitsConfig(prepared_dir=reversed_dir, n_folds=5, seed=42))

    assert (ordered_dir / "splits.json").read_text() == (reversed_dir / "splits.json").read_text()


def test_splits_have_expected_count(tmp_path: Path):
    prepared_dir = _build_index(tmp_path)
    cfg = SplitsConfig(prepared_dir=prepared_dir, n_folds=5, seed=42)
    make_splits(cfg)

    payload = json.loads((prepared_dir / "splits.json").read_text())
    assert sorted(payload["folds"].keys()) == ["0", "1", "2", "3", "4"]


def test_splits_write_versioned_metadata_schema(tmp_path: Path):
    prepared_dir = _build_index(tmp_path)
    cfg = SplitsConfig(prepared_dir=prepared_dir, n_folds=5, seed=42)

    make_splits(cfg)

    payload = json.loads((prepared_dir / "splits.json").read_text())
    assert sorted(payload.keys()) == ["folds", "metadata"]
    assert payload["metadata"] == {
        "version": 1,
        "algorithm": "StratifiedGroupKFold",
        "n_folds": 5,
        "seed": 42,
        "shuffle": True,
        "group_col": "subject_id",
        "stratify_col": "gesture",
        "sequence_count": 360,
        "index_hash": payload["metadata"]["index_hash"],
    }
    assert len(payload["metadata"]["index_hash"]) == 64
    assert sorted(payload["folds"].keys()) == ["0", "1", "2", "3", "4"]


def test_splits_refuse_to_overwrite_without_force(tmp_path: Path):
    prepared_dir = _build_index(tmp_path)
    cfg = SplitsConfig(prepared_dir=prepared_dir, n_folds=5, seed=42)
    make_splits(cfg)

    with pytest.raises(FileExistsError, match="splits.json already exists"):
        make_splits(cfg)
