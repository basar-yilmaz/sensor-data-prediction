"""Per-subject demographics: column contract, fold-wise normalization, lookup.

Demographics are static per subject. The continuous columns are z-scored with
statistics fit on the training fold's unique subjects only (no validation leakage);
the binary columns are passed through as float32 0/1 values. ``subject`` is never a
model input -- it is only the join key onto the sequence index's ``subject_id``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SUBJECT_COL = "subject"
BINARY_COLUMNS: tuple[str, ...] = ("adult_child", "sex", "handedness")
CONTINUOUS_COLUMNS: tuple[str, ...] = (
    "age",
    "height_cm",
    "shoulder_to_wrist_cm",
    "elbow_to_wrist_cm",
)
OUTPUT_COLUMNS: tuple[str, ...] = (
    *BINARY_COLUMNS,
    *tuple(f"{name}_z" for name in CONTINUOUS_COLUMNS),
)
DEMOGRAPHICS_DIM = len(OUTPUT_COLUMNS)


def write_demographics_parquet(
    demographics_csv: Path,
    index: pd.DataFrame,
    out_path: Path,
) -> Path:
    """Validate the demographics CSV against the index and write a parquet of raw
    values restricted to subjects present in the index."""
    demographics = pd.read_csv(demographics_csv)
    required = {SUBJECT_COL, *BINARY_COLUMNS, *CONTINUOUS_COLUMNS}
    missing_cols = required - set(demographics.columns)
    if missing_cols:
        raise ValueError(f"demographics CSV missing columns: {sorted(missing_cols)}")

    index_subjects = set(index["subject_id"].astype(str))
    demographics[SUBJECT_COL] = demographics[SUBJECT_COL].astype(str)
    demographics_subjects = set(demographics[SUBJECT_COL])

    without = index_subjects - demographics_subjects
    if without:
        raise ValueError(
            f"{len(without)} index subjects without demographics: {sorted(without)[:10]}"
        )
    extra = demographics_subjects - index_subjects
    if extra:
        logger.warning("Dropping %d demographics subjects absent from index", len(extra))

    demographics = demographics[demographics[SUBJECT_COL].isin(index_subjects)].copy()

    for column in BINARY_COLUMNS:
        if demographics[column].isna().any():
            raise ValueError(f"binary column {column!r} has null values")
        values = set(demographics[column].unique())
        if not values <= {0, 1}:
            raise ValueError(f"binary column {column!r} has values outside {{0,1}}: {values}")

    for column in CONTINUOUS_COLUMNS:
        col = demographics[column]
        if col.isna().any() or not np.isfinite(col.to_numpy(dtype=float)).all():
            raise ValueError(f"continuous column {column!r} has null/non-finite values")
        if (col <= 0).any():
            logger.warning("continuous column %r has non-positive values", column)

    out = demographics.rename(columns={SUBJECT_COL: "subject_id"})[
        ["subject_id", *BINARY_COLUMNS, *CONTINUOUS_COLUMNS]
    ].reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path)
    logger.info("Wrote demographics for %d subjects to %s", len(out), out_path)
    return out_path


def demographics_stats_path(artifacts_dir: Path, fold_idx: int) -> Path:
    return Path(artifacts_dir) / f"demographics_stats_fold{fold_idx}.joblib"


def fit_demographics_stats(
    demographics_parquet: Path,
    index: pd.DataFrame,
    train_sequence_ids: list[str],
    fold_idx: int,
    artifacts_dir: Path,
) -> Path:
    """Fit z-score stats for continuous demographics over the unique TRAIN subjects."""
    demographics = pd.read_parquet(demographics_parquet).set_index("subject_id")
    train = index[index["sequence_id"].isin(set(train_sequence_ids))]
    train_subjects = sorted(set(train["subject_id"].astype(str)))
    if not train_subjects:
        raise ValueError("fit_demographics_stats requires at least one train subject")

    continuous = demographics.loc[train_subjects, list(CONTINUOUS_COLUMNS)].to_numpy(
        dtype=np.float64
    )
    mean = continuous.mean(axis=0).astype(np.float32)
    std = continuous.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-8, np.float32(1.0), std)  # zero-variance guard

    payload = {
        "fold_idx": int(fold_idx),
        "continuous_columns": list(CONTINUOUS_COLUMNS),
        "binary_columns": list(BINARY_COLUMNS),
        "output_columns": list(OUTPUT_COLUMNS),
        "mean": mean,
        "std": std,
    }
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = demographics_stats_path(artifacts_dir, fold_idx)
    joblib.dump(payload, path)
    logger.info(
        "Fit demographics stats (fold %d) over %d unique subjects -> %s",
        fold_idx,
        len(train_subjects),
        path,
    )
    return path


def load_demographics_stats(path: Path) -> dict:
    return joblib.load(Path(path))


class DemographicsLookup:
    """Maps ``subject_id`` to the fixed ``(DEMOGRAPHICS_DIM,)`` float32 model vector."""

    def __init__(self, demographics_parquet: Path, stats: dict) -> None:
        demographics = pd.read_parquet(demographics_parquet).set_index("subject_id")
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)

        binary = demographics[list(BINARY_COLUMNS)].to_numpy(dtype=np.float64)
        continuous = demographics[list(CONTINUOUS_COLUMNS)].to_numpy(dtype=np.float64)
        normalized = (continuous - mean.astype(np.float64)) / std.astype(np.float64)
        vectors = np.concatenate([binary, normalized], axis=1).astype(np.float32)

        self._by_subject = {
            str(subject): vectors[row] for row, subject in enumerate(demographics.index.astype(str))
        }

    def vector(self, subject_id: str) -> np.ndarray:
        try:
            return self._by_subject[str(subject_id)]
        except KeyError as exc:
            raise KeyError(f"no demographics for subject {subject_id!r}") from exc
