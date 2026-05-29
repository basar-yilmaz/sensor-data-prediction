"""Per-subject demographics: column contract, fold-wise normalization, lookup.

Demographics are static per subject. The continuous columns are z-scored with
statistics fit on the training fold's unique subjects only (no validation leakage);
the binary columns are passed through as float32 0/1 values. ``subject`` is never a
model input -- it is only the join key onto the sequence index's ``subject_id``.
"""

from __future__ import annotations

import logging
from pathlib import Path

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
