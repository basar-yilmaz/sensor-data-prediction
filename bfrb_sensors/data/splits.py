"""StratifiedGroupKFold splits over (gesture_label, subject_id) pairs."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitsConfig:
    prepared_dir: Path
    n_folds: int = 5
    seed: int = 42
    group_col: str = "subject_id"
    stratify_col: str = "gesture"
    force: bool = False


def _index_hash(index: pd.DataFrame, *, group_col: str, stratify_col: str) -> str:
    rows = (
        index[["sequence_id", group_col, stratify_col]]
        .sort_values("sequence_id")
        .astype(str)
        .to_dict(orient="records")
    )
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_splits(cfg: SplitsConfig) -> None:
    prepared_dir = Path(cfg.prepared_dir)
    index_path = prepared_dir / "index.parquet"
    if not index_path.exists():
        raise FileNotFoundError(
            f"index parquet not found at {index_path}. Run `bfrb prepare` first."
        )

    splits_path = prepared_dir / "splits.json"
    if splits_path.exists() and not cfg.force:
        raise FileExistsError(
            f"{splits_path} already exists; pass data.splits.force=true to regenerate folds."
        )

    index = pd.read_parquet(index_path)
    logger.info(
        "Building %d-fold StratifiedGroupKFold splits over %d sequences, %d subjects",
        cfg.n_folds,
        len(index),
        index[cfg.group_col].nunique(),
    )

    splitter = StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    sequence_ids = index["sequence_id"].to_numpy()
    stratify = index[cfg.stratify_col].to_numpy()
    groups = index[cfg.group_col].to_numpy()

    folds: dict[str, dict[str, list[str]]] = {}
    global_class_freq = index[cfg.stratify_col].value_counts(normalize=True)

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(sequence_ids, stratify, groups)):
        train_ids = sequence_ids[train_idx].tolist()
        val_ids = sequence_ids[val_idx].tolist()
        folds[str(fold_idx)] = {"train": train_ids, "val": val_ids}

        val_subset = index.iloc[val_idx]
        val_class_freq = val_subset[cfg.stratify_col].value_counts(normalize=True)
        deviation = float((val_class_freq - global_class_freq).abs().max())
        logger.info(
            "Fold %d: train=%d, val=%d, unique_subjects(train=%d, val=%d), max_class_dev=%.3f",
            fold_idx,
            len(train_ids),
            len(val_ids),
            index.iloc[train_idx][cfg.group_col].nunique(),
            val_subset[cfg.group_col].nunique(),
            deviation,
        )

    payload = {
        "metadata": {
            "version": 1,
            "algorithm": "StratifiedGroupKFold",
            "n_folds": int(cfg.n_folds),
            "seed": int(cfg.seed),
            "shuffle": True,
            "group_col": cfg.group_col,
            "stratify_col": cfg.stratify_col,
            "sequence_count": int(len(index)),
            "index_hash": _index_hash(
                index, group_col=cfg.group_col, stratify_col=cfg.stratify_col
            ),
        },
        "folds": folds,
    }
    splits_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    logger.info("Wrote splits to %s", splits_path)
