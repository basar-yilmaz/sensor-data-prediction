"""Fixed, subject-disjoint train/validation/test splits over prepared sequences.

The split is stratified by gesture *and* grouped by subject (via two-stage
``StratifiedGroupKFold``), so no participant appears in more than one of
train/val/test. Because whole subjects move together, the realized fractions
approximate ``train_size/val_size/test_size`` rather than matching them exactly.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitsConfig:
    prepared_dir: Path
    train_size: float = 0.8
    val_size: float = 0.1
    test_size: float = 0.1
    seed: int = 42
    stratify_col: str = "gesture"
    group_col: str = "subject_id"
    force: bool = False


def load_split_file(prepared_dir: Path | str) -> dict[str, Any]:
    path = Path(prepared_dir) / "splits.json"
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or set(payload) != {"metadata", "splits"}:
        raise ValueError(f"{path} must use the versioned split schema with metadata and splits")

    metadata = payload["metadata"]
    splits = payload["splits"]
    if not isinstance(metadata, dict) or metadata.get("version") != 1:
        raise ValueError(f"{path} has unsupported split schema metadata")
    if not isinstance(splits, dict) or set(splits) != {"train", "val", "test"}:
        raise ValueError(f"{path} split payload must contain train, val, and test lists")
    for split_name, sequence_ids in splits.items():
        if not isinstance(sequence_ids, list):
            raise ValueError(f"{path} {split_name} split must be a list")
        if not all(isinstance(sequence_id, str) for sequence_id in sequence_ids):
            raise ValueError(f"{path} {split_name} split sequence IDs must be strings")
    return payload


def load_splits(prepared_dir: Path | str) -> dict[str, list[str]]:
    """Load the train/val/test split, verifying it matches the local prepared index.

    The split file is committed to git (frozen across machines). This guard
    recomputes the index hash and raises if the local ``index.parquet`` differs
    from the data the split was built on, so a mismatched split fails loudly
    instead of silently training on the wrong partition.
    """
    prepared_dir = Path(prepared_dir)
    payload = load_split_file(prepared_dir)
    _verify_index_hash(prepared_dir, payload["metadata"])
    return payload["splits"]


def _verify_index_hash(prepared_dir: Path, metadata: dict[str, Any]) -> None:
    expected = metadata.get("index_hash")
    stratify_col = metadata.get("stratify_col")
    group_col = metadata.get("group_col")
    index_path = prepared_dir / "index.parquet"
    # Only verifiable when the split carries the hashing metadata and the index exists.
    if not (expected and stratify_col and group_col and index_path.exists()):
        return
    index = pd.read_parquet(index_path)
    actual = _index_hash(index, stratify_col=str(stratify_col), group_col=str(group_col))
    if actual != expected:
        raise ValueError(
            f"index.parquet does not match splits.json (index_hash mismatch: expected "
            f"{expected[:12]}…, got {actual[:12]}…). The committed split was built from "
            "different prepared data; regenerate the split (`bfrb splits "
            "data.splits.force=true`) or restore the matching index."
        )


def _index_hash(index: pd.DataFrame, *, stratify_col: str, group_col: str) -> str:
    rows = (
        index[["sequence_id", stratify_col, group_col]]
        .sort_values("sequence_id")
        .astype(str)
        .to_dict(orient="records")
    )
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _holdout_fold(
    sequence_ids: np.ndarray,
    stratify: np.ndarray,
    groups: np.ndarray,
    holdout_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Carve one subject-disjoint, gesture-stratified holdout off the input.

    Uses ``StratifiedGroupKFold`` with ``n_splits = round(1 / holdout_fraction)``
    and returns ``(rest_idx, holdout_idx)`` positional indices.
    """
    n_splits = max(2, round(1.0 / holdout_fraction))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest_idx, holdout_idx = next(splitter.split(sequence_ids, stratify, groups))
    return rest_idx, holdout_idx


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
            f"{splits_path} already exists; pass data.splits.force=true to regenerate splits."
        )

    split_total = cfg.train_size + cfg.val_size + cfg.test_size
    if abs(split_total - 1.0) > 1e-8:
        raise ValueError("train_size, val_size, and test_size must sum to 1.0")
    if min(cfg.train_size, cfg.val_size, cfg.test_size) <= 0:
        raise ValueError("train_size, val_size, and test_size must all be positive")

    index = pd.read_parquet(index_path).sort_values("sequence_id").reset_index(drop=True)
    logger.info(
        "Building subject-disjoint train/val/test splits over %d sequences, %d subjects "
        "(target %.2f/%.2f/%.2f)",
        len(index),
        index[cfg.group_col].nunique(),
        cfg.train_size,
        cfg.val_size,
        cfg.test_size,
    )

    sequence_ids = index["sequence_id"].astype(str).to_numpy()
    stratify = index[cfg.stratify_col].to_numpy()
    groups = index[cfg.group_col].astype(str).to_numpy()

    # Stage 1: peel off the test split (subject-disjoint, gesture-stratified).
    trainval_idx, test_idx = _holdout_fold(sequence_ids, stratify, groups, cfg.test_size, cfg.seed)
    # Stage 2: peel val off the remaining train+val, sized relative to that subset.
    relative_val = cfg.val_size / (cfg.train_size + cfg.val_size)
    rel_train_idx, rel_val_idx = _holdout_fold(
        sequence_ids[trainval_idx],
        stratify[trainval_idx],
        groups[trainval_idx],
        relative_val,
        cfg.seed,
    )
    train_idx = trainval_idx[rel_train_idx]
    val_idx = trainval_idx[rel_val_idx]

    splits = {
        "train": sequence_ids[train_idx].tolist(),
        "val": sequence_ids[val_idx].tolist(),
        "test": sequence_ids[test_idx].tolist(),
    }
    total = len(index)
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        logger.info(
            "Split %s: %d sequences (%.3f), %d subjects",
            name,
            len(idx),
            len(idx) / total,
            len(set(groups[idx])),
        )

    payload = {
        "metadata": {
            "version": 1,
            "algorithm": "subject_disjoint_stratified_group_kfold",
            "train_size": float(cfg.train_size),
            "val_size": float(cfg.val_size),
            "test_size": float(cfg.test_size),
            "seed": int(cfg.seed),
            "shuffle": True,
            "stratify_col": cfg.stratify_col,
            "group_col": cfg.group_col,
            "sequence_count": int(len(index)),
            "index_hash": _index_hash(
                index, stratify_col=cfg.stratify_col, group_col=cfg.group_col
            ),
        },
        "splits": splits,
    }
    splits_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    logger.info("Wrote splits to %s", splits_path)
