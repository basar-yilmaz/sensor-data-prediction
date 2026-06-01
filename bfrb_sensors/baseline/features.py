"""IMU-only, sequence-level feature aggregation for the tabular XGBoost baseline.

A tree model needs a fixed-length feature vector per sequence, but the prepared
sequences are variable-length time series. We collapse the time axis with simple
per-channel summary statistics (mean/std/min/max/median by default), using *only*
the IMU-derived channels: the 7 raw IMU channels and the 7 IMU-derived channels
(linear acceleration, body-frame angular velocity, angular distance). No
thermopile or ToF information is used, so this is a strictly IMU-only baseline.

Tree splits are scale-invariant, so unlike the neural pipeline the features are
taken raw from the prepared parquet (no StandardScaler).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from bfrb_sensors.data.label_encoder import LabelEncoder

logger = logging.getLogger(__name__)

# Channel names match the column order written by the prepare stage.
IMU_CHANNELS: tuple[str, ...] = (
    "acc_x",
    "acc_y",
    "acc_z",
    "rot_w",
    "rot_x",
    "rot_y",
    "rot_z",
)
IMU_DERIVED_CHANNELS: tuple[str, ...] = (
    "lin_acc_x",
    "lin_acc_y",
    "lin_acc_z",
    "ang_vel_x",
    "ang_vel_y",
    "ang_vel_z",
    "ang_dist",
)

_STAT_FNS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "mean": lambda a: a.mean(axis=0),
    "std": lambda a: a.std(axis=0),
    "min": lambda a: a.min(axis=0),
    "max": lambda a: a.max(axis=0),
    "median": lambda a: np.median(a, axis=0),
}


@dataclass(frozen=True)
class FeatureConfig:
    use_imu_raw: bool = True
    use_imu_derived: bool = True
    stats: tuple[str, ...] = ("mean", "std", "min", "max", "median")

    def __post_init__(self) -> None:
        if not (self.use_imu_raw or self.use_imu_derived):
            raise ValueError("at least one of use_imu_raw / use_imu_derived must be true")
        unknown = [stat for stat in self.stats if stat not in _STAT_FNS]
        if unknown:
            raise ValueError(f"unknown stats {unknown}; expected subset of {sorted(_STAT_FNS)}")


@dataclass
class _FeatureSpec:
    """Resolved (parquet column, channel names) blocks to aggregate."""

    blocks: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)

    @property
    def columns(self) -> list[str]:
        return [column for column, _ in self.blocks]


def _feature_spec(cfg: FeatureConfig) -> _FeatureSpec:
    blocks: list[tuple[str, tuple[str, ...]]] = []
    if cfg.use_imu_raw:
        blocks.append(("imu", IMU_CHANNELS))
    if cfg.use_imu_derived:
        blocks.append(("imu_derived", IMU_DERIVED_CHANNELS))
    return _FeatureSpec(blocks)


def feature_names(cfg: FeatureConfig) -> list[str]:
    """Deterministic feature column names: ``<channel>_<stat>``."""
    spec = _feature_spec(cfg)
    names: list[str] = []
    for _, channels in spec.blocks:
        for channel in channels:
            names.extend(f"{channel}_{stat}" for stat in cfg.stats)
    return names


def _aggregate_block(array: np.ndarray, stats: Sequence[str]) -> np.ndarray:
    """Aggregate a (T, C) block into a (C * len(stats),) feature vector.

    Stats are interleaved per channel so the layout matches ``feature_names``:
    ``[c0_stat0, c0_stat1, ..., c1_stat0, ...]``.
    """
    per_stat = np.stack([_STAT_FNS[stat](array) for stat in stats], axis=1)  # (C, S)
    return per_stat.reshape(-1)


def extract_sequence_features(
    prepared_dir: Path, sequence_id: str, cfg: FeatureConfig
) -> np.ndarray:
    spec = _feature_spec(cfg)
    path = Path(prepared_dir) / "sequences" / f"{sequence_id}.parquet"
    payload = pq.read_table(path, columns=spec.columns).to_pydict()
    parts = [
        _aggregate_block(np.asarray(payload[column][0], dtype=np.float64), cfg.stats)
        for column, _ in spec.blocks
    ]
    return np.concatenate(parts).astype(np.float32)


def build_feature_matrix(
    prepared_dir: Path,
    sequence_ids: Sequence[str],
    label_encoder: LabelEncoder,
    gesture_by_sequence: dict[str, str],
    cfg: FeatureConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) for ``sequence_ids``.

    Returns X (n_sequences, n_features) float32 and y (n_sequences,) int64 of
    encoded gesture ids.
    """
    sequence_ids = list(sequence_ids)
    if not sequence_ids:
        raise ValueError("build_feature_matrix requires at least one sequence id")

    rows = [extract_sequence_features(prepared_dir, sid, cfg) for sid in sequence_ids]
    X = np.stack(rows, axis=0)
    y = np.array(
        [label_encoder.encode(gesture_by_sequence[sid]) for sid in sequence_ids],
        dtype=np.int64,
    )
    logger.info("Built IMU-only feature matrix: %d sequences x %d features", X.shape[0], X.shape[1])
    return X, y
