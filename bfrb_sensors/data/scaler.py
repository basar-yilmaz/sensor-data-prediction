"""Per-fold StandardScaler computed on train-fold IMU + THM channels only.

ToF is left unscaled (bounded distance values, per-pixel scaling is brittle).
The scaler is stored as a plain dict of numpy arrays for fast load + transparent inspection.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScalerConfig:
    prepared_dir: Path
    artifacts_dir: Path
    fold_idx: int


def _load_sequence(prepared_dir: Path, sequence_id: str) -> tuple[np.ndarray, np.ndarray]:
    path = prepared_dir / "sequences" / f"{sequence_id}.parquet"
    table = pq.read_table(path, columns=["imu", "thm"])
    payload = table.to_pydict()
    imu = np.asarray(payload["imu"][0], dtype=np.float64)
    thm = np.asarray(payload["thm"][0], dtype=np.float64)
    return imu, thm


def _scaler_path(cfg: ScalerConfig) -> Path:
    return Path(cfg.artifacts_dir) / f"scaler_fold{cfg.fold_idx}.joblib"


def fit_scaler(cfg: ScalerConfig, train_sequence_ids: Iterable[str]) -> Path:
    sequence_ids = list(train_sequence_ids)
    if not sequence_ids:
        raise ValueError("fit_scaler requires at least one training sequence id")

    artifacts_dir = Path(cfg.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    imu_chunks: list[np.ndarray] = []
    thm_chunks: list[np.ndarray] = []
    for sequence_id in sequence_ids:
        imu, thm = _load_sequence(Path(cfg.prepared_dir), sequence_id)
        imu_chunks.append(imu)
        thm_chunks.append(thm)

    imu_all = np.concatenate(imu_chunks, axis=0)
    thm_all = np.concatenate(thm_chunks, axis=0)

    imu_mean = imu_all.mean(axis=0).astype(np.float32)
    imu_std = imu_all.std(axis=0).astype(np.float32)
    thm_mean = thm_all.mean(axis=0).astype(np.float32)
    thm_std = thm_all.std(axis=0).astype(np.float32)

    # Guard against zero-variance channels.
    imu_std = np.where(imu_std < 1e-8, np.float32(1.0), imu_std)
    thm_std = np.where(thm_std < 1e-8, np.float32(1.0), thm_std)

    payload = {
        "fold_idx": int(cfg.fold_idx),
        "n_timesteps": int(imu_all.shape[0]),
        "imu_mean": imu_mean,
        "imu_std": imu_std,
        "thm_mean": thm_mean,
        "thm_std": thm_std,
    }

    path = _scaler_path(cfg)
    joblib.dump(payload, path)
    logger.info(
        "Fit scaler (fold %d) on %d sequences / %d timesteps -> %s",
        cfg.fold_idx,
        len(sequence_ids),
        imu_all.shape[0],
        path,
    )
    logger.debug(
        "Scaler stats: imu_mean=%s, imu_std=%s, thm_mean=%s, thm_std=%s",
        imu_mean.tolist(),
        imu_std.tolist(),
        thm_mean.tolist(),
        thm_std.tolist(),
    )
    return path


def load_scaler(path: Path) -> dict:
    return joblib.load(Path(path))


def scaler_path(cfg: ScalerConfig) -> Path:
    return _scaler_path(cfg)
