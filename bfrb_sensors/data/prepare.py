"""Offline preparation stage: raw CMI CSV -> per-sequence parquet + index + label encoder."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from bfrb_sensors.data.features import (
    angular_velocity_and_distance,
    remove_gravity_from_acc,
    tof_per_sensor_stats,
)
from bfrb_sensors.data.label_encoder import build_label_encoder

logger = logging.getLogger(__name__)

IMU_COLUMNS: tuple[str, ...] = (
    "acc_x",
    "acc_y",
    "acc_z",
    "rot_w",
    "rot_x",
    "rot_y",
    "rot_z",
)
THM_COLUMNS: tuple[str, ...] = tuple(f"thm_{i}" for i in range(1, 6))
TOF_SENSORS: int = 5
TOF_PIXELS: int = 64  # 8x8 grid per sensor
TOF_COLUMNS: tuple[str, ...] = tuple(
    f"tof_{sensor}_v{pixel}" for sensor in range(1, TOF_SENSORS + 1) for pixel in range(TOF_PIXELS)
)


@dataclass(frozen=True)
class PrepareConfig:
    raw_csv: Path
    prepared_dir: Path
    min_length: int = 10
    nan_threshold: float = 0.5
    verbose: bool = False
    sequence_id_col: str = "sequence_id"
    subject_id_col: str = "subject"
    gesture_col: str = "gesture"
    step_col: str = "sequence_counter"
    orientation_col: str = "orientation"
    sequence_type_col: str = "sequence_type"
    expected_n_classes: int | None = None
    gravity: float = 9.81
    sample_hz: float = 200.0
    quaternion_eps: float = 1e-8
    tof_missing_sentinel: float = -1.0


def _fill_nan(arr: np.ndarray) -> tuple[np.ndarray, int]:
    """Forward-fill then back-fill per channel; residual NaN -> 0.

    Returns (filled_array, residual_nan_count).
    """
    df = pd.DataFrame(arr)
    df = df.ffill().bfill()
    residual = int(df.isna().sum().sum())
    if residual:
        df = df.fillna(0.0)
    return df.to_numpy(dtype=np.float32), residual


def _reshape_tof(tof_flat: np.ndarray) -> np.ndarray:
    """Reshape (T, 320) -> (T, 5, 8, 8)."""
    timesteps = tof_flat.shape[0]
    return tof_flat.reshape(timesteps, TOF_SENSORS, 8, 8)


def _process_one_sequence(seq_df: pd.DataFrame, cfg: PrepareConfig) -> dict[str, object] | None:
    sequence_id = str(seq_df[cfg.sequence_id_col].iloc[0])
    subject_id = str(seq_df[cfg.subject_id_col].iloc[0])
    gesture = str(seq_df[cfg.gesture_col].iloc[0])
    orientation = str(seq_df[cfg.orientation_col].iloc[0])
    sequence_type = str(seq_df[cfg.sequence_type_col].iloc[0])
    length = len(seq_df)

    if length < cfg.min_length:
        logger.warning(
            "Dropping sequence %s (subject=%s): length %d < min_length %d",
            sequence_id,
            subject_id,
            length,
            cfg.min_length,
        )
        return None

    imu_raw = seq_df[list(IMU_COLUMNS)].to_numpy(dtype=np.float64)
    thm_raw = seq_df[list(THM_COLUMNS)].to_numpy(dtype=np.float64)
    tof_raw = seq_df[list(TOF_COLUMNS)].to_numpy(dtype=np.float64)

    frac_nan_imu = float(np.isnan(imu_raw).mean())
    frac_nan_thm = float(np.isnan(thm_raw).mean())
    frac_nan_tof = float(np.isnan(tof_raw).mean())

    if frac_nan_imu > cfg.nan_threshold:
        logger.warning(
            "Dropping sequence %s: IMU NaN fraction %.2f exceeds threshold %.2f",
            sequence_id,
            frac_nan_imu,
            cfg.nan_threshold,
        )
        return None

    has_thm = frac_nan_thm <= cfg.nan_threshold
    has_tof = frac_nan_tof <= cfg.nan_threshold

    imu, imu_residual = _fill_nan(imu_raw)
    if has_thm:
        thm, thm_residual = _fill_nan(thm_raw)
    else:
        thm = np.zeros_like(thm_raw, dtype=np.float32)
        thm_residual = 0
    if has_tof:
        tof_filled, tof_residual = _fill_nan(tof_raw)
    else:
        tof_filled = np.zeros_like(tof_raw, dtype=np.float32)
        tof_residual = 0
    tof = _reshape_tof(tof_filled)

    acc = imu[:, 0:3]
    quat = imu[:, [4, 5, 6, 3]]
    linear_acc = remove_gravity_from_acc(acc, quat, gravity=cfg.gravity, eps=cfg.quaternion_eps)
    angular_vel, angular_distance = angular_velocity_and_distance(
        quat, sample_hz=cfg.sample_hz, eps=cfg.quaternion_eps
    )
    imu_derived = np.concatenate([linear_acc, angular_vel, angular_distance[:, None]], axis=1)

    tof_stats_raw = tof_per_sensor_stats(
        tof_raw.reshape(length, TOF_SENSORS, 8, 8), sentinel=cfg.tof_missing_sentinel
    )
    tof_stats, _ = _fill_nan(tof_stats_raw)

    if imu_residual or thm_residual or tof_residual:
        logger.warning(
            "Sequence %s had residual NaN after ffill/bfill (imu=%d, thm=%d, tof=%d); zero-filled",
            sequence_id,
            imu_residual,
            thm_residual,
            tof_residual,
        )

    if cfg.verbose:
        logger.debug(
            "Sequence %s: T=%d, has_thm=%s, has_tof=%s, frac_nan=(imu=%.3f, thm=%.3f, tof=%.3f)",
            sequence_id,
            length,
            has_thm,
            has_tof,
            frac_nan_imu,
            frac_nan_thm,
            frac_nan_tof,
        )

    return {
        "sequence_id": sequence_id,
        "subject_id": subject_id,
        "gesture": gesture,
        "orientation": orientation,
        "sequence_type": sequence_type,
        "length": length,
        "has_thm": has_thm,
        "has_tof": has_tof,
        "frac_nan_imu": frac_nan_imu,
        "frac_nan_thm": frac_nan_thm,
        "frac_nan_tof": frac_nan_tof,
        "imu": imu.astype(np.float32),
        "thm": thm.astype(np.float32),
        "tof": tof.astype(np.float32),
        "imu_derived": imu_derived.astype(np.float32),
        "tof_stats": tof_stats.astype(np.float32),
    }


def _validate_columns(df: pd.DataFrame, cfg: PrepareConfig) -> None:
    required_meta = {
        cfg.sequence_id_col,
        cfg.subject_id_col,
        cfg.gesture_col,
        cfg.step_col,
        cfg.orientation_col,
        cfg.sequence_type_col,
    }
    missing_meta = required_meta - set(df.columns)
    if missing_meta:
        raise ValueError(f"raw CSV missing metadata columns: {sorted(missing_meta)}")

    missing_sensors = (set(IMU_COLUMNS) | set(THM_COLUMNS) | set(TOF_COLUMNS)) - set(df.columns)
    if missing_sensors:
        raise ValueError(
            f"raw CSV missing sensor columns ({len(missing_sensors)}): "
            f"e.g. {sorted(missing_sensors)[:5]}"
        )


def _write_sequence_file(record: dict[str, object], sequences_dir: Path) -> None:
    sequences_dir.mkdir(parents=True, exist_ok=True)
    path = sequences_dir / f"{record['sequence_id']}.parquet"
    imu = record["imu"]
    thm = record["thm"]
    tof = record["tof"]
    imu_derived = record["imu_derived"]
    tof_stats = record["tof_stats"]
    table = pa.table(
        {
            "imu": [imu.tolist()],
            "thm": [thm.tolist()],
            "tof": [tof.tolist()],
            "imu_derived": [imu_derived.tolist()],
            "tof_stats": [tof_stats.tolist()],
        }
    )
    pq.write_table(table, path)


def _write_index(records: list[dict[str, object]], prepared_dir: Path) -> None:
    index = pd.DataFrame(
        [
            {
                "sequence_id": r["sequence_id"],
                "subject_id": r["subject_id"],
                "gesture": r["gesture"],
                "orientation": r["orientation"],
                "sequence_type": r["sequence_type"],
                "length": r["length"],
                "has_thm": r["has_thm"],
                "has_tof": r["has_tof"],
                "frac_nan_imu": r["frac_nan_imu"],
                "frac_nan_thm": r["frac_nan_thm"],
                "frac_nan_tof": r["frac_nan_tof"],
            }
            for r in records
        ]
    )
    index_path = prepared_dir / "index.parquet"
    index.to_parquet(index_path, index=False)
    logger.info(
        "Wrote index with %d sequences to %s; n_subjects=%d, class_hist=%s",
        len(index),
        index_path,
        index["subject_id"].nunique(),
        dict(index["gesture"].value_counts().sort_index()),
    )
    p50, p95, p99 = np.percentile(index["length"], [50, 95, 99])
    logger.info("Sequence length percentiles: p50=%.0f, p95=%.0f, p99=%.0f", p50, p95, p99)
    logger.info(
        "Modality availability: has_thm=%d/%d, has_tof=%d/%d",
        int(index["has_thm"].sum()),
        len(index),
        int(index["has_tof"].sum()),
        len(index),
    )


def prepare(cfg: PrepareConfig) -> None:
    start = time.perf_counter()
    raw_csv = Path(cfg.raw_csv)
    prepared_dir = Path(cfg.prepared_dir)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    sequences_dir = prepared_dir / "sequences"

    logger.info("Reading raw CSV: %s", raw_csv)
    if not raw_csv.exists():
        raise FileNotFoundError(
            f"raw CSV not found at {raw_csv}. "
            "Run `bfrb download` (or `dvc pull -r bfrb-data`) first."
        )

    df = pd.read_csv(raw_csv)
    _validate_columns(df, cfg)
    logger.info("Loaded %d raw rows from CSV", len(df))

    records: list[dict[str, object]] = []
    for _, seq_df in df.groupby(cfg.sequence_id_col, sort=False):
        seq_df = seq_df.sort_values(cfg.step_col).reset_index(drop=True)
        record = _process_one_sequence(seq_df, cfg)
        if record is None:
            continue
        _write_sequence_file(record, sequences_dir)
        records.append(record)

    if not records:
        raise RuntimeError("prepare produced zero sequences; check input data and config")

    _write_index(records, prepared_dir)

    encoder = build_label_encoder(r["gesture"] for r in records)
    if cfg.expected_n_classes is not None and encoder.n_classes != cfg.expected_n_classes:
        raise ValueError(
            f"expected {cfg.expected_n_classes} gesture classes but prepare saw "
            f"{encoder.n_classes}; check the raw CSV read and gesture column"
        )
    encoder.save(prepared_dir / "label_encoder.json")

    elapsed = time.perf_counter() - start
    logger.info("Prepare stage finished in %.2fs; %d sequences written", elapsed, len(records))
