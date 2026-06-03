"""CSV -> model-ready tensor pipeline for the deploy service.

Self-contained NumPy/SciPy implementation that mirrors the training-side
``bfrb_sensors.data.prepare`` + ``bfrb_sensors.data.dataset`` behavior. Kept
in this package so the deploy service can run without importing the
training pipeline; the only shared artifact is the fitted scaler on disk
(``artifacts/scaler.joblib``).

Accepted CSV format: one row per timestep, with at least the columns
``sequence_id``, ``sequence_counter``, ``acc_{x,y,z}``, ``rot_{w,x,y,z}``,
``thm_{1..5}``, and ``tof_{1..5}_v{0..63}``. Extra metadata columns
(``gesture``, ``subject``, ``orientation``, ...) are ignored.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation

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
TOF_COLUMNS: tuple[str, ...] = tuple(
    f"tof_{sensor}_v{pixel}" for sensor in range(1, 6) for pixel in range(64)
)
META_COLUMNS: tuple[str, ...] = ("sequence_id", "sequence_counter")
REQUIRED_COLUMNS: tuple[str, ...] = META_COLUMNS + IMU_COLUMNS + THM_COLUMNS + TOF_COLUMNS

TOF_MISSING_SENTINEL: float = -1.0
GRAVITY: float = 9.81
SAMPLE_HZ: float = 200.0
QUAT_EPS: float = 1e-8
IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)  # scalar-last [x, y, z, w]


class PreprocessingError(ValueError):
    """Raised when a CSV cannot be turned into a model-ready tensor.

    The ``code`` attribute is surfaced to API consumers via the error handler.
    """

    def __init__(self, code: str, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


@dataclass(frozen=True)
class RawSequence:
    sequence_id: str
    imu: np.ndarray
    thm: np.ndarray
    tof: np.ndarray
    has_thm: bool
    has_tof: bool

    @property
    def length(self) -> int:
        return int(self.imu.shape[0])


@dataclass(frozen=True)
class FeaturizedSequence:
    sequence_id: str
    imu: np.ndarray
    imu_derived: np.ndarray
    thm: np.ndarray
    tof_stats: np.ndarray
    tof: np.ndarray | None
    has_thm: bool
    has_tof: bool


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        example = missing[:5]
        raise PreprocessingError(
            code="missing_columns",
            message=f"CSV is missing {len(missing)} required column(s); first few: {example}",
            details={"missing": missing, "example": example},
        )


def _frac_nan(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 1.0
    return float(np.isnan(arr).mean())


def _fill_nan(arr: np.ndarray) -> np.ndarray:
    """Forward-fill then back-fill per channel; residual NaN -> 0.0."""
    if not np.isnan(arr).any():
        return arr.astype(np.float32, copy=False)
    df = pd.DataFrame(arr)
    df = df.ffill().bfill().fillna(0.0)
    return df.to_numpy(dtype=np.float32)


def parse_csv(
    file_bytes: bytes,
    *,
    min_seq_length: int = 10,
    nan_threshold: float = 0.5,
) -> list[RawSequence]:
    """Parse a CSV byte string into a list of per-sequence raw arrays.

    Sequences shorter than ``min_seq_length`` or with more than
    ``nan_threshold`` NaN in the IMU channels are dropped (mirrors the
    training-side ``prepare._process_one_sequence`` behavior).

    THM/ToF availability is decided by the fraction of *sentinel* values
    (NaN for THM, ``-1`` for ToF) rather than NaN alone; the raw CSV can
    encode "this modality is absent" with the sentinel value rather than
    with NaN, and we want to match the training-side semantics.
    """
    if not file_bytes:
        raise PreprocessingError("empty_file", "uploaded file is empty")
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as exc:
        raise PreprocessingError("invalid_csv", f"could not read CSV: {exc}") from exc
    if df.empty:
        raise PreprocessingError("empty_file", "CSV contains no rows")
    _validate_columns(df)

    sequences: list[RawSequence] = []
    for sequence_id, seq_df in df.groupby("sequence_id", sort=False):
        seq_df = seq_df.sort_values("sequence_counter").reset_index(drop=True)
        length = len(seq_df)
        if length < min_seq_length:
            continue
        imu = seq_df[list(IMU_COLUMNS)].to_numpy(dtype=np.float64)
        thm = seq_df[list(THM_COLUMNS)].to_numpy(dtype=np.float64)
        tof = seq_df[list(TOF_COLUMNS)].to_numpy(dtype=np.float64)

        if _frac_nan(imu) > nan_threshold:
            continue
        has_thm = _frac_nan(thm) <= nan_threshold
        # ToF uses -1 as the explicit "missing" sentinel (matches the
        # training-side prepare stage and the official CMI dataset).
        tof_missing_mask = (tof == TOF_MISSING_SENTINEL) | np.isnan(tof)
        has_tof = float(tof_missing_mask.mean()) <= nan_threshold

        imu_filled = _fill_nan(imu)
        thm_filled = _fill_nan(thm) if has_thm else np.zeros_like(thm, dtype=np.float32)
        tof_filled = _fill_nan(tof) if has_tof else np.zeros_like(tof, dtype=np.float32)

        sequences.append(
            RawSequence(
                sequence_id=str(sequence_id),
                imu=imu_filled.astype(np.float32),
                thm=thm_filled.astype(np.float32),
                tof=tof_filled.reshape(length, 5, 8, 8).astype(np.float32),
                has_thm=has_thm,
                has_tof=has_tof,
            )
        )
    if not sequences:
        raise PreprocessingError(
            "no_sequences",
            "no valid sequences found (check min length and NaN thresholds)",
            details={"n_rows": len(df)},
        )
    return sequences


def _normalize_quaternions(quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """quat (T, 4) scalar-last [x, y, z, w] -> (unit_quat (T, 4), valid (T,))."""
    quat = np.asarray(quat, dtype=np.float64)
    norms = np.linalg.norm(quat, axis=1)
    valid = np.isfinite(quat).all(axis=1) & (norms > QUAT_EPS)
    unit = np.tile(IDENTITY_QUAT, (quat.shape[0], 1))
    unit[valid] = quat[valid] / norms[valid, None]
    return unit, valid


def _remove_gravity_from_acc(acc: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """acc (T, 3), quat_xyzw (T, 4) scalar-last. Returns (T, 3) float32."""
    unit, valid = _normalize_quaternions(quat_xyzw)
    linear = acc.astype(np.float64, copy=True)
    if valid.any():
        rot = Rotation.from_quat(unit[valid])
        g_world = np.tile(np.array([0.0, 0.0, GRAVITY]), (int(valid.sum()), 1))
        g_sensor = rot.apply(g_world, inverse=True)
        linear[valid] = acc[valid].astype(np.float64) - g_sensor
    return linear.astype(np.float32)


def _angular_velocity_and_distance(
    quat_xyzw: np.ndarray, sample_hz: float = SAMPLE_HZ
) -> tuple[np.ndarray, np.ndarray]:
    """quat_xyzw (T, 4) scalar-last. Returns (ang_vel (T,3) float32, ang_dist (T,) float32)."""
    unit, valid = _normalize_quaternions(quat_xyzw)
    n = unit.shape[0]
    ang_vel = np.zeros((n, 3), dtype=np.float64)
    ang_dist = np.zeros(n, dtype=np.float64)
    if n >= 2:
        dt = 1.0 / sample_hz
        pair_valid = valid[:-1] & valid[1:]
        if pair_valid.any():
            r_t = Rotation.from_quat(unit[:-1][pair_valid])
            r_t1 = Rotation.from_quat(unit[1:][pair_valid])
            rotvec = (r_t.inv() * r_t1).as_rotvec()
            idx = np.flatnonzero(pair_valid)
            ang_vel[idx] = rotvec / dt
            ang_dist[idx] = np.linalg.norm(rotvec, axis=1)
    return ang_vel.astype(np.float32), ang_dist.astype(np.float32)


def _tof_per_sensor_stats(
    tof_grid: np.ndarray, sentinel: float = TOF_MISSING_SENTINEL
) -> np.ndarray:
    """tof_grid (T, 5, 8, 8) -> (T, 20) float32: per-sensor mean/std/min/max."""
    tof = np.asarray(tof_grid, dtype=np.float64)
    timesteps = tof.shape[0]
    flat = tof.reshape(timesteps, 5, 64)
    valid = (flat != sentinel) & np.isfinite(flat)
    counts = valid.sum(axis=2)
    has_values = counts > 0
    safe = np.where(valid, flat, 0.0)
    total = safe.sum(axis=2)
    mean = np.divide(total, counts, out=np.full_like(total, np.nan), where=has_values)
    total_sq = (safe * safe).sum(axis=2)
    mean_sq = np.divide(total_sq, counts, out=np.full_like(total_sq, np.nan), where=has_values)
    std = np.sqrt(np.clip(mean_sq - mean * mean, a_min=0.0, a_max=None))
    minima = np.where(has_values, np.where(valid, flat, np.inf).min(axis=2), np.nan)
    maxima = np.where(has_values, np.where(valid, flat, -np.inf).max(axis=2), np.nan)
    stats = np.stack([mean, std, minima, maxima], axis=2)
    return stats.reshape(timesteps, 20).astype(np.float32)


def featurize(sequence: RawSequence, scaler: dict[str, np.ndarray]) -> FeaturizedSequence:
    """Build IMU/IMU-derived/THM/ToF-stats arrays and z-score IMU+THM.

    The raw ToF grid is kept (filled) and not z-scored; only the ToF stats
    are concatenated to the per-timestep vector (ToF stats stay in native
    scale too, matching training).
    """
    imu = sequence.imu.astype(np.float64)
    acc = imu[:, 0:3]
    quat_xyzw = imu[:, [4, 5, 6, 3]]  # rot_w, rot_x, rot_y, rot_z -> x, y, z, w
    linear_acc = _remove_gravity_from_acc(acc, quat_xyzw)
    ang_vel, ang_dist = _angular_velocity_and_distance(quat_xyzw)
    imu_derived = np.concatenate(
        [linear_acc, ang_vel, ang_dist[:, None].astype(np.float32)], axis=1
    ).astype(np.float32)

    tof_stats = _tof_per_sensor_stats(sequence.tof, sentinel=TOF_MISSING_SENTINEL)
    tof_stats = np.nan_to_num(tof_stats, nan=0.0, posinf=0.0, neginf=0.0)

    imu_norm = (imu.astype(np.float32) - scaler["imu_mean"]) / scaler["imu_std"]
    thm_norm = (sequence.thm - scaler["thm_mean"]) / scaler["thm_std"]

    return FeaturizedSequence(
        sequence_id=sequence.sequence_id,
        imu=imu_norm.astype(np.float32),
        imu_derived=imu_derived,
        thm=thm_norm.astype(np.float32),
        tof_stats=tof_stats,
        tof=sequence.tof if sequence.has_tof else None,
        has_thm=sequence.has_thm,
        has_tof=sequence.has_tof,
    )


def _pad_to_batch(
    arrays: list[np.ndarray], pad_value: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Right-pad a list of (T, C) arrays to the longest in the batch.

    Returns (padded (B, T_max, C), lengths (B,)).
    """
    if not arrays:
        raise ValueError("cannot pad an empty list")
    lengths = np.array([a.shape[0] for a in arrays], dtype=np.int64)
    max_len = int(lengths.max())
    channels = arrays[0].shape[1]
    padded = np.full((len(arrays), max_len, channels), pad_value, dtype=np.float32)
    for i, arr in enumerate(arrays):
        padded[i, : arr.shape[0]] = arr
    return padded, lengths


def _pad_tof(arrays: list[np.ndarray | None], max_len: int) -> np.ndarray | None:
    """Pad (B, T, 5, 8, 8) with zeros; returns None if every input is None."""
    if all(a is None for a in arrays):
        return None
    out = np.zeros((len(arrays), max_len, 5, 8, 8), dtype=np.float32)
    for i, arr in enumerate(arrays):
        if arr is None:
            continue
        out[i, : arr.shape[0]] = arr
    return out


def collate(
    featurized: Sequence[FeaturizedSequence],
    *,
    max_seq_length: int | None = None,
) -> dict[str, torch.Tensor]:
    """Collate one or more featurized sequences into a model-ready batch.

    Caps sequence length to ``max_seq_length`` (drops excess tail) to prevent
    OOM on adversarial uploads. Truncation uses the most recent timesteps,
    which is more informative for transient gestures.
    """
    if not featurized:
        raise ValueError("collate requires at least one sequence")

    truncated: list[FeaturizedSequence] = []
    for seq in featurized:
        if max_seq_length is not None and seq.imu.shape[0] > max_seq_length:
            tail = seq.imu.shape[0] - max_seq_length
            truncated.append(
                FeaturizedSequence(
                    sequence_id=seq.sequence_id,
                    imu=seq.imu[tail:],
                    imu_derived=seq.imu_derived[tail:],
                    thm=seq.thm[tail:],
                    tof_stats=seq.tof_stats[tail:],
                    tof=seq.tof[tail:] if seq.tof is not None else None,
                    has_thm=seq.has_thm,
                    has_tof=seq.has_tof,
                )
            )
        else:
            truncated.append(seq)
    featurized = truncated

    imu, _ = _pad_to_batch([s.imu for s in featurized])
    imu_derived, _ = _pad_to_batch([s.imu_derived for s in featurized])
    thm, _ = _pad_to_batch([s.thm for s in featurized])
    tof_stats, lengths = _pad_to_batch([s.tof_stats for s in featurized])
    max_len = int(lengths.max())
    tof = _pad_tof([s.tof for s in featurized], max_len)

    timesteps = torch.arange(max_len)
    attention_mask = timesteps.unsqueeze(0) < torch.as_tensor(lengths).unsqueeze(1)

    batch: dict[str, torch.Tensor] = {
        "imu": torch.as_tensor(imu, dtype=torch.float32),
        "imu_derived": torch.as_tensor(imu_derived, dtype=torch.float32),
        "thm": torch.as_tensor(thm, dtype=torch.float32),
        "tof_stats": torch.as_tensor(tof_stats, dtype=torch.float32),
        "attention_mask": attention_mask,
        "length": torch.as_tensor(lengths, dtype=torch.long),
    }
    if tof is not None:
        batch["tof"] = torch.as_tensor(tof, dtype=torch.float32)
    return batch


def featurize_and_collate(
    raw_sequences: Iterable[RawSequence],
    scaler: dict[str, np.ndarray],
    *,
    max_seq_length: int | None = None,
) -> dict[str, torch.Tensor]:
    raw_sequences = list(raw_sequences)
    featurized = [featurize(s, scaler) for s in raw_sequences]
    return collate(featurized, max_seq_length=max_seq_length)
