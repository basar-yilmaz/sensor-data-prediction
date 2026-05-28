"""Deterministic, pure feature derivations from raw sensor channels.

No labels, no train/val split — safe to materialize in the prepare stage.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])  # scalar-last [x, y, z, w]


def normalize_quaternions(quat: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """Normalize scalar-last quaternions and flag validity.

    quat: (N, 4) as [x, y, z, w]. A row is valid iff all components are finite and
    its norm exceeds ``eps``. Invalid rows are returned as the identity quaternion
    so SciPy never sees a bad value; callers gate on the returned mask.
    Returns (unit_quaternions (N,4), valid_mask (N,)).
    """
    quat = np.asarray(quat, dtype=np.float64)
    norms = np.linalg.norm(quat, axis=1)
    valid = np.isfinite(quat).all(axis=1) & (norms > eps)
    unit = np.tile(IDENTITY_QUAT, (quat.shape[0], 1))
    unit[valid] = quat[valid] / norms[valid, None]
    return unit, valid


def remove_gravity_from_acc(
    acc: np.ndarray, quat: np.ndarray, gravity: float = 9.81, eps: float = 1e-8
) -> np.ndarray:
    """Linear acceleration = acc - R^-1 . g_world, per timestep.

    acc:  (T, 3) acc_x, acc_y, acc_z.
    quat: (T, 4) scalar-last [rot_x, rot_y, rot_z, rot_w].
    Invalid quaternion rows fall back to raw acc (no fabrication).
    Returns (T, 3) float32.
    """
    acc = np.asarray(acc, dtype=np.float64)
    unit, valid = normalize_quaternions(quat, eps=eps)
    linear = acc.copy()
    if valid.any():
        rot = Rotation.from_quat(unit[valid])
        g_world = np.tile(np.array([0.0, 0.0, gravity]), (int(valid.sum()), 1))
        g_sensor = rot.apply(g_world, inverse=True)
        linear[valid] = acc[valid] - g_sensor
    return linear.astype(np.float32)
