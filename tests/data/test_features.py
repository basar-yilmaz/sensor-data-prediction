"""Unit tests for deterministic feature derivations."""

from __future__ import annotations

import numpy as np

from bfrb_sensors.data.features import (
    normalize_quaternions,
    remove_gravity_from_acc,
)

GRAVITY = 9.81


def test_normalize_marks_zero_and_nan_invalid():
    quat = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],  # identity, valid
            [0.0, 0.0, 0.0, 0.0],  # zero, invalid
            [np.nan, 0.0, 0.0, 1.0],  # nan, invalid
            [0.0, 0.0, 0.0, 2.0],  # non-unit, valid -> normalizes to identity
        ]
    )
    unit, valid = normalize_quaternions(quat, eps=1e-8)
    assert valid.tolist() == [True, False, False, True]
    np.testing.assert_allclose(unit[0], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(unit[3], [0.0, 0.0, 0.0, 1.0])  # normalized
    assert np.isclose(np.linalg.norm(unit[3]), 1.0)


def test_gravity_removal_identity_orientation():
    acc = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    quat = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
    linear = remove_gravity_from_acc(acc, quat, gravity=GRAVITY)
    np.testing.assert_allclose(linear, [[1.0, 2.0, 3.0 - GRAVITY], [0.0, 0.0, -GRAVITY]], atol=1e-5)


def test_gravity_removal_invalid_quaternion_falls_back_to_raw_acc():
    acc = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    quat = np.array([[0.0, 0.0, 0.0, 0.0], [np.nan, 0.0, 0.0, 1.0]])
    linear = remove_gravity_from_acc(acc, quat, gravity=GRAVITY)
    np.testing.assert_allclose(linear, acc, atol=1e-6)
