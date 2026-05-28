"""Unit tests for deterministic feature derivations."""

from __future__ import annotations

import numpy as np

from bfrb_sensors.data.features import (
    angular_velocity_and_distance,
    normalize_quaternions,
    remove_gravity_from_acc,
    tof_per_sensor_stats,
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


def _yaw_quat(angle_rad: float) -> list[float]:
    return [0.0, 0.0, np.sin(angle_rad / 2.0), np.cos(angle_rad / 2.0)]


def test_angular_features_constant_orientation_are_zero():
    quat = np.array([[0.0, 0.0, 0.0, 1.0]] * 4)
    vel, dist = angular_velocity_and_distance(quat, sample_hz=200.0)
    np.testing.assert_allclose(vel, np.zeros((4, 3)), atol=1e-6)
    np.testing.assert_allclose(dist, np.zeros(4), atol=1e-6)


def test_angular_features_known_yaw_step():
    quat = np.array([_yaw_quat(0.0), _yaw_quat(np.pi / 2.0)])
    vel, dist = angular_velocity_and_distance(quat, sample_hz=200.0)
    np.testing.assert_allclose(dist, [np.pi / 2.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(vel[0], [0.0, 0.0, (np.pi / 2.0) * 200.0], atol=1e-4)
    np.testing.assert_allclose(vel[1], [0.0, 0.0, 0.0], atol=1e-6)


def test_angular_features_invalid_transition_is_zero():
    quat = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    vel, dist = angular_velocity_and_distance(quat, sample_hz=200.0)
    np.testing.assert_allclose(vel, np.zeros((2, 3)), atol=1e-6)
    np.testing.assert_allclose(dist, np.zeros(2), atol=1e-6)


def _grid_from_sensor_rows(rows_per_sensor: list[np.ndarray]) -> np.ndarray:
    timesteps = rows_per_sensor[0].shape[0]
    grid = np.stack(rows_per_sensor, axis=1)
    return grid.reshape(timesteps, 5, 8, 8)


def test_tof_stats_shape_and_order():
    grid = np.zeros((3, 5, 8, 8), dtype=np.float64)
    stats = tof_per_sensor_stats(grid, sentinel=-1.0)
    assert stats.shape == (3, 20)


def test_tof_stats_single_valid_pixel_among_sentinels():
    sensors = []
    sensor_1 = np.full((1, 64), -1.0)
    sensor_1[0, 0] = 7.0
    sensors.append(sensor_1)
    for _ in range(4):
        sensors.append(np.full((1, 64), -1.0))

    grid = _grid_from_sensor_rows(sensors)
    stats = tof_per_sensor_stats(grid, sentinel=-1.0)

    np.testing.assert_allclose(stats[0, 0:4], [7.0, 0.0, 7.0, 7.0], atol=1e-6)
    assert np.isnan(stats[0, 4:8]).all()


def test_tof_stats_no_runtime_warnings_on_all_missing(recwarn):
    grid = np.full((2, 5, 8, 8), -1.0)
    stats = tof_per_sensor_stats(grid, sentinel=-1.0)
    assert np.isnan(stats).all()
    assert len(recwarn) == 0
