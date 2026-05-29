"""Tests for modality dropout augmentation."""

from __future__ import annotations

import pytest
import torch

from bfrb_sensors.data.collate import ModalityDropout, pad_collate


def _sample(length: int, label: int) -> dict[str, torch.Tensor]:
    return {
        "imu": torch.full((length, 7), float(label + 1)),
        "thm": torch.full((length, 5), float(label + 10)),
        "tof": torch.full((length, 5, 8, 8), float(label + 100)),
        "label": torch.tensor(label, dtype=torch.long),
        "has_thm": torch.tensor(True, dtype=torch.bool),
        "has_tof": torch.tensor(True, dtype=torch.bool),
        "length": torch.tensor(length, dtype=torch.long),
    }


def _batch(size: int = 4) -> dict[str, torch.Tensor]:
    return pad_collate([_sample(2, i) for i in range(size)])


def test_p_one_zeroes_modalities_and_flips_flags():
    batch = _batch()

    result = ModalityDropout(p_thm=1.0, p_tof=1.0, generator_seed=0)(batch)

    assert torch.count_nonzero(result["thm"]) == 0
    assert torch.count_nonzero(result["tof"]) == 0
    torch.testing.assert_close(result["has_thm"], torch.zeros(4, dtype=torch.bool))
    torch.testing.assert_close(result["has_tof"], torch.zeros(4, dtype=torch.bool))


def test_p_zero_is_no_op():
    batch = _batch()
    expected = {key: value.clone() for key, value in batch.items()}

    result = ModalityDropout(p_thm=0.0, p_tof=0.0, generator_seed=0)(batch)

    for key, value in expected.items():
        torch.testing.assert_close(result[key], value)


def test_imu_is_never_touched():
    batch = _batch()
    expected_imu = batch["imu"].clone()

    result = ModalityDropout(p_thm=1.0, p_tof=1.0, generator_seed=0)(batch)

    torch.testing.assert_close(result["imu"], expected_imu)


def test_observed_frequency_matches_probability():
    batch = _batch(size=1000)

    result = ModalityDropout(p_thm=0.5, p_tof=0.5, generator_seed=123)(batch)

    thm_drop_rate = (~result["has_thm"]).float().mean().item()
    tof_drop_rate = (~result["has_tof"]).float().mean().item()

    assert 0.45 <= thm_drop_rate <= 0.55
    assert 0.45 <= tof_drop_rate <= 0.55


def test_outcomes_are_independent_across_samples():
    batch = _batch(size=32)

    result = ModalityDropout(p_thm=0.5, p_tof=0.0, generator_seed=0)(batch)

    kept = result["has_thm"].sum().item()

    assert 0 < kept < len(result["has_thm"])


@pytest.mark.parametrize(
    ("p_thm", "p_tof"),
    [(-0.1, 0.0), (1.1, 0.0), (0.0, -0.1), (0.0, 1.1)],
)
def test_probabilities_must_be_between_zero_and_one(p_thm: float, p_tof: float):
    with pytest.raises(ValueError, match="probabilities"):
        ModalityDropout(p_thm=p_thm, p_tof=p_tof)
