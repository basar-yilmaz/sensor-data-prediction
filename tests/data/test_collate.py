"""Tests for batch collation utilities."""

from __future__ import annotations

import pytest
import torch

from bfrb_sensors.data.collate import pad_collate


def _sample(length: int, label: int, has_thm: bool, has_tof: bool) -> dict[str, torch.Tensor]:
    return {
        "imu": torch.full((length, 7), float(label + 1)),
        "imu_derived": torch.full((length, 7), float(label + 2)),
        "thm": torch.full((length, 5), float(label + 10)),
        "tof": torch.full((length, 5, 8, 8), float(label + 100)),
        "tof_stats": torch.full((length, 20), float(label + 200)),
        "label": torch.tensor(label, dtype=torch.long),
        "has_thm": torch.tensor(has_thm, dtype=torch.bool),
        "has_tof": torch.tensor(has_tof, dtype=torch.bool),
        "length": torch.tensor(length, dtype=torch.long),
    }


def test_pad_collate_pads_modalities_to_max_length():
    batch = pad_collate([_sample(2, 0, True, False), _sample(4, 1, False, True)])

    assert batch["imu"].shape == (2, 4, 7)
    assert batch["imu_derived"].shape == (2, 4, 7)
    assert batch["thm"].shape == (2, 4, 5)
    assert batch["tof"].shape == (2, 4, 5, 8, 8)
    assert batch["tof_stats"].shape == (2, 4, 20)


def test_pad_collate_builds_attention_mask_from_lengths():
    batch = pad_collate([_sample(2, 0, True, True), _sample(4, 1, True, True)])

    expected = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, True],
        ],
        dtype=torch.bool,
    )
    torch.testing.assert_close(batch["attention_mask"], expected)


def test_pad_collate_zero_fills_padding_regions():
    batch = pad_collate([_sample(2, 0, True, True), _sample(4, 1, True, True)])

    assert torch.count_nonzero(batch["imu"][0, 2:]) == 0
    assert torch.count_nonzero(batch["imu_derived"][0, 2:]) == 0
    assert torch.count_nonzero(batch["thm"][0, 2:]) == 0
    assert torch.count_nonzero(batch["tof"][0, 2:]) == 0
    assert torch.count_nonzero(batch["tof_stats"][0, 2:]) == 0


def test_pad_collate_preserves_labels_lengths_and_modality_flags():
    batch = pad_collate([_sample(2, 3, True, False), _sample(4, 5, False, True)])

    torch.testing.assert_close(batch["label"], torch.tensor([3, 5], dtype=torch.long))
    torch.testing.assert_close(batch["length"], torch.tensor([2, 4], dtype=torch.long))
    torch.testing.assert_close(batch["has_thm"], torch.tensor([True, False], dtype=torch.bool))
    torch.testing.assert_close(batch["has_tof"], torch.tensor([False, True], dtype=torch.bool))


def test_pad_collate_rejects_empty_batch():
    with pytest.raises(ValueError, match="empty batch"):
        pad_collate([])
