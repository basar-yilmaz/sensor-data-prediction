"""Collation utilities for BFRB sensor batches."""

from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence


def pad_collate(batch: list[dict]) -> dict:
    """Pad variable-length sensor samples into a batch."""
    if not batch:
        raise ValueError("empty batch")

    lengths = torch.stack([sample["length"] for sample in batch])
    max_length = int(lengths.max().item())

    tof = batch[0]["tof"].new_zeros((len(batch), max_length, 5, 8, 8))
    for i, sample in enumerate(batch):
        length = int(sample["length"].item())
        tof[i, :length] = sample["tof"]

    timesteps = torch.arange(max_length, device=lengths.device)

    collated = {
        "imu": pad_sequence([sample["imu"] for sample in batch], batch_first=True),
        "imu_derived": pad_sequence([sample["imu_derived"] for sample in batch], batch_first=True),
        "thm": pad_sequence([sample["thm"] for sample in batch], batch_first=True),
        "tof": tof,
        "tof_stats": pad_sequence([sample["tof_stats"] for sample in batch], batch_first=True),
        "label": torch.stack([sample["label"] for sample in batch]),
        "has_thm": torch.stack([sample["has_thm"] for sample in batch]),
        "has_tof": torch.stack([sample["has_tof"] for sample in batch]),
        "length": lengths,
        "attention_mask": timesteps.unsqueeze(0) < lengths.unsqueeze(1),
    }
    if "demographics" in batch[0]:
        collated["demographics"] = torch.stack([sample["demographics"] for sample in batch])
    return collated


class ModalityDropout:
    """Randomly drop non-IMU modalities from padded batches."""

    def __init__(self, p_thm: float, p_tof: float, generator_seed: int | None = None) -> None:
        if not 0.0 <= p_thm <= 1.0 or not 0.0 <= p_tof <= 1.0:
            raise ValueError("probabilities must be between 0 and 1")

        self.p_thm = p_thm
        self.p_tof = p_tof
        self.generator = None
        if generator_seed is not None:
            self.generator = torch.Generator().manual_seed(generator_seed)

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self._drop_modality(batch, "thm", "has_thm", self.p_thm)
        dropped_tof = self._drop_modality(batch, "tof", "has_tof", self.p_tof)
        batch["tof_stats"][dropped_tof] = 0
        return batch

    def _drop_modality(
        self, batch: dict[str, torch.Tensor], key: str, flag_key: str, p: float
    ) -> torch.Tensor:
        dropped = (
            torch.rand(
                batch[flag_key].shape,
                device=batch[flag_key].device,
                generator=self.generator,
            )
            < p
        )
        newly_dropped = dropped & batch[flag_key]
        batch[key][dropped] = 0
        batch[flag_key][dropped] = False
        return newly_dropped
