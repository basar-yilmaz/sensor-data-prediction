"""Class-imbalance weights computed from training-fold label counts."""

from __future__ import annotations

import logging

import pandas as pd
import torch

from bfrb_sensors.data.label_encoder import LabelEncoder

logger = logging.getLogger(__name__)

_SCHEMES = ("none", "sqrt_inv_freq")


def compute_class_weights(
    index: pd.DataFrame,
    label_encoder: LabelEncoder,
    train_sequence_ids: list[str],
    scheme: str,
    num_classes: int,
) -> torch.Tensor | None:
    """Return per-class CE weights for ``scheme``, or ``None`` for ``"none"``.

    ``sqrt_inv_freq``: weight_c = sqrt(total / count_c), normalized to mean 1, so
    rarer classes are up-weighted but the overall loss scale is preserved. Counts
    are taken only over ``train_sequence_ids`` to avoid leaking validation labels.
    """
    if scheme == "none":
        return None
    if scheme not in _SCHEMES:
        raise ValueError(f"unknown class_weighting scheme {scheme!r}; expected one of {_SCHEMES}")

    train = index[index["sequence_id"].isin(set(train_sequence_ids))]
    counts = torch.zeros(num_classes, dtype=torch.float)
    for gesture in train["gesture"].astype(str):
        counts[label_encoder.encode(gesture)] += 1.0
    counts = counts.clamp_min(1.0)

    weights = (counts.sum() / counts).sqrt()
    weights = weights * num_classes / weights.sum()  # mean 1
    logger.info(
        "Computed sqrt_inv_freq class weights (min=%.3f, max=%.3f)",
        float(weights.min()),
        float(weights.max()),
    )
    return weights
