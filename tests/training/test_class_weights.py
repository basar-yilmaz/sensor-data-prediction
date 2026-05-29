from __future__ import annotations

import math

import pandas as pd
import torch

from bfrb_sensors.data.label_encoder import build_label_encoder
from bfrb_sensors.training.class_weights import compute_class_weights


def _index() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"sequence_id": "s1", "gesture": "a"},
            {"sequence_id": "s2", "gesture": "a"},
            {"sequence_id": "s3", "gesture": "a"},
            {"sequence_id": "s4", "gesture": "b"},
        ]
    )


def test_none_scheme_returns_none():
    encoder = build_label_encoder(["a", "b"])
    weights = compute_class_weights(
        _index(), encoder, ["s1", "s2", "s3", "s4"], scheme="none", num_classes=2
    )
    assert weights is None


def test_sqrt_inv_freq_upweights_rare_class():
    encoder = build_label_encoder(["a", "b"])  # a->0, b->1
    weights = compute_class_weights(
        _index(), encoder, ["s1", "s2", "s3", "s4"], scheme="sqrt_inv_freq", num_classes=2
    )
    assert weights is not None
    assert weights.shape == (2,)
    assert weights[1] > weights[0]
    assert math.isclose(float(weights.mean()), 1.0, rel_tol=1e-5)


def test_only_train_sequences_count():
    encoder = build_label_encoder(["a", "b"])
    weights = compute_class_weights(
        _index(), encoder, ["s1", "s2", "s3"], scheme="sqrt_inv_freq", num_classes=2
    )
    assert torch.isfinite(weights).all()
