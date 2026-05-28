from __future__ import annotations

import pandas as pd
import pytest
import torch

from bfrb_sensors.data.label_encoder import build_label_encoder
from bfrb_sensors.training.metrics import HierarchyMapping


def test_hierarchy_mapping_transforms_labels():
    index = pd.DataFrame(
        [
            {"gesture": "target_a", "sequence_type": "Target"},
            {"gesture": "target_b", "sequence_type": "Target"},
            {"gesture": "non_a", "sequence_type": "Non-Target"},
        ]
    )
    encoder = build_label_encoder(["target_a", "target_b", "non_a"])
    mapping = HierarchyMapping.from_index(index, encoder)
    labels = torch.tensor([encoder.encode("target_a"), encoder.encode("non_a")])

    assert mapping.to_binary(labels).tolist() == [1, 0]
    assert mapping.to_collapsed(labels).tolist() == [
        mapping.collapsed_id_by_original[int(labels[0])],
        mapping.non_target_collapsed_id,
    ]


def test_hierarchy_mapping_rejects_ambiguous_gesture_type():
    index = pd.DataFrame(
        [
            {"gesture": "same", "sequence_type": "Target"},
            {"gesture": "same", "sequence_type": "Non-Target"},
        ]
    )
    encoder = build_label_encoder(["same"])

    with pytest.raises(ValueError, match="multiple sequence_type"):
        HierarchyMapping.from_index(index, encoder)
