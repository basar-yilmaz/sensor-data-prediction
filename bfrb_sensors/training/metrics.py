"""Competition-like hierarchical metric target transforms."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch

from bfrb_sensors.data.label_encoder import LabelEncoder


@dataclass(frozen=True)
class HierarchyMapping:
    target_class_ids: tuple[int, ...]
    non_target_class_ids: tuple[int, ...]
    collapsed_id_by_original: dict[int, int]
    non_target_collapsed_id: int

    @classmethod
    def from_index(cls, index: pd.DataFrame, label_encoder: LabelEncoder) -> HierarchyMapping:
        by_gesture = index.groupby("gesture")["sequence_type"].nunique()
        ambiguous = by_gesture[by_gesture > 1]
        if not ambiguous.empty:
            raise ValueError(
                f"gestures have multiple sequence_type values: {ambiguous.index.tolist()}"
            )

        pairs = index[["gesture", "sequence_type"]].drop_duplicates()
        target_ids: list[int] = []
        non_target_ids: list[int] = []
        for row in pairs.itertuples(index=False):
            class_id = label_encoder.encode(str(row.gesture))
            if str(row.sequence_type) == "Target":
                target_ids.append(class_id)
            elif str(row.sequence_type) == "Non-Target":
                non_target_ids.append(class_id)
            else:
                raise ValueError(f"unknown sequence_type {row.sequence_type!r}")

        if not target_ids or not non_target_ids:
            raise ValueError("hierarchy mapping requires both Target and Non-Target classes")

        target_ids = sorted(target_ids)
        non_target_ids = sorted(non_target_ids)
        collapsed = {class_id: idx for idx, class_id in enumerate(target_ids)}
        non_target_collapsed_id = len(target_ids)
        for class_id in non_target_ids:
            collapsed[class_id] = non_target_collapsed_id
        return cls(tuple(target_ids), tuple(non_target_ids), collapsed, non_target_collapsed_id)

    def to_binary(self, labels: torch.Tensor) -> torch.Tensor:
        target_ids = torch.tensor(self.target_class_ids, device=labels.device)
        return torch.isin(labels, target_ids).to(torch.long)

    def to_collapsed(self, labels: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(labels)
        for original_id, collapsed_id in self.collapsed_id_by_original.items():
            out[labels == original_id] = collapsed_id
        return out

    @property
    def n_collapsed_classes(self) -> int:
        return self.non_target_collapsed_id + 1
