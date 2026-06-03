"""Competition-like hierarchical metric target transforms."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score

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


def evaluate_predictions(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    hierarchy: HierarchyMapping,
    num_classes: int,
    prefix: str = "val",
    y_proba: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> dict[str, float]:
    """Hierarchical metrics from integer predictions, shared by every model.

    Definitions match :class:`bfrb_sensors.training.module.BFRBClassificationModule`
    so the neural model and the baselines report directly comparable metrics:
    ``{prefix}_hierarchical_f1 = 0.5 * (binary_f1 + macro_f1_collapsed)``.
    ``prefix`` is typically ``"val"`` or ``"test"``.
    """
    yt = torch.as_tensor(np.asarray(y_true), dtype=torch.long)
    yp = torch.as_tensor(np.asarray(y_pred), dtype=torch.long)

    binary_true = hierarchy.to_binary(yt).numpy()
    binary_pred = hierarchy.to_binary(yp).numpy()
    collapsed_true = hierarchy.to_collapsed(yt).numpy()
    collapsed_pred = hierarchy.to_collapsed(yp).numpy()

    macro_f1_18 = f1_score(
        yt.numpy(), yp.numpy(), labels=list(range(num_classes)), average="macro", zero_division=0
    )
    binary_precision = precision_score(
        binary_true, binary_pred, pos_label=1, average="binary", zero_division=0
    )
    binary_recall = recall_score(
        binary_true, binary_pred, pos_label=1, average="binary", zero_division=0
    )
    binary_f1 = f1_score(binary_true, binary_pred, pos_label=1, average="binary", zero_division=0)
    macro_f1_collapsed = f1_score(
        collapsed_true,
        collapsed_pred,
        labels=list(range(hierarchy.n_collapsed_classes)),
        average="macro",
        zero_division=0,
    )
    hierarchical_f1 = 0.5 * (binary_f1 + macro_f1_collapsed)

    scores = {
        f"{prefix}_accuracy": float(accuracy_score(yt.numpy(), yp.numpy())),
        f"{prefix}_macro_f1_18": float(macro_f1_18),
        f"{prefix}_binary_precision": float(binary_precision),
        f"{prefix}_binary_recall": float(binary_recall),
        f"{prefix}_binary_f1": float(binary_f1),
        f"{prefix}_macro_f1_collapsed": float(macro_f1_collapsed),
        f"{prefix}_hierarchical_f1": float(hierarchical_f1),
    }
    if y_proba is not None:
        scores[f"{prefix}_log_loss"] = float(
            log_loss(yt.numpy(), np.asarray(y_proba), labels=list(range(num_classes)))
        )
    return scores
