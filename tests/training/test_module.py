from __future__ import annotations

import pandas as pd
import pytest
import torch

from bfrb_sensors.data.label_encoder import build_label_encoder
from bfrb_sensors.models.baseline import BaselineMLPClassifier
from bfrb_sensors.training.metrics import HierarchyMapping
from bfrb_sensors.training.module import BFRBClassificationModule


def _batch(batch_size: int = 4, timesteps: int = 6) -> dict[str, torch.Tensor]:
    return {
        "imu": torch.randn(batch_size, timesteps, 7),
        "imu_derived": torch.randn(batch_size, timesteps, 7),
        "thm": torch.randn(batch_size, timesteps, 5),
        "tof": torch.randn(batch_size, timesteps, 5, 8, 8),
        "tof_stats": torch.randn(batch_size, timesteps, 20),
        "attention_mask": torch.ones(batch_size, timesteps, dtype=torch.bool),
        "label": torch.tensor([0, 1, 2, 0])[:batch_size],
    }


def _mapping() -> HierarchyMapping:
    encoder = build_label_encoder(["target_a", "target_b", "non_a"])
    index = pd.DataFrame(
        [
            {"gesture": "target_a", "sequence_type": "Target"},
            {"gesture": "target_b", "sequence_type": "Target"},
            {"gesture": "non_a", "sequence_type": "Non-Target"},
        ]
    )
    return HierarchyMapping.from_index(index, encoder)


@pytest.mark.filterwarnings("ignore:You are trying to `self.log\\(\\)`")
def test_training_step_returns_loss():
    module = BFRBClassificationModule(
        model=BaselineMLPClassifier(input_dim=39, hidden_dim=16, num_classes=3, dropout=0.0),
        num_classes=3,
        lr=1e-3,
        weight_decay=0.0,
        hierarchy=_mapping(),
    )

    loss = module.training_step(_batch(), 0)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


@pytest.mark.filterwarnings("ignore:You are trying to `self.log\\(\\)`")
def test_training_step_with_aux_binary_head():
    module = BFRBClassificationModule(
        model=BaselineMLPClassifier(
            input_dim=39, hidden_dim=16, num_classes=3, dropout=0.0, aux_binary=True
        ),
        num_classes=3,
        lr=1e-3,
        weight_decay=0.0,
        hierarchy=_mapping(),
        aux_binary_weight=0.3,
    )
    loss = module.training_step(_batch(), 0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def _module(scheduler: str = "none", **kwargs) -> BFRBClassificationModule:
    return BFRBClassificationModule(
        model=BaselineMLPClassifier(input_dim=39, hidden_dim=16, num_classes=3, dropout=0.0),
        num_classes=3,
        lr=1e-3,
        weight_decay=0.0,
        hierarchy=_mapping(),
        scheduler=scheduler,
        **kwargs,
    )


def test_configure_optimizers_without_scheduler_returns_bare_optimizer():
    module = _module(scheduler="none")

    result = module.configure_optimizers()

    assert isinstance(result, torch.optim.Optimizer)


def test_configure_optimizers_with_reduce_on_plateau_returns_scheduler_dict():
    module = _module(
        scheduler="reduce_on_plateau",
        scheduler_factor=0.5,
        scheduler_patience=3,
        scheduler_min_lr=1e-6,
        monitor="val_hierarchical_f1",
        monitor_mode="max",
    )

    result = module.configure_optimizers()

    assert isinstance(result, dict)
    assert isinstance(result["optimizer"], torch.optim.Optimizer)
    lr_scheduler = result["lr_scheduler"]
    assert lr_scheduler["monitor"] == "val_hierarchical_f1"
    scheduler = lr_scheduler["scheduler"]
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert scheduler.mode == "max"
    assert scheduler.factor == 0.5
    assert scheduler.patience == 3
    assert scheduler.min_lrs == [1e-6]
