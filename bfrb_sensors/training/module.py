"""LightningModule for BFRB gesture classification."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics.classification import BinaryF1Score, MulticlassAccuracy, MulticlassF1Score

from bfrb_sensors.models.outputs import ModelOutput
from bfrb_sensors.training.metrics import HierarchyMapping


class BFRBClassificationModule(pl.LightningModule):
    """Lightning module wrapping an injected classifier.

    The model and hierarchy are excluded from saved hyperparameters, so
    ``load_from_checkpoint`` must be called with ``model=build_model(cfg.model)``
    and ``hierarchy=...`` supplied explicitly.
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        lr: float,
        weight_decay: float,
        hierarchy: HierarchyMapping,
        class_weights: torch.Tensor | None = None,
        aux_binary_weight: float = 0.0,
        scheduler: str = "none",
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 3,
        scheduler_min_lr: float = 1.0e-6,
        monitor: str = "val_hierarchical_f1",
        monitor_mode: str = "max",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["hierarchy", "model", "class_weights"])
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.hierarchy = hierarchy
        self.aux_binary_weight = aux_binary_weight
        self.scheduler = scheduler
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.scheduler_min_lr = scheduler_min_lr
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.register_buffer("class_weights", class_weights)
        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.val_macro_f1_18 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_binary_f1 = BinaryF1Score()
        self.val_macro_f1_collapsed = MulticlassF1Score(
            num_classes=hierarchy.n_collapsed_classes, average="macro"
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        return self.model(batch)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        out = self(batch)
        loss = F.cross_entropy(out.logits, batch["label"], weight=self.class_weights)
        if out.binary_logits is not None and self.aux_binary_weight > 0:
            binary_labels = self.hierarchy.to_binary(batch["label"])
            aux_loss = F.cross_entropy(out.binary_logits, binary_labels)
            self.log("train_loss_main", loss, on_step=False, on_epoch=True)
            self.log("train_loss_aux", aux_loss, on_step=False, on_epoch=True)
            loss = loss + self.aux_binary_weight * aux_loss
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        out = self(batch)
        logits = out.logits
        labels = batch["label"]
        loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        preds = logits.argmax(dim=1)
        binary_labels = self.hierarchy.to_binary(labels)
        binary_preds = self.hierarchy.to_binary(preds)
        collapsed_labels = self.hierarchy.to_collapsed(labels)
        collapsed_preds = self.hierarchy.to_collapsed(preds)

        self.val_accuracy.update(preds, labels)
        self.val_macro_f1_18.update(preds, labels)
        self.val_binary_f1.update(binary_preds, binary_labels)
        self.val_macro_f1_collapsed.update(collapsed_preds, collapsed_labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        val_accuracy = self.val_accuracy.compute()
        val_macro_f1_18 = self.val_macro_f1_18.compute()
        val_binary_f1 = self.val_binary_f1.compute()
        val_macro_f1_collapsed = self.val_macro_f1_collapsed.compute()
        val_hierarchical_f1 = 0.5 * (val_binary_f1 + val_macro_f1_collapsed)
        self.log("val_accuracy", val_accuracy, prog_bar=True)
        self.log("val_macro_f1_18", val_macro_f1_18)
        self.log("val_binary_f1", val_binary_f1)
        self.log("val_macro_f1_collapsed", val_macro_f1_collapsed)
        self.log("val_hierarchical_f1", val_hierarchical_f1, prog_bar=True)
        self.val_accuracy.reset()
        self.val_macro_f1_18.reset()
        self.val_binary_f1.reset()
        self.val_macro_f1_collapsed.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.scheduler == "none":
            return optimizer
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=self.monitor_mode,
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            min_lr=self.scheduler_min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": self.monitor},
        }
