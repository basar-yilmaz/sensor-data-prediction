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
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["hierarchy", "model"])
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.hierarchy = hierarchy
        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.val_macro_f1_18 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_binary_f1 = BinaryF1Score()
        self.val_macro_f1_collapsed = MulticlassF1Score(
            num_classes=hierarchy.n_collapsed_classes, average="macro"
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        return self.model(batch)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        logits = self(batch).logits
        loss = F.cross_entropy(logits, batch["label"])
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        logits = self(batch).logits
        labels = batch["label"]
        loss = F.cross_entropy(logits, labels)
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
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
