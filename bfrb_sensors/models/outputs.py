"""Structured model outputs shared across architectures."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ModelOutput:
    """Classifier outputs. ``binary_logits`` is populated only by models with an
    auxiliary target/non-target head; it is ``None`` otherwise."""

    logits: torch.Tensor
    binary_logits: torch.Tensor | None = None
