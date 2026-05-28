"""Training orchestration entry point."""

from __future__ import annotations

from omegaconf import DictConfig


def train_from_config(cfg: DictConfig) -> None:
    raise NotImplementedError("training orchestration is not wired yet")
