"""Model construction from Hydra model config."""

from __future__ import annotations

from torch import nn

from bfrb_sensors.models.baseline import BaselineMLPClassifier
from bfrb_sensors.models.temporal import TemporalConvGRUClassifier


def build_model(model_cfg) -> nn.Module:
    name = str(model_cfg.name)
    if name == "baseline_mlp":
        return BaselineMLPClassifier(
            input_dim=int(model_cfg.input_dim),
            hidden_dim=int(model_cfg.hidden_dim),
            num_classes=int(model_cfg.num_classes),
            dropout=float(model_cfg.dropout),
            aux_binary=bool(model_cfg.aux_binary),
        )
    if name == "temporal_conv_gru":
        return TemporalConvGRUClassifier(
            input_dim=int(model_cfg.input_dim),
            hidden_dim=int(model_cfg.hidden_dim),
            num_classes=int(model_cfg.num_classes),
            dropout=float(model_cfg.dropout),
            num_conv_blocks=int(model_cfg.num_conv_blocks),
            gru_layers=int(model_cfg.gru_layers),
            use_tof_raw=bool(model_cfg.use_tof_raw),
            tof_embed_dim=int(model_cfg.tof_embed_dim),
            aux_binary=bool(model_cfg.aux_binary),
            use_demographics=bool(model_cfg.use_demographics),
            meta_embed_dim=int(model_cfg.meta_embed_dim),
        )
    raise ValueError(f"unknown model {name!r}")
