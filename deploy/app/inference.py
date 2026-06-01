"""Model loading and inference for the deploy service.

The Lightning checkpoint is loaded with ``strict=False`` because the
saving side (``bfrb_sensors.training.module.BFRBClassificationModule``) stores
the training-time ``class_weights`` buffer and ``hierarchy`` attribute, which
the inference model does not need. The ``model.*`` prefix is stripped so the
state dict matches the bare ``TemporalConvGRUClassifier`` layout.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn

from deploy.app.config import Settings
from deploy.app.model_arch import TemporalConvGRUClassifier

logger = logging.getLogger(__name__)


def _load_label_encoder(path: Path) -> dict[int, str]:
    payload = json.loads(path.read_text())
    return {int(v): k for k, v in payload.items()}


def _load_scaler(path: Path) -> dict[str, np.ndarray]:
    return joblib.load(path)


def _resolve_checkpoint_path(path: Path) -> Path:
    """Accept either a file or a directory and return a concrete .ckpt file.

    A directory is scanned for the lexicographically last ``*.ckpt`` so that
    newer checkpoints win. This is deterministic and reproducible; pick a
    different checkpoint explicitly via the env var if you need to.
    """
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Model checkpoint path {path} is neither a file nor a directory")
    candidates = sorted(path.rglob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No .ckpt files found under {path}")
    chosen = candidates[-1]
    logger.info("Auto-selected checkpoint: %s", chosen)
    return chosen


@dataclass
class ModelBundle:
    model: nn.Module
    label_encoder: dict[int, str]
    scaler: dict[str, np.ndarray]
    device: torch.device
    checkpoint_path: Path
    architecture: str
    hidden_dim: int
    num_classes: int
    num_conv_blocks: int
    gru_layers: int
    dropout: float
    use_tof_raw: bool
    tof_embed_dim: int
    input_dim: int


def load_model_bundle(settings: Settings) -> ModelBundle:
    """Build the model, load the checkpoint, return everything inference needs."""
    if not settings.scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at {settings.scaler_path}")
    if not settings.label_encoder_path.exists():
        raise FileNotFoundError(f"Label encoder not found at {settings.label_encoder_path}")

    checkpoint_path = _resolve_checkpoint_path(settings.model_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TemporalConvGRUClassifier(
        input_dim=settings.input_dim,
        hidden_dim=settings.hidden_dim,
        num_classes=settings.num_classes,
        dropout=settings.dropout,
        num_conv_blocks=settings.num_conv_blocks,
        gru_layers=settings.gru_layers,
        use_tof_raw=settings.use_tof_raw,
        tof_embed_dim=settings.tof_embed_dim,
        aux_binary=False,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    stripped = {
        (k[len("model.") :] if k.startswith("model.") else k): v for k, v in state_dict.items()
    }
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    if missing:
        logger.warning("Missing keys when loading model: %s", missing[:5])
    if unexpected:
        logger.warning("Unexpected keys when loading model: %s", unexpected[:5])

    model.to(device)
    model.eval()

    return ModelBundle(
        model=model,
        label_encoder=_load_label_encoder(settings.label_encoder_path),
        scaler=_load_scaler(settings.scaler_path),
        device=device,
        checkpoint_path=checkpoint_path,
        architecture="TemporalConvGRUClassifier",
        hidden_dim=settings.hidden_dim,
        num_classes=settings.num_classes,
        num_conv_blocks=settings.num_conv_blocks,
        gru_layers=settings.gru_layers,
        dropout=settings.dropout,
        use_tof_raw=settings.use_tof_raw,
        tof_embed_dim=settings.tof_embed_dim,
        input_dim=settings.input_dim,
    )


@torch.no_grad()
def predict(
    bundle: ModelBundle,
    batch: dict[str, torch.Tensor],
    *,
    top_k: int = 5,
) -> tuple[int, np.ndarray, float]:
    """Run inference on a pre-collated batch.

    Returns (predicted_class_id, full_probabilities (num_classes,), inference_ms).
    """
    batch = {key: value.to(bundle.device) for key, value in batch.items()}
    start = time.perf_counter()
    logits = bundle.model(batch).logits
    probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    inference_ms = (time.perf_counter() - start) * 1000.0
    predicted_class_id = int(np.argmax(probabilities))
    return predicted_class_id, probabilities, inference_ms
