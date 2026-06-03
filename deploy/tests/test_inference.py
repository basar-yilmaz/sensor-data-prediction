"""Tests for model loading and inference primitives."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from app.config import Settings
from app.inference import load_model_bundle, predict
from app.model_arch import TemporalConvGRUClassifier
from app.preprocessing import featurize_and_collate, parse_csv


@pytest.fixture
def settings_with_dummy_checkpoint(tmp_path, tmp_scaler_path) -> Settings:
    model = TemporalConvGRUClassifier(
        input_dim=39,
        hidden_dim=128,
        num_classes=18,
        dropout=0.2,
        num_conv_blocks=2,
        gru_layers=1,
        use_tof_raw=True,
        tof_embed_dim=32,
        aux_binary=False,
    )
    ckpt_path = tmp_path / "model.ckpt"
    torch.save({"state_dict": model.state_dict()}, ckpt_path)
    label_encoder = tmp_path / "label_encoder.json"
    label_encoder.write_text(
        '{"Forehead - scratch": 0, "Wave hello": 1, "Drink from bottle/cup": 2}'
    )
    return Settings(
        model_checkpoint=ckpt_path,
        scaler_path=tmp_scaler_path,
        label_encoder_path=label_encoder,
        sample_data_path=tmp_path / "demo_sequence.csv",
        hidden_dim=128,
        num_conv_blocks=2,
        gru_layers=1,
        dropout=0.2,
        use_tof_raw=True,
        tof_embed_dim=32,
        num_classes=18,
        input_dim=39,
    )


def test_load_model_bundle_strips_lightning_prefix(tmp_path, tmp_scaler_path):
    """The training-side LightningModule stores its inner model with a 'model.'
    prefix. The deploy loader must strip that so the state dict matches the
    bare TemporalConvGRUClassifier layout.
    """
    from app.model_arch import TemporalConvGRUClassifier

    model = TemporalConvGRUClassifier(
        input_dim=39,
        hidden_dim=128,
        num_classes=18,
        dropout=0.2,
        num_conv_blocks=2,
        gru_layers=1,
        use_tof_raw=True,
        tof_embed_dim=32,
        aux_binary=False,
    )
    prefixed = {f"model.{k}": v for k, v in model.state_dict().items()}
    prefixed["class_weights"] = torch.zeros(18)
    prefixed["hierarchy_dummy"] = torch.zeros(1)
    ckpt_path = tmp_path / "model.ckpt"
    torch.save({"state_dict": prefixed, "epoch": 0}, ckpt_path)

    label_encoder = tmp_path / "label_encoder.json"
    label_encoder.write_text('{"a": 0}')

    settings = Settings(
        model_checkpoint=ckpt_path,
        scaler_path=tmp_scaler_path,
        label_encoder_path=label_encoder,
        sample_data_path=tmp_path / "demo_sequence.csv",
        hidden_dim=128,
        num_conv_blocks=2,
        gru_layers=1,
        dropout=0.2,
        use_tof_raw=True,
        tof_embed_dim=32,
        num_classes=18,
        input_dim=39,
    )
    bundle = load_model_bundle(settings)
    assert bundle.num_classes == 18
    assert bundle.label_encoder[0] == "a"


def test_predict_returns_top_class_and_full_distribution(
    settings_with_dummy_checkpoint, sample_csv_bytes
):
    torch.manual_seed(0)
    bundle = load_model_bundle(settings_with_dummy_checkpoint)
    raw_sequences = parse_csv(sample_csv_bytes, min_seq_length=5, nan_threshold=0.99)
    batch = featurize_and_collate(raw_sequences, bundle.scaler)
    predicted, probabilities, inference_ms = predict(bundle, batch, top_k=5)
    assert 0 <= predicted < bundle.num_classes
    assert probabilities.shape == (bundle.num_classes,)
    assert np.all(np.isfinite(probabilities))
    assert pytest.approx(probabilities.sum(), abs=1e-4) == 1.0
    assert np.all(probabilities >= 0.0)
    assert inference_ms >= 0.0


def test_load_model_bundle_raises_when_checkpoint_missing(tmp_path, tmp_scaler_path):
    label_encoder = tmp_path / "label_encoder.json"
    label_encoder.write_text("{}")
    settings = Settings(
        model_checkpoint=tmp_path / "missing.ckpt",
        scaler_path=tmp_scaler_path,
        label_encoder_path=label_encoder,
        sample_data_path=tmp_path / "demo_sequence.csv",
    )
    with pytest.raises(FileNotFoundError):
        load_model_bundle(settings)


def test_resolve_checkpoint_picks_latest(tmp_path):
    from app.inference import _resolve_checkpoint_path

    (tmp_path / "a.ckpt").write_bytes(b"a")
    (tmp_path / "b.ckpt").write_bytes(b"b")
    chosen = _resolve_checkpoint_path(tmp_path)
    assert chosen.name == "b.ckpt"
