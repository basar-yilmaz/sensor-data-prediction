"""End-to-end API tests using a TestClient with a randomly-initialized
``TemporalConvGRUClassifier`` injected as the model bundle. No real checkpoint
or scaler file is required.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from deploy.app.config import Settings
from deploy.app.inference import ModelBundle
from deploy.app.main import create_app
from deploy.app.model_arch import TemporalConvGRUClassifier


def _make_bundle(device: torch.device | None = None) -> ModelBundle:
    if device is None:
        device = torch.device("cpu")
    torch.manual_seed(0)
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
    model.to(device)
    model.eval()
    label_encoder = {i: f"class_{i}" for i in range(18)}
    scaler = {
        "n_timesteps": 1,
        "imu_mean": np.zeros(7, dtype=np.float32),
        "imu_std": np.ones(7, dtype=np.float32),
        "thm_mean": np.zeros(5, dtype=np.float32),
        "thm_std": np.ones(5, dtype=np.float32),
    }
    return ModelBundle(
        model=model,
        label_encoder=label_encoder,
        scaler=scaler,
        device=device,
        checkpoint_path=Path("/tmp/dummy.ckpt"),
        architecture="TemporalConvGRUClassifier",
        hidden_dim=128,
        num_classes=18,
        num_conv_blocks=2,
        gru_layers=1,
        dropout=0.2,
        use_tof_raw=True,
        tof_embed_dim=32,
        input_dim=39,
    )


@pytest.fixture
def dummy_settings(tmp_path) -> Settings:
    label_encoder = tmp_path / "label_encoder.json"
    label_encoder.write_text(
        '{"Forehead - scratch": 0, "Wave hello": 1, "Drink from bottle/cup": 2}'
    )
    return Settings(
        model_checkpoint=tmp_path / "missing.ckpt",
        scaler_path=tmp_path / "scaler.joblib",
        label_encoder_path=label_encoder,
        sample_data_path=tmp_path / "demo_sequence.csv",
        host="127.0.0.1",
        port=8000,
    )


@pytest.fixture
def client_with_dummy_model(monkeypatch, dummy_settings):
    bundle = _make_bundle()
    monkeypatch.setattr("deploy.app.main.load_model_bundle", lambda settings: bundle)
    app = create_app(settings=dummy_settings)
    with TestClient(app) as client:
        client.app.state.bundle = bundle
        client.app.state.startup_error = None
        yield client


def test_index_renders(client_with_dummy_model):
    response = client_with_dummy_model.get("/")
    assert response.status_code == 200
    assert "BFRB Gesture Classifier" in response.text
    assert "Drop a CSV here" in response.text


def test_health_endpoint(client_with_dummy_model):
    response = client_with_dummy_model.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["n_classes"] == 18


def test_model_info_endpoint(client_with_dummy_model):
    response = client_with_dummy_model.get("/api/model_info")
    assert response.status_code == 200
    body = response.json()
    assert body["architecture"] == "TemporalConvGRUClassifier"
    assert body["hidden_dim"] == 128
    assert body["num_classes"] == 18


def test_predict_endpoint_accepts_csv(client_with_dummy_model, sample_csv_bytes):
    response = client_with_dummy_model.post(
        "/api/predict",
        files={"file": ("demo.csv", sample_csv_bytes, "text/csv")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["predicted_gesture"].startswith("class_")
    assert 0.0 <= body["predicted_confidence"] <= 1.0
    assert 1 <= len(body["top_k"]) <= 5
    assert body["n_sequences"] >= 1
    assert body["sequence_length"] >= 1


def test_predict_endpoint_rejects_empty_file(client_with_dummy_model):
    response = client_with_dummy_model.post(
        "/api/predict",
        files={"file": ("empty.csv", b"", "text/csv")},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "empty_file"


def test_predict_endpoint_rejects_missing_columns(client_with_dummy_model):
    csv_bytes = b"foo,bar\n1,2\n3,4\n"
    response = client_with_dummy_model.post(
        "/api/predict",
        files={"file": ("bad.csv", csv_bytes, "text/csv")},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "missing_columns"


def test_predict_json_endpoint(client_with_dummy_model, sample_csv_bytes):
    import pandas as pd

    df = pd.read_csv(io.BytesIO(sample_csv_bytes))
    records: list[dict[str, Any]] = df.to_dict(orient="records")
    response = client_with_dummy_model.post("/api/predict_json", json=records)
    assert response.status_code == 200, response.text
    body = response.json()
    assert "predicted_gesture" in body


def test_health_reports_degraded_when_model_missing(monkeypatch, dummy_settings):
    def _raise(_settings):
        raise FileNotFoundError("no checkpoint")

    monkeypatch.setattr("deploy.app.main.load_model_bundle", _raise)
    app = create_app(settings=dummy_settings)
    with TestClient(app) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        body = response.json()
        assert body["model_loaded"] is False
        assert body["status"] == "degraded"
        index = client.get("/")
        assert "Heads up" in index.text or "Heads up." in index.text
