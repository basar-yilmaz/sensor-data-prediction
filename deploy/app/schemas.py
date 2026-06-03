"""Pydantic v2 request/response schemas for the deploy API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TopKPrediction(BaseModel):
    gesture: str
    confidence: float = Field(ge=0.0, le=1.0)
    class_id: int = Field(ge=0)


class ModalityFlags(BaseModel):
    has_thm: bool
    has_tof: bool


class PredictionResponse(BaseModel):
    predicted_gesture: str
    predicted_confidence: float = Field(ge=0.0, le=1.0)
    predicted_class_id: int = Field(ge=0)
    top_k: list[TopKPrediction]
    has_thm: bool
    has_tof: bool
    n_sequences: int = Field(ge=1)
    sequence_length: int = Field(ge=1)
    inference_ms: float = Field(ge=0.0)
    sequence_predictions: list[SequencePrediction]


class SequencePrediction(BaseModel):
    sequence_id: str
    predicted_gesture: str
    predicted_confidence: float = Field(ge=0.0, le=1.0)
    predicted_class_id: int = Field(ge=0)
    top_k: list[TopKPrediction]
    has_thm: bool
    has_tof: bool
    sequence_length: int = Field(ge=1)


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: dict | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    checkpoint_path: str | None
    device: str
    n_classes: int


class ModelInfoResponse(BaseModel):
    architecture: str
    hidden_dim: int
    num_classes: int
    num_conv_blocks: int
    gru_layers: int
    dropout: float
    use_tof_raw: bool
    tof_embed_dim: int
    input_dim: int
    label_encoder_path: str
    scaler_path: str
