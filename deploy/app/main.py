"""FastAPI entry point for the BFRB gesture-classification deploy service."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from deploy.app.config import Settings, get_settings
from deploy.app.inference import ModelBundle, load_model_bundle, predict
from deploy.app.preprocessing import (
    PreprocessingError,
    featurize_and_collate,
    parse_csv,
)
from deploy.app.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    TopKPrediction,
)

logger = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = PACKAGE_DIR / "templates"
STATIC_DIR = PACKAGE_DIR / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _build_topk(
    probabilities: np.ndarray, label_encoder: dict[int, str], k: int
) -> list[TopKPrediction]:
    k = min(k, len(probabilities))
    top_indices = np.argsort(probabilities)[::-1][:k]
    return [
        TopKPrediction(
            gesture=label_encoder.get(int(idx), f"class_{idx}"),
            confidence=float(probabilities[idx]),
            class_id=int(idx),
        )
        for idx in top_indices
    ]


def _parse_settings_overrides(overrides: list[str] | None = None) -> Settings:
    return Settings(_env_file=None)  # type: ignore[call-arg]


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    try:
        bundle = load_model_bundle(settings)
    except FileNotFoundError as exc:
        logger.error("Startup failed: %s", exc)
        app.state.bundle = None
        app.state.startup_error = str(exc)
    else:
        app.state.bundle = bundle
        app.state.startup_error = None
        logger.info(
            "Loaded model %s on %s (checkpoint=%s, n_classes=%d)",
            bundle.architecture,
            bundle.device,
            bundle.checkpoint_path,
            bundle.num_classes,
        )
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or _parse_settings_overrides()
    app = FastAPI(
        title="BFRB Gesture Classifier",
        version="0.1.0",
        description=(
            "FastAPI service for Body-Focused Repetitive Behavior gesture "
            "classification from wrist-worn multimodal sensor data."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.exception_handler(PreprocessingError)
    async def _preprocessing_error_handler(
        request: Request, exc: PreprocessingError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                code=exc.code, message=exc.message, details=exc.details
            ).model_dump(),
        )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index(request: Request) -> Response:
        bundle: ModelBundle | None = request.app.state.bundle
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "model_loaded": bundle is not None,
                "checkpoint": str(bundle.checkpoint_path) if bundle else None,
                "device": str(bundle.device) if bundle else None,
                "startup_error": request.app.state.startup_error,
            },
        )

    @app.get("/api/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        bundle: ModelBundle | None = request.app.state.bundle
        return HealthResponse(
            status="ok" if bundle is not None else "degraded",
            model_loaded=bundle is not None,
            checkpoint_path=str(bundle.checkpoint_path) if bundle else None,
            device=str(bundle.device) if bundle else "cpu",
            n_classes=bundle.num_classes if bundle else 0,
        )

    @app.get("/api/model_info", response_model=ModelInfoResponse)
    async def model_info(request: Request) -> ModelInfoResponse:
        bundle: ModelBundle | None = request.app.state.bundle
        if bundle is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        return ModelInfoResponse(
            architecture=bundle.architecture,
            hidden_dim=bundle.hidden_dim,
            num_classes=bundle.num_classes,
            num_conv_blocks=bundle.num_conv_blocks,
            gru_layers=bundle.gru_layers,
            dropout=bundle.dropout,
            use_tof_raw=bundle.use_tof_raw,
            tof_embed_dim=bundle.tof_embed_dim,
            input_dim=bundle.input_dim,
            label_encoder_path=str(request.app.state.settings.label_encoder_path),
            scaler_path=str(request.app.state.settings.scaler_path),
        )

    @app.get("/api/sample", response_class=PlainTextResponse)
    async def sample(request: Request) -> PlainTextResponse:
        path: Path = request.app.state.settings.sample_data_path
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"sample data not found at {path}")
        return PlainTextResponse(path.read_text(), media_type="text/csv")

    @app.post("/api/predict", response_model=PredictionResponse)
    async def predict_endpoint(
        request: Request,
        file: UploadFile = File(...),  # noqa: B008
    ) -> PredictionResponse:
        bundle: ModelBundle | None = request.app.state.bundle
        if bundle is None:
            raise HTTPException(status_code=503, detail="model not loaded")

        file_bytes = await file.read()
        raw_sequences = parse_csv(
            file_bytes,
            min_seq_length=request.app.state.settings.min_seq_length,
            nan_threshold=request.app.state.settings.nan_threshold,
        )
        batch = featurize_and_collate(
            raw_sequences,
            bundle.scaler,
            max_seq_length=request.app.state.settings.max_seq_length,
        )
        predicted_class_id, probabilities, inference_ms = predict(
            bundle, batch, top_k=request.app.state.settings.top_k
        )

        top_k = _build_topk(probabilities, bundle.label_encoder, request.app.state.settings.top_k)
        has_thm = any(seq.has_thm for seq in raw_sequences)
        has_tof = any(seq.has_tof for seq in raw_sequences)
        return PredictionResponse(
            predicted_gesture=top_k[0].gesture,
            predicted_confidence=top_k[0].confidence,
            predicted_class_id=predicted_class_id,
            top_k=top_k,
            has_thm=has_thm,
            has_tof=has_tof,
            n_sequences=len(raw_sequences),
            sequence_length=int(batch["length"][0].item()),
            inference_ms=inference_ms,
        )

    @app.post("/api/predict_json", response_model=PredictionResponse)
    async def predict_json_endpoint(
        request: Request,
        records: list[dict[str, Any]],
    ) -> PredictionResponse:
        bundle: ModelBundle | None = request.app.state.bundle
        if bundle is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        if not records:
            raise HTTPException(status_code=400, detail="records must be a non-empty list")

        df = pd.DataFrame.from_records(records)
        if "sequence_id" not in df.columns:
            df["sequence_id"] = "0"
        if "sequence_counter" not in df.columns:
            df["sequence_counter"] = np.arange(len(df))
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        raw_sequences = parse_csv(
            csv_bytes,
            min_seq_length=request.app.state.settings.min_seq_length,
            nan_threshold=request.app.state.settings.nan_threshold,
        )
        batch = featurize_and_collate(
            raw_sequences,
            bundle.scaler,
            max_seq_length=request.app.state.settings.max_seq_length,
        )
        predicted_class_id, probabilities, inference_ms = predict(
            bundle, batch, top_k=request.app.state.settings.top_k
        )
        top_k = _build_topk(probabilities, bundle.label_encoder, request.app.state.settings.top_k)
        has_thm = any(seq.has_thm for seq in raw_sequences)
        has_tof = any(seq.has_tof for seq in raw_sequences)
        return PredictionResponse(
            predicted_gesture=top_k[0].gesture,
            predicted_confidence=top_k[0].confidence,
            predicted_class_id=predicted_class_id,
            top_k=top_k,
            has_thm=has_thm,
            has_tof=has_tof,
            n_sequences=len(raw_sequences),
            sequence_length=int(batch["length"][0].item()),
            inference_ms=inference_ms,
        )

    return app


app = create_app()


def run() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "deploy.app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=settings.reload,
    )


if __name__ == "__main__":
    run()
