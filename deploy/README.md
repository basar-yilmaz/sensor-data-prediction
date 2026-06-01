# BFRB Deploy Service

A self-contained FastAPI service that exposes the trained BFRB gesture
classifier as a local HTTP API and a small web UI. Lives in `deploy/` and does
**not** import from the training pipeline; the model is loaded from a
Lightning checkpoint and the scaler / label encoder are read from disk.

## Layout

```
deploy/
├── app/
│   ├── main.py            FastAPI app + routes
│   ├── inference.py       Checkpoint loading, model bundle, predict()
│   ├── preprocessing.py   CSV -> model-ready tensor pipeline
│   ├── model_arch.py      Copy of TemporalConvGRUClassifier (inference only)
│   ├── config.py          Pydantic settings (env-driven)
│   ├── schemas.py         Pydantic request/response models
│   ├── templates/         Jinja2 HTML
│   └── static/            CSS, JS, favicon
├── sample_data/           One demo CSV for the "Try sample" button
├── tests/                 pytest suite (preprocessing, inference, API)
├── scripts/run.sh         Helper to launch the service
└── pyproject.toml         Standalone package manifest
```

## Install

The deploy package declares its own deps (FastAPI, Uvicorn, Jinja2, Pydantic,
etc.) plus the same torch / numpy / scipy stack as the training side. Sync
into the existing project venv:

```bash
uv sync
```

The package is built as `bfrb-deploy` (separate from the training
`bfrb-sensors` package) so its dependencies stay scoped.

## Run

```bash
uv run bfrb-serve
# or, equivalently:
uv run uvicorn deploy.app.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>.

The service loads the model checkpoint at startup. By default it picks the
lexicographically-last `*.ckpt` under `artifacts/checkpoints/`. Set
`BFRB_MODEL_CHECKPOINT` to point at a specific file:

```bash
BFRB_MODEL_CHECKPOINT=artifacts/checkpoints/<run_id>/epoch=42-val_hierarchical_f1=0.71.ckpt \
  uv run bfrb-serve
```

## Configuration

All settings are environment variables, prefixed with `BFRB_`.

| Variable                  | Default                                | Description                                     |
| ------------------------- | -------------------------------------- | ----------------------------------------------- |
| `BFRB_HOST`               | `127.0.0.1`                            | Bind address                                    |
| `BFRB_PORT`               | `8000`                                 | Bind port                                       |
| `BFRB_LOG_LEVEL`          | `info`                                 | Uvicorn log level                               |
| `BFRB_RELOAD`             | `false`                                | Enable autoreload (dev only)                    |
| `BFRB_MODEL_CHECKPOINT`   | `artifacts/checkpoints`                | File or directory                               |
| `BFRB_SCALER_PATH`        | `artifacts/scaler.joblib`              | Fitted StandardScaler                           |
| `BFRB_LABEL_ENCODER_PATH` | `data/prepared/label_encoder.json`     | Gesture id map                                  |
| `BFRB_SAMPLE_DATA_PATH`   | `deploy/sample_data/demo_sequence.csv` | Demo CSV                                        |
| `BFRB_HIDDEN_DIM`         | `128`                                  | Model hidden dim                                |
| `BFRB_NUM_CONV_BLOCKS`    | `2`                                    | Residual Conv1D blocks                          |
| `BFRB_GRU_LAYERS`         | `1`                                    | BiGRU layers                                    |
| `BFRB_DROPOUT`            | `0.2`                                  | Dropout rate                                    |
| `BFRB_USE_TOF_RAW`        | `true`                                 | Use raw ToF 2D CNN encoder                      |
| `BFRB_TOF_EMBED_DIM`      | `32`                                   | ToF embedding dim                               |
| `BFRB_NUM_CLASSES`        | `18`                                   | Output classes                                  |
| `BFRB_INPUT_DIM`          | `39`                                   | Per-step vector dim (IMU+IMU-der+THM+ToF-stats) |
| `BFRB_MAX_SEQ_LENGTH`     | `4096`                                 | Hard cap on per-upload sequence length          |
| `BFRB_MIN_SEQ_LENGTH`     | `10`                                   | Drop shorter sequences                          |
| `BFRB_TOP_K`              | `5`                                    | Top-K returned in predictions                   |
| `BFRB_NAN_THRESHOLD`      | `0.5`                                  | Drop a sequence if NaN fraction exceeds this    |

## API

### `GET /`

The web UI (HTML).

### `GET /api/health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "checkpoint_path": ".../epoch=42-val_hierarchical_f1=0.71.ckpt",
  "device": "cuda",
  "n_classes": 18
}
```

### `GET /api/model_info`

Architecture + paths of the active bundle.

### `GET /api/sample`

Returns `sample_data/demo_sequence.csv` for the "Try sample" button.

### `POST /api/predict`

Multipart upload with a `file` field (CSV). Returns:

```json
{
  "predicted_gesture": "Forehead - scratch",
  "predicted_confidence": 0.873,
  "predicted_class_id": 7,
  "top_k": [
    { "gesture": "Forehead - scratch", "confidence": 0.873, "class_id": 7 },
    { "gesture": "Neck - scratch", "confidence": 0.09, "class_id": 10 }
  ],
  "has_thm": true,
  "has_tof": true,
  "n_sequences": 1,
  "sequence_length": 67,
  "inference_ms": 12.4
}
```

Errors are returned as `{"code": "...", "message": "...", "details": {...}}`
with HTTP 400.

### `POST /api/predict_json`

JSON body: a list of per-timestep records (same column shape as the CSV
format). Useful from scripts / curl.

## CSV format

One row per timestep. Required columns:

- `sequence_id`, `sequence_counter` (metadata)
- `acc_x`, `acc_y`, `acc_z`, `rot_w`, `rot_x`, `rot_y`, `rot_z` (IMU)
- `thm_1` … `thm_5` (thermopile)
- `tof_1_v0` … `tof_5_v63` (5 ToF sensors × 8×8 pixels = 320 columns)

`-1.0` is treated as a missing ToF value. Sequences shorter than
`BFRB_MIN_SEQ_LENGTH` are dropped.

## Tests

```bash
uv run --package bfrb-deploy pytest deploy/tests
```

Tests inject a randomly-initialized `TemporalConvGRUClassifier` so they
don't require a real checkpoint or scaler on disk.

## Updating the model

The deploy service loads whatever checkpoint lives on disk at startup. To
swap in a new model, just retrain (`uv run bfrb train`) and restart
`bfrb-serve`. The service picks the newest `*.ckpt` automatically; pin a
specific file via `BFRB_MODEL_CHECKPOINT` if you need to.

## Out of scope

- Authentication / rate limiting (the proposal calls for a local service).
- Persistent prediction history (no DB).
- Hot-swap / model versioning (restart the service to load a new
  checkpoint).
- Streaming or WebSocket inference.
