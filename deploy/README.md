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
into the deploy venv:

```bash
uv --project deploy sync
```

The package is built as `bfrb-deploy` (separate from the training
`bfrb-sensors` package) so its dependencies stay scoped.

## Run

```bash
uv --project deploy run bfrb-serve
# or, equivalently:
uv --project deploy run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>.

Inference runs on **CPU by default**; set `BFRB_DEVICE=cuda` to use a GPU (it
falls back to CPU with a warning if no GPU is available).

The service loads the model checkpoint at startup. By default it uses
`models/temporal_conv_gru_tof.ckpt`. If that file is not on local disk it is
pulled from the `bfrb-models` DVC remote automatically; likewise the label
encoder (`data/prepared/label_encoder.json`) is pulled from `bfrb-data` when
missing. A fresh checkout can therefore serve without re-running the pipeline,
as long as the DVC remotes are reachable.

`BFRB_MODEL_CHECKPOINT` can point at a specific file or a directory (the newest
`*.ckpt` underneath wins):

```bash
BFRB_MODEL_CHECKPOINT=artifacts/checkpoints/<run_id>/epoch=42-val_hierarchical_f1=0.71.ckpt \
  uv --project deploy run bfrb-serve
```

You can also swap the model at runtime from the web UI: the **Model** card lets
you pick a trained `.ckpt` from your filesystem and load it without restarting
(see `POST /api/load_model`).

## Configuration

All settings are environment variables, prefixed with `BFRB_`.

| Variable                  | Default                                | Description                                     |
| ------------------------- | -------------------------------------- | ----------------------------------------------- |
| `BFRB_HOST`               | `127.0.0.1`                            | Bind address                                    |
| `BFRB_PORT`               | `8000`                                 | Bind port                                       |
| `BFRB_LOG_LEVEL`          | `info`                                 | Uvicorn log level                               |
| `BFRB_RELOAD`             | `false`                                | Enable autoreload (dev only)                    |
| `BFRB_DEVICE`             | `cpu`                                  | Inference device (`cpu` or `cuda`)              |
| `BFRB_MODEL_CHECKPOINT`   | `models/temporal_conv_gru_tof.ckpt`    | File or directory; pulled from DVC if missing   |
| `BFRB_SCALER_PATH`        | `artifacts/scaler.joblib`              | Fitted StandardScaler                           |
| `BFRB_LABEL_ENCODER_PATH` | `data/prepared/label_encoder.json`     | Gesture id map; pulled from DVC if missing      |
| `BFRB_SAMPLE_DATA_PATH`   | `deploy/sample_data/demo_sequence.csv` | Demo CSV                                        |
| `BFRB_REPO_ROOT`          | repo root                              | Root used for DVC pulls / uploaded checkpoints  |
| `BFRB_DATA_REMOTE`        | `bfrb-data`                            | DVC remote for the label encoder                |
| `BFRB_MODEL_REMOTE`       | `bfrb-models`                          | DVC remote for the checkpoint                   |
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

### `POST /api/load_model`

Multipart upload with a `file` field (a `.ckpt`). Stores the checkpoint under
`artifacts/uploaded_checkpoints/` and rebuilds the active model bundle from it
(reusing the configured scaler and label encoder). Returns the same payload as
`GET /api/health`. Backs the **Model** card in the web UI.

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
uv --project deploy run pytest -c deploy/pyproject.toml deploy/tests
```

Tests inject a randomly-initialized `TemporalConvGRUClassifier` so they
don't require a real checkpoint or scaler on disk.

## Updating the model

The deploy service loads `models/temporal_conv_gru_tof.ckpt` at startup,
pulling it from the `bfrb-models` DVC remote when it is not already on disk. To
swap in a new model you can either retrain (`uv run bfrb train`, which updates
and pushes that checkpoint) and restart `bfrb-serve`, or upload a `.ckpt` at
runtime via the **Model** card in the UI / `POST /api/load_model`. Pin a
different startup checkpoint with `BFRB_MODEL_CHECKPOINT`.
