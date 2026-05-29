# BFRB Sensor Data Prediction

Train a multimodal time-series classifier for the CMI Body-Focused Repetitive Behavior
(BFRB) sensor dataset. The project uses DVC for data versioning, MinIO as the
S3-compatible DVC remote, Hydra for configuration, PyTorch Lightning for training, and
MLflow for experiment tracking.

## Quick Start

Requires Python 3.11, [`uv`](https://docs.astral.sh/uv/), Docker, and Docker Compose.

### 1. Install Dependencies

```bash
uv sync
```

### 2. Create `.env`

Copy the example environment file:

```bash
cp .env.example .env
```

The defaults are enough for local development:

```env
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
```

These credentials must match the DVC remote credentials in `.dvc/config`. The default
remote is `bfrb-data` at `http://localhost:9000`.

### 3. Start Local Services

Start MinIO and MLflow:

```bash
docker compose up -d minio-init mlflow
```

Services:

- MinIO S3 API: `http://127.0.0.1:9000`
- MinIO console: `http://127.0.0.1:9001`
- MLflow: `http://127.0.0.1:8080`

`minio-init` creates the required buckets:

- `bfrb-data`
- `bfrb-models`

### 4. Run Training

```bash
uv run bfrb train
```

Training automatically handles data setup. You do not need to run the data processing
commands separately before training.

On a fresh machine, `bfrb train` will:

1. Check that MLflow is reachable.
2. Ensure `data/raw/train.csv` exists.
3. Try to pull raw data from the DVC remote `bfrb-data`.
4. If raw data is missing from the remote, download the public dataset mirror and push
   the raw CSV to MinIO through DVC.
5. Ensure prepared data exists.
6. Try to pull prepared artifacts, including `splits.json`, from the DVC remote.
7. If prepared artifacts are missing from the remote, run the DVC `prepare` and `splits`
   stages locally and push those outputs to MinIO.
8. Start model training.

This keeps `splits.json` shared through MinIO so machines connected to the same DVC
remote use the same train/validation folds.

## Common Training Overrides

Hydra overrides can be passed directly after the command:

```bash
uv run bfrb train training.max_epochs=60
uv run bfrb train training.fold=1 training.seed=123
uv run bfrb train model.use_tof_raw=false
```

Training outputs include checkpoints under `artifacts/checkpoints/`, plots under
`plots/`, and metrics/artifacts in MLflow.

## Optional Data Commands

These commands are useful for debugging or manually preparing data, but they are not
required before `uv run bfrb train`.

### Fetch Data

```bash
uv run bfrb fetch
```

Ensures raw and prepared data are available. It pulls from MinIO when possible, falls
back to the public dataset download for raw data, and prepares/pushes missing prepared
artifacts when needed.

### Pull DVC Data Only

```bash
uv run bfrb download
```

Runs a DVC pull from the configured remote. This does not use the public dataset
download fallback.

### Reproduce DVC Pipeline

```bash
uv run dvc repro
```

Runs the DVC pipeline stages:

- `prepare`: raw CSV to per-sequence parquet files, index, and label encoder
- `splits`: subject-disjoint stratified train/validation fold assignments

### Run Individual Data Stages

```bash
uv run bfrb prepare
uv run bfrb splits
```

Use these only when you explicitly want to run one stage outside the full DVC pipeline.

## Data And Model Summary

The dataset contains wrist-worn sensor sequences from the CMI BFRB competition:

- IMU: acceleration and rotation quaternion channels
- Thermopile: five temperature channels
- Time-of-Flight: five 8x8 distance sensors
- Target: 18 gesture/orientation classes

Validation uses 5-fold `StratifiedGroupKFold` with subject IDs as groups, so the same
participant does not appear in both training and validation for a fold.

The default model is `temporal_conv_gru_tof`, a temporal Conv1D + bidirectional GRU
classifier with optional raw-ToF spatial features. The default training config uses
class weighting and an auxiliary binary BFRB/non-BFRB loss.
