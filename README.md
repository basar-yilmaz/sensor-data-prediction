# BFRB Sensor Data Prediction

Train a multimodal time-series classifier for the CMI Body-Focused Repetitive Behavior
(BFRB) sensor dataset. The project uses DVC for data versioning, MinIO as the
S3-compatible DVC remote, Hydra for configuration, PyTorch Lightning for training, and
MLflow for experiment tracking.

## Problem Statement

Body-Focused Repetitive Behaviors (BFRBs), including trichotillomania, excoriation
disorder, and onychophagia, affect roughly 1-5% of the global population. Clinical
assessment is often based on self-reporting, which is subjective and can miss events due
to recall errors.

The goal of this project is to classify BFRB-like gestures and routine gestures from
wrist-worn multimodal sensor data. The work is based on the Child Mind Institute Kaggle
competition
[CMI - Detect Behavior with Sensor Data](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data).
The project focuses on training code for a reproducible model pipeline.

## Input And Output Data

Each example is a variable-length time-series sequence from a wrist-worn Helios device.
The input contains three sensor modalities:

- IMU: 7 persistent channels, with 3-axis acceleration (`acc_x`, `acc_y`, `acc_z`) and a
  4-component rotation quaternion (`rot_w`, `rot_x`, `rot_y`, `rot_z`).
- Thermopile / THM: 5 temperature channels (`thm_1` to `thm_5`), which may be missing for
  some sequences.
- Time-of-Flight / ToF: 5 sensors with 8x8 spatial resolution each, for 320 raw distance
  pixels per timestep, also missing for some sequences.

The model predicts one `gesture` label per sequence. The label space has 18 classes: 8
BFRB-like target gestures and 10 non-target routine gestures. Subject orientation and
sequence phase are treated as metadata, not prediction targets.

## Metrics

The primary metric is hierarchical macro-F1, matching the competition-style objective:

```text
hierarchical macro-F1 = 0.5 * (binary F1 + collapsed macro-F1)
```

Binary F1 measures target-vs-non-target BFRB detection. Collapsed macro-F1 measures
gesture discrimination after collapsing all non-target gestures into one class. The
training pipeline also logs binary precision, binary recall, macro-F1 over all 18
classes, accuracy, and log-loss.

## Validation

The project uses a fixed three-way split: approximately `0.8 / 0.1 / 0.1` for
train/validation/test. The split is stratified by gesture and subject-disjoint through a
two-stage `StratifiedGroupKFold`, so no participant appears in more than one split.

Validation drives early stopping, learning-rate scheduling, and checkpoint selection. The test split is evaluated once after training to estimate generalization to unseen people. The same split file is used by both the neural model and the XGBoost baseline so their
metrics are directly comparable.

## Data Processing

The raw dataset is a large CSV with about 575k sensor rows and about 8k sequences.
Preparation converts it into per-sequence parquet files plus an index and label encoder.
The preprocessing stage:

- sorts each sequence by timestep;
- drops very short sequences and sequences with too much IMU missingness;
- fills partial missing sensor values by forward/back fill, with zero fallback;
- derives IMU features: gravity-removed linear acceleration, angular velocity, and
  angular distance;
- computes ToF summary statistics per sensor;
- stores modality-availability flags for THM and ToF.

During model training, IMU and thermopile channels are z-scored with statistics fit only
on the training split. Variable-length batches are padded and include an `attention_mask`
so pooling ignores padded timesteps. Training-only modality dropout randomly zeros THM
and ToF inputs to improve robustness to missing sensors.

## Models

### Baseline

The baseline is an IMU-only XGBoost classifier. It converts each variable-length sequence
into fixed-length summary features by computing mean, standard deviation, minimum,
maximum, and median over raw and derived IMU channels. It trains with class weighting and
early stopping on validation log-loss, then reports the same hierarchical metrics as the
neural model.

### Main Model

The default neural model is `temporal_conv_gru_tof`, a multimodal temporal classifier.
Per timestep, it concatenates raw IMU, derived IMU, thermopile, and ToF summary features
for a 39-dimensional vector. When raw ToF is enabled, a small 2D CNN encodes each
5-channel 8x8 ToF frame into a 32-dimensional embedding, giving a 71-dimensional
per-timestep input.

The model stack is:

- ToF spatial encoder: small Conv2D network with global average pooling.
- Input projection: linear projection to hidden dimension 128 with LayerNorm, GELU, and
  dropout.
- Temporal encoder: two residual Conv1D blocks followed by a bidirectional GRU.
- Pooling: masked attention pooling over timesteps.
- Heads: an 18-class gesture classifier and an auxiliary binary BFRB/non-BFRB head.

The training loss is class-weighted 18-way cross-entropy plus `0.2 *` auxiliary binary
cross-entropy. Optimization uses AdamW, ReduceLROnPlateau on validation hierarchical F1,
early stopping, and deterministic seeding.

## Setup

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

## Train

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
6. Download the small prepared JSON artifacts, `splits.json` and
   `label_encoder.json`, from `https://router.basaryilmaz.com/` when missing.
7. Try to pull remaining prepared artifacts from the DVC remote.
8. If prepared artifacts are missing from the remotes, run the DVC `prepare` and
   `splits` stages locally and push DVC-cached outputs to MinIO.
9. Start model training.

This keeps `splits.json` and `label_encoder.json` out of Git while every machine uses
the same train/validation/test split and label mapping.

## Common Training Overrides

Hydra overrides can be passed directly after the command:

```bash
uv run bfrb train training.max_epochs=60
uv run bfrb train training.seed=123
uv run bfrb train model.use_tof_raw=false
```

Training outputs include checkpoints under `artifacts/checkpoints/`, plots under
`plots/`, and metrics/artifacts in MLflow. After training, the best checkpoint is also
copied to `models/temporal_conv_gru_tof.ckpt`, tracked with DVC, and pushed to the
`bfrb-models` MinIO bucket. This creates `models/temporal_conv_gru_tof.ckpt.dvc`.

To fetch the latest DVC-tracked model from the s3:

```bash
uv run dvc pull models/temporal_conv_gru_tof.ckpt.dvc -r bfrb-models
```

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
- `splits`: fixed subject-disjoint stratified train/validation/test assignments

### Run Individual Data Stages

```bash
uv run bfrb prepare
uv run bfrb splits
```

Use these only when you explicitly want to run one stage outside the full DVC pipeline.

## Baseline Model

An IMU-only XGBoost baseline runs in parallel to the neural model: both start from the
same `splits.json` and report the same `val_*` / `test_*` metric names and definitions,
so their scores are directly comparable. It collapses each variable-length sequence into
per-channel summary statistics over the IMU channels (no thermopile / ToF).

```bash
uv run bfrb train_baseline
uv run bfrb train_baseline baseline.xgboost.max_depth=8
```

Baseline training similarly creates `models/xgboost_baseline.joblib.dvc` and pushes the
actual `models/xgboost_baseline.joblib` file to `bfrb-models`. To restore it after the
pointer file exists:

```bash
uv run dvc pull models/xgboost_baseline.joblib.dvc -r bfrb-models
```
