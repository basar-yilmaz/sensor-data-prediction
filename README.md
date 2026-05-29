# BFRB Detection from Sensor Data

Multimodal time-series classification of Body-Focused Repetitive Behaviors (BFRBs)
from wrist-worn sensor data. The project ingests IMU, thermopile, and time-of-flight
signals from the [CMI — Detect Behavior with Sensor Data](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data)
Kaggle competition and classifies each sequence into one of 18 gesture classes.

---

## Problem Statement

Body-Focused Repetitive Behaviors (BFRBs) — including trichotillomania (hair
pulling), excoriation disorder (skin picking), and onychophagia (nail biting) —
affect roughly 1–5% of the global population and can cause significant physical
damage and emotional strain. Current clinical assessment relies on self-reporting,
which is subjective and prone to recall error.

This project builds a multimodal time-series classifier that distinguishes BFRB
gestures from routine gestures using wrist-worn sensor data from the Helios device.
The system fuses three sensor modalities (IMU, thermopile, time-of-flight), tolerates
missing modality streams, and is evaluated for generalization across unseen subjects.

The work is situated within the CMI — Detect Behavior with Sensor Data competition.
The competition has ended, so the dataset and benchmark solutions are available for
comparison.

## Data

The dataset is publicly available from the Child Mind Institute via the Kaggle
competition page.

- **Size:** ~574,945 sensor readings across ~8,151 sequences, stored as CSV (>1 GB,
  well above the 10 MB minimum).
- **Labels:** 18 classes combining gesture type (8 BFRB behaviors + non-BFRB
  gestures) and hand orientation. The distribution is imbalanced — BFRB-related
  gestures are ~44% of samples.
- **Missing data:** ~half of the test sequences contain only IMU data (THM and ToF
  intentionally removed); ~6.85% of sequences have partial sensor-level missingness
  within available modalities.

### Input format

Variable-length multimodal time-series per sequence, from three modalities:

- **IMU (7 channels):** 3-axis acceleration (`acc_x/y/z`) and a 4-component rotation
  quaternion (`rot_w/x/y/z`). Always present.
- **Thermopile / THM (5 channels):** five temperature sensors (`thm_1`–`thm_5`).
  Absent in ~half of test observations.
- **Time-of-Flight / ToF (320 channels):** five distance sensors at 8×8 spatial
  resolution (`tof_1`–`tof_5`, 320 pixels total). Also missing in ~half of the
  test set.

### Output format

A single class label per sequence, over the 18 gesture × orientation classes.

## Metrics

The primary metric is a hierarchical F1 score that mirrors the competition's
weighted F1: the mean of a **binary F1** (BFRB target vs. non-target) and a
**macro F1** over the collapsed gesture classes. F1 is appropriate because of the
class imbalance and balances precision and recall. Standard 18-class accuracy and
macro-F1 are also tracked.

Target ranges (single lightweight model, no ensemble): a hand-crafted-feature
gradient-boosting baseline lands near ~0.60 weighted F1; a single CNN/recurrent
model is expected around ~0.65–0.75. Competition winners used large multi-model
ensembles; matching those is a non-goal here.

## Validation

Stratified 5-fold cross-validation via scikit-learn's `StratifiedGroupKFold`, with
the **subject/participant ID as the grouping key** so no participant appears in both
train and validation folds. Stratification is on the target label to keep class
balance across folds. Fold indices are persisted and the random seed is fixed for
reproducibility.

## Modeling

Models share a common `forward(batch) -> ModelOutput` contract and are selected via the
Hydra `model` config group.

- **`temporal_conv_gru_tof`** (default) — a temporally-aware model over a 39-dim
  engineered feature vector (`imu` + `imu_derived` + `thm` + `tof_stats`): linear
  projection → residual Conv1D blocks (local motion patterns) → bidirectional GRU
  (gesture progression) → masked attention pooling → classifier. It adds a raw-ToF
  spatial branch: a per-timestep 2D CNN over the raw 5×8×8 ToF frames, whose
  embedding is fused with the engineered features (instead of relying only on the
  20 ToF summary statistics). The raw-ToF branch can be disabled with
  `model.use_tof_raw=false`.

Two training-time options address the class imbalance and the hierarchical metric,
and can be combined with any model:

- `training.class_weighting=sqrt_inv_freq` weights the loss by inverse class
  frequency (computed from the training fold only), up-weighting rare gestures.
- `training.aux_binary_weight=<λ>` adds an auxiliary binary (BFRB target vs.
  non-target) head; the total loss becomes `CE_18 + λ · CE_binary`.
- `model.use_demographics=true` (temporal model) fuses a per-subject demographics
  vector (age, sex, handedness, body measurements; continuous fields z-scored
  fold-wise) through a small MLP into the classifier head. The auxiliary binary
  head stays sensor-only.

Training uses PyTorch Lightning; configuration is Hydra-managed; experiments
(hyperparameters, git commit, metrics, and plot artifacts) are tracked in MLflow.

---

# Technical Guide

## Setup

Requires Python 3.11 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

This installs the package and all dependencies from the locked `uv.lock`. To enable
the git hooks for local development:

```bash
uv run pre-commit install
```

## Fetch Data

Data is versioned with DVC. The data remote is public and read-only, so pulling does
not require credentials.

```bash
uv run bfrb download
```

This materializes the raw and prepared data under `data/`. Training also auto-fetches
prepared data via DVC if it is missing, so this step is optional before training.

## Run the Data Pipeline

To regenerate the prepared data from the raw CSV (per-sequence parquet files, the
sequence index, the label encoder, and subject-disjoint stratified folds):

```bash
uv run dvc repro
```

Individual stages are also exposed as commands:

```bash
uv run bfrb prepare   # raw CSV -> per-sequence parquet + index + label encoder
uv run bfrb splits    # StratifiedGroupKFold fold assignments
```

## Train

Start the local MLflow tracking server (`127.0.0.1:8080`):

```bash
docker compose up -d mlflow
```

Run training (auto-fetches prepared data via DVC if missing):

```bash
uv run bfrb train
```

Hyperparameters are Hydra-managed; override any of them from the CLI. The default run
is the `temporal_conv_gru_tof` model with `class_weighting=sqrt_inv_freq` and
`aux_binary_weight=0.2`:

```bash
# default best config
uv run bfrb train

# longer run with a larger epoch budget (scheduler + early stopping are on by default)
uv run bfrb train training.max_epochs=60

# disable the raw-ToF spatial branch (engineered ToF stats only)
uv run bfrb train model.use_tof_raw=false

# Demographics ablation A (sensor-only, default) vs B (+ demographics branch)
uv run bfrb train                              # A
uv run bfrb train model.use_demographics=true  # B
```

For repeated experiment configs, use the named Hydra experiment group instead of
rewriting long override lists:

```bash
# Current raw-ToF, no-demographics baseline
uv run bfrb train +experiment=tof_no_demo

# Auxiliary binary loss sweep
uv run bfrb train +experiment=tof_no_demo_aux_01
uv run bfrb train +experiment=tof_no_demo_aux_02
uv run bfrb train +experiment=tof_no_demo_aux_03
uv run bfrb train +experiment=tof_no_demo_aux_05

# Keep the named config, but vary fold/seed when needed
uv run bfrb train +experiment=tof_no_demo training.fold=1 training.seed=123
```

After training, metric/loss curves and a confusion matrix are written to `plots/`
and logged to MLflow alongside the run's hyperparameters and git commit.
