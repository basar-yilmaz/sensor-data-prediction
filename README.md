# BFRB Sensors

Multimodal time-series classification of Body-Focused Repetitive Behaviors (BFRBs) — hair-pulling, skin-picking, nail-biting — from a wrist-worn Helios device. Source: CMI Kaggle competition _Detect Behavior with Sensor Data_.

The model receives variable-length sequences with three modalities (IMU 7ch, Thermopile 5ch, Time-of-Flight 5×8×8 = 320ch) and predicts one of 18 classes (gesture × hand orientation). About half the test sequences contain only IMU; the data pipeline simulates that with train-time modality dropout.

## Setup

Requires Python 3.11 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pre-commit install
```

Run hooks once to confirm the environment is clean:

```bash
uv run pre-commit run --all-files
```

Run the test suite:

```bash
uv run pytest -v
```

## Data layer

Two DVC remotes back the project: `bfrb-data` (raw + prepared) and `bfrb-models` (scalers and, later, training artifacts), both on Google Drive. Configuration lives in `.dvc/config`; you must have read access to both folders.

### Download

```bash
uv run bfrb download
```

This calls `dvc pull -r bfrb-data` under the hood. If you do not have access to the remote, download the CMI competition data manually from Kaggle into `data/raw/`.

### Prepare

Convert raw CMI CSV into per-sequence parquet plus a sequence-level index and label encoder:

```bash
uv run bfrb prepare
```

Outputs land under `data/prepared/`:

- `sequences/{sequence_id}.parquet` — one file per sequence with `imu (T,7)`, `thm (T,5)`, `tof (T,5,8,8)`
- `index.parquet` — one row per sequence with subject, gesture, length, modality flags, NaN fractions
- `label_encoder.json` — deterministic gesture → int mapping

Override defaults with Hydra:

```bash
uv run bfrb prepare data.prepare.min_length=20 data.prepare.verbose=true
```

### Splits

Build subject-disjoint, gesture-stratified 5-fold splits:

```bash
uv run bfrb splits
```

Writes `data/prepared/splits.json`.

### Scaler

The per-fold `StandardScaler` on IMU + THM is fit lazily the first time the `BFRBDataModule` is set up. To fit eagerly:

```bash
uv run bfrb warm_scaler data.datamodule.fold_idx=0
```

Scalers land in `artifacts/scaler_fold{idx}.joblib`.

### DVC pipeline

Both `prepare` and `splits` are also wrapped as DVC stages — see `dvc.yaml`. To re-run the whole pipeline reproducibly:

```bash
uv run dvc repro
```

## Project layout

```
bfrb_sensors/
├── data/                # data pipeline (this revision)
│   ├── prepare.py       # raw CSV → per-sequence parquet + index + label encoder
│   ├── splits.py        # StratifiedGroupKFold → splits.json
│   ├── scaler.py        # per-fold StandardScaler
│   ├── dataset.py       # BFRBDataset (PyTorch Dataset)
│   ├── collate.py       # pad_collate + ModalityDropout
│   ├── datamodule.py    # BFRBDataModule (LightningDataModule)
│   ├── download.py      # dvc pull wrapper
│   └── label_encoder.py # deterministic gesture-string ↔ int
├── commands.py          # CLI entry point (fire + Hydra Compose)
configs/                 # Hydra config tree
tests/data/              # pytest tests for the data layer
```

## Roadmap

This revision lands the data layer for MLOps Task 2. The training pipeline (LightningModule, training loop, MLflow logging, ≥3 logged metric plots, deployment) is the subject of follow-up specs.
