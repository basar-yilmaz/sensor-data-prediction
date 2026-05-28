# BFRB Sensors

Multimodal time-series classification of Body-Focused Repetitive Behaviors (BFRBs) — hair-pulling, skin-picking, nail-biting — from a wrist-worn Helios device. Source: CMI Kaggle competition _Detect Behavior with Sensor Data_.

The model receives variable-length sequences with three modalities (IMU 7ch, Thermopile 5ch, Time-of-Flight 5×8×8 = 320ch) and predicts one of 18 gesture classes. Hand orientation (4 values) and sequence type (Target/Non-Target) are recorded as metadata, not modeled as the label. About half the test sequences contain only IMU; the data pipeline simulates that with train-time modality dropout.

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

Two DVC remotes back the project, both on Cloudflare R2 (S3-compatible): `bfrb-data` (raw + prepared) and `bfrb-models` (scalers and, later, training artifacts). The committed `.dvc/config` points each remote at its **public, read-only** R2 URL, so pulling needs no credentials or login.

### Download

```bash
uv run bfrb download
```

This pulls from the public `bfrb-data` remote — no authentication required. If the remote is ever unavailable, download the CMI competition data manually from Kaggle into `data/raw/`.

### Pushing data (maintainers only)

Writing to the remotes requires R2 API credentials, which are kept out of git in `.dvc/config.local`. Configure them once:

```bash
ENDPOINT=https://4c8600513f554fa0547f3ef7c9319540.r2.cloudflarestorage.com
for remote in bfrb-data bfrb-models; do
  uv run dvc remote modify --local "$remote" url "s3://$remote"
  uv run dvc remote modify --local "$remote" endpointurl "$ENDPOINT"
  uv run dvc remote modify --local "$remote" region auto
  uv run dvc remote modify --local "$remote" access_key_id "$R2_ACCESS_KEY_ID"
  uv run dvc remote modify --local "$remote" secret_access_key "$R2_SECRET_ACCESS_KEY"
done
```

The `--local` flag writes to `.dvc/config.local` (gitignored), overriding the public HTTPS URL with the authenticated `s3://` endpoint for your machine only. Then `uv run dvc push` / `uv run dvc push -r bfrb-models` upload via the S3 API.

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
