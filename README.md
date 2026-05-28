# BFRB Sensors

Multimodal time-series classification of Body-Focused Repetitive Behaviors (BFRBs) from wrist-worn sensor data. The project uses IMU, thermopile, and time-of-flight signals from the CMI Kaggle competition _Detect Behavior with Sensor Data_.

## Setup

Requires Python 3.11 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Fetch Data

The DVC data remote is public and read-only, so pulling data does not require credentials.

```bash
uv run bfrb download
```

This materializes the raw and prepared data under `data/`.

## Run Pipeline

Run the full DVC data pipeline:

```bash
uv run dvc repro
```

The pipeline prepares per-sequence parquet files, writes the sequence index and label encoder, and builds subject-disjoint stratified splits under `data/prepared/`.
