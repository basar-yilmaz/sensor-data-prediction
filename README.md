# BFRB Sensors

Multimodal time-series classification of Body-Focused Repetitive Behaviors (BFRBs) from a wrist-worn Helios device, using IMU, thermopile, and time-of-flight sensors. Source: CMI Kaggle competition *Detect Behavior with Sensor Data*.

## Setup

Requires Python 3.11 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pre-commit install
```

## Data

Raw data lives on a DVC remote backed by Google Drive. Once credentials are configured, pull with:

```bash
uv run bfrb download
```

## Train

(Coming in the training pipeline iteration — this revision builds the data layer only.)

Prepare and split the data:

```bash
uv run bfrb prepare
uv run bfrb splits
```
