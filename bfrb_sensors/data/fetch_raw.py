"""Download the raw dataset archive over HTTP (curl) and extract ``train.csv``.

The dataset is a public mirror of the competition data and is downloadable without
credentials, so no Kaggle SDK / API token is required.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

RAW_CSV_NAME = "train.csv"


def fetch_raw_dataset(url: str, raw_dir: Path) -> Path:
    """Download the dataset zip from ``url`` with curl and extract ``train.csv`` into ``raw_dir``.

    Returns the path to the extracted ``train.csv``. The downloaded archive is written
    next to the data (not /tmp, which may be a small tmpfs) and removed afterwards.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / RAW_CSV_NAME
    zip_path = raw_dir / "_raw_dataset.zip"

    try:
        logger.info("Downloading raw dataset from %s", url)
        subprocess.run(
            ["curl", "-L", "--fail", "--retry", "3", "-o", str(zip_path), url],
            check=True,
        )

        logger.info("Extracting %s from %s", RAW_CSV_NAME, zip_path.name)
        with zipfile.ZipFile(zip_path) as zf:
            member = next((m for m in zf.namelist() if Path(m).name == RAW_CSV_NAME), None)
            if member is None:
                raise FileNotFoundError(
                    f"{RAW_CSV_NAME} not found in archive from {url}; "
                    f"got {zf.namelist()[:10]}. If this looks like an HTML page, the URL "
                    "may require authentication."
                )
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    finally:
        zip_path.unlink(missing_ok=True)

    logger.info("Raw data ready at %s (%d bytes)", target, target.stat().st_size)
    return target
