"""Download the raw ``train.csv`` over HTTP without loading it into memory."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

RAW_CSV_NAME = "train.csv"


def fetch_raw_dataset(url: str, raw_dir: Path) -> Path:
    """Stream ``train.csv`` from ``url`` into ``raw_dir``.

    The file is downloaded to ``train.csv.part`` first and atomically renamed only
    after curl succeeds, which avoids treating partial multi-GB downloads as valid
    raw data on the next run.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / RAW_CSV_NAME
    partial = target.with_suffix(target.suffix + ".part")

    try:
        logger.info("Downloading raw CSV from %s to %s", url, target)
        subprocess.run(
            [
                "curl",
                "-L",
                "--fail",
                "--retry",
                "3",
                "--continue-at",
                "-",
                "-o",
                str(partial),
                url,
            ],
            check=True,
        )
        partial.replace(target)
    except Exception:
        logger.warning("Raw CSV download failed; partial file kept at %s", partial)
        raise

    logger.info("Raw data ready at %s (%d bytes)", target, target.stat().st_size)
    return target
