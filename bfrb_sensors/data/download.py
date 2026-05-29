"""Wrapper around `dvc pull` for grader-friendly one-command data acquisition."""

from __future__ import annotations

import logging
from pathlib import Path

from dvc.repo import Repo as DvcRepo

logger = logging.getLogger(__name__)


def download_data(
    repo_root: Path,
    remote: str = "bfrb-data",
    targets: list[str] | None = None,
) -> None:
    """Pull DVC-tracked data from ``remote``.

    When ``targets`` is given, only those paths are pulled (e.g. the raw inputs),
    which avoids failing on pipeline outputs that were never pushed — those are
    rebuilt locally by :func:`ensure_prepared_data`. With ``targets=None`` the full
    workspace is pulled (used by the standalone ``bfrb download`` command).
    """
    repo_root = Path(repo_root).resolve()
    if targets:
        logger.info("Pulling DVC targets %s from remote %r", targets, remote)
    else:
        logger.info("Pulling all data from DVC remote %r (repo=%s)", remote, repo_root)
    try:
        with DvcRepo(str(repo_root)) as repo:
            repo.pull(targets=targets, remote=remote)
    except Exception:
        logger.exception(
            "Failed to pull from DVC remote %r. Check that the remote is configured "
            "(`dvc remote list`) and your credentials are valid.",
            remote,
        )
        raise
    logger.info("DVC pull complete for remote %r", remote)


def ensure_raw_data(
    repo_root: Path,
    raw_csv: Path,
    dataset_url: str,
    remote: str = "bfrb-data",
) -> None:
    """Ensure the raw CSV is present, fetching it once if needed.

    Resolution order:
      1. already on disk -> no-op;
      2. pull from the DVC remote (MinIO);
      3. download the dataset zip over HTTP, then ``dvc add`` + ``dvc push`` to cache
         it in MinIO.

    This realizes the "download once, then cache in MinIO" flow: the first run on a
    fresh remote hits the dataset URL, every later run pulls from MinIO.
    """
    repo_root = Path(repo_root).resolve()
    raw_csv = Path(raw_csv)

    if raw_csv.exists():
        logger.info("Raw data present at %s; skipping fetch", raw_csv)
        return

    try:
        download_data(repo_root, remote=remote, targets=[str(raw_csv)])
    except Exception:
        logger.warning(
            "Could not pull raw data from remote %r; falling back to dataset download.",
            remote,
        )

    if raw_csv.exists():
        return

    from bfrb_sensors.data.fetch_raw import fetch_raw_dataset

    logger.info("Raw data not in remote; downloading dataset from %s", dataset_url)
    fetch_raw_dataset(dataset_url, raw_csv.parent)

    logger.info("Caching raw data in DVC remote %r (dvc add + push)", remote)
    with DvcRepo(str(repo_root)) as repo:
        repo.add(str(raw_csv))
        repo.push(targets=[str(raw_csv)], remote=remote)
    logger.info("Raw data cached in remote %r", remote)


def ensure_prepared_data(
    repo_root: Path,
    prepared_dir: Path,
) -> None:
    """Run the prepare+splits DVC stages if their outputs are missing.

    Idempotent: when the required prepared artifacts already exist this is a no-op,
    so it is cheap to call on every training run.
    """
    prepared_dir = Path(prepared_dir)
    required = [prepared_dir / "index.parquet", prepared_dir / "splits.json"]

    missing = [path for path in required if not path.exists()]
    if not missing:
        logger.info("Prepared data present; skipping dvc repro")
        return

    logger.info(
        "Prepared data missing (%s); running dvc repro for prepare+splits",
        ", ".join(path.name for path in missing),
    )
    with DvcRepo(str(Path(repo_root).resolve())) as repo:
        repo.reproduce(targets=["prepare", "splits"])
    logger.info("dvc repro complete; prepared data is ready")
