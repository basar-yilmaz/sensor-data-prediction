"""DVC-backed data acquisition helpers for the remote MinIO storage."""

from __future__ import annotations

import logging
from pathlib import Path

from dvc.repo import Repo as DvcRepo

logger = logging.getLogger(__name__)

DATA_DVC_PUSH_ENV = "BFRB_DVC_PUSH_DATA"
PREPARED_DVC_TARGETS = ["prepare", "splits"]


def _data_dvc_push_enabled(push_to_dvc: bool | None = None) -> bool:
    if push_to_dvc is not None:
        return push_to_dvc
    import os

    value = os.getenv(DATA_DVC_PUSH_ENV)
    if value is None:
        return True
    return value.strip().lower() in {"1", "true", "yes", "on"}


def pull_dvc_data(
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
        logger.info("Pulling DVC targets %s from MinIO remote %r", targets, remote)
    else:
        logger.info("Pulling all data from MinIO DVC remote %r (repo=%s)", remote, repo_root)
    try:
        with DvcRepo(str(repo_root)) as repo:
            repo.pull(targets=targets, remote=remote)
    except Exception as exc:
        logger.error(
            "Failed to pull from DVC remote %r (%s). Check that the remote is configured "
            "(`dvc remote list`) and your credentials are valid.",
            remote,
            exc,
        )
        raise
    logger.info("DVC pull complete for MinIO remote %r", remote)


def ensure_raw_data(
    repo_root: Path,
    raw_csv: Path,
    remote: str = "bfrb-data",
) -> None:
    """Ensure the raw CSV is present, fetching it once if needed.

    Resolution order:
      1. already on disk -> no-op;
      2. pull the DVC-tracked raw CSV from the configured MinIO remote.
    """
    repo_root = Path(repo_root).resolve()
    raw_csv = Path(raw_csv)

    if raw_csv.exists():
        logger.info("Raw data present at %s; skipping fetch", raw_csv)
        return

    try:
        pull_dvc_data(repo_root, remote=remote, targets=[str(raw_csv)])
    except Exception as exc:
        logger.warning(
            "Could not pull raw data from MinIO DVC remote %r: %s",
            remote,
            exc,
        )

    if raw_csv.exists():
        return

    raise FileNotFoundError(
        f"Raw data missing at {raw_csv}; DVC remote {remote!r} did not restore it."
    )


def ensure_prepared_data(
    repo_root: Path,
    prepared_dir: Path,
    remote: str = "bfrb-data",
    push_to_dvc: bool | None = None,
) -> None:
    """Ensure prepared data is present, preferring remotes before repro.

    Prepared artifacts come from the DVC remote or local repro. Reproduced artifacts
    are pushed back to the configured MinIO remote when data pushes are enabled.
    """
    prepared_dir = Path(prepared_dir)
    dvc_required = [
        prepared_dir / "sequences",
        prepared_dir / "index.parquet",
    ]
    dvc_json_required = [
        prepared_dir / "label_encoder.json",
        prepared_dir / "splits.json",
    ]
    required = dvc_required + dvc_json_required

    missing = [path for path in required if not path.exists()]
    if not missing:
        logger.info("Prepared data present; skipping fetch/repro")
        return

    logger.info(
        "Prepared data missing (%s); trying DVC targets %s from remote %r first",
        ", ".join(path.name for path in missing),
        PREPARED_DVC_TARGETS,
        remote,
    )
    try:
        pull_dvc_data(repo_root, remote=remote, targets=PREPARED_DVC_TARGETS)
    except Exception:
        logger.warning(
            "Could not pull prepared data from remote %r; continuing with local repro.",
            remote,
        )

    missing = [path for path in required if not path.exists()]
    if not missing:
        logger.info("Prepared data restored from MinIO DVC remote")
        return

    logger.info(
        "Prepared data missing (%s); running dvc repro for prepare+splits",
        ", ".join(path.name for path in missing),
    )
    with DvcRepo(str(Path(repo_root).resolve())) as repo:
        repo.reproduce(targets=["prepare", "splits"])
        if _data_dvc_push_enabled(push_to_dvc):
            logger.info("Caching prepared data in DVC remote %r", remote)
            repo.push(targets=["prepare", "splits"], remote=remote)
        else:
            logger.info(
                "Prepared data reproduced locally; skipping DVC push because data remote writes are disabled"
            )
    logger.info("dvc repro complete; prepared data is ready")
