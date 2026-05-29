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


def ensure_prepared_data(
    repo_root: Path,
    prepared_dir: Path,
    require_demographics: bool,
) -> None:
    """Run the prepare+splits DVC stages if their outputs are missing.

    Idempotent: when the required prepared artifacts already exist this is a no-op,
    so it is cheap to call on every training run. ``demographics.parquet`` is only
    required when demographics are configured.
    """
    prepared_dir = Path(prepared_dir)
    required = [prepared_dir / "index.parquet", prepared_dir / "splits.json"]
    if require_demographics:
        required.append(prepared_dir / "demographics.parquet")

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
