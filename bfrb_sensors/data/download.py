"""Wrapper around `dvc pull` for grader-friendly one-command data acquisition."""

from __future__ import annotations

import logging
from pathlib import Path

from dvc.repo import Repo as DvcRepo

logger = logging.getLogger(__name__)


def download_data(repo_root: Path, remote: str = "bfrb-data") -> None:
    repo_root = Path(repo_root).resolve()
    logger.info("Pulling data from DVC remote %r (repo=%s)", remote, repo_root)
    try:
        with DvcRepo(str(repo_root)) as repo:
            repo.pull(remote=remote)
    except Exception:
        logger.exception(
            "Failed to pull from DVC remote %r. Check that the remote is configured "
            "(`dvc remote list`) and your credentials are valid.",
            remote,
        )
        raise
    logger.info("DVC pull complete for remote %r", remote)
