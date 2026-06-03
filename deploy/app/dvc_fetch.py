"""Best-effort DVC fetch for inference artifacts stored in the remote.

The deploy service is otherwise self-contained, but some artifacts live only in
the DVC remote rather than on local disk — notably the label encoder produced by
the training ``prepare`` stage (``data/prepared/label_encoder.json``). When such
a file is missing we pull just that target from the configured remote so the
service can come up without a full retrain.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_dvc_artifact(path: Path, repo_root: Path, remote: str = "bfrb-data") -> bool:
    """Pull a single DVC-tracked file from ``remote`` when it is absent locally.

    Returns ``True`` if the file ends up present (already on disk or successfully
    pulled), ``False`` otherwise. This never raises: a missing remote, missing
    credentials, or absent ``dvc`` install should degrade to the caller's own
    "not found" handling rather than crashing startup.
    """
    path = Path(path)
    if path.exists():
        return True

    try:
        from dvc.repo import Repo as DvcRepo
    except ImportError:
        logger.warning("dvc is not installed; cannot fetch %s from remote %r", path, remote)
        return False

    repo_root = Path(repo_root).resolve()
    logger.info("Artifact %s missing; pulling from DVC remote %r", path, remote)
    try:
        with DvcRepo(str(repo_root)) as repo:
            repo.pull(targets=[str(path)], remote=remote)
    except Exception as exc:
        logger.warning("Failed to pull %s from DVC remote %r: %s", path, remote, exc)
        return False

    if path.exists():
        logger.info("Restored %s from DVC remote %r", path, remote)
        return True
    return False
