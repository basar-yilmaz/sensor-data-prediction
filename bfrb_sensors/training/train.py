"""Training orchestration entry point."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import requests
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from bfrb_sensors.data.datamodule import BFRBDataModule, DataModuleConfig
from bfrb_sensors.data.download import download_data
from bfrb_sensors.data.label_encoder import LabelEncoder
from bfrb_sensors.training.metrics import HierarchyMapping
from bfrb_sensors.training.module import BFRBClassificationModule

logger = logging.getLogger(__name__)


def check_mlflow_server(uri: str) -> None:
    try:
        response = requests.get(uri, timeout=3)
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            f"MLflow server is not reachable at {uri}. Start it with: docker compose up -d mlflow"
        ) from exc


def git_state() -> dict[str, object]:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "status", "--short"], text=True).strip()
    return {"sha": sha, "dirty": bool(status)}


def make_checkpoint_callback(checkpoint_dir: Path, monitor: str, mode: str) -> ModelCheckpoint:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch:02d}-{val_hierarchical_f1:.4f}",
        monitor=monitor,
        mode=mode,
        save_top_k=1,
    )


def _datamodule_config(cfg: DictConfig) -> DataModuleConfig:
    return DataModuleConfig(
        prepared_dir=Path(cfg.data.datamodule.prepared_dir),
        artifacts_dir=Path(cfg.data.datamodule.artifacts_dir),
        fold_idx=int(cfg.training.fold),
        batch_size=int(cfg.training.batch_size),
        num_workers=int(cfg.training.num_workers),
        p_thm=float(cfg.data.datamodule.p_thm),
        p_tof=float(cfg.data.datamodule.p_tof),
        pin_memory=bool(cfg.data.datamodule.pin_memory),
        persistent_workers=bool(cfg.data.datamodule.persistent_workers),
    )


def train_from_config(cfg: DictConfig) -> None:
    if bool(cfg.mlflow.require_server):
        logger.info("Checking MLflow server at %s", cfg.mlflow.tracking_uri)
        check_mlflow_server(str(cfg.mlflow.tracking_uri))

    repo_root = Path(__file__).resolve().parents[2]
    logger.info("Ensuring prepared data is available via DVC")
    download_data(repo_root=repo_root)

    pl.seed_everything(int(cfg.training.seed), workers=True)
    dm = BFRBDataModule(_datamodule_config(cfg))
    dm.prepare_data()
    dm.setup("fit")

    prepared_dir = Path(cfg.data.datamodule.prepared_dir)
    index = pd.read_parquet(prepared_dir / "index.parquet")
    encoder = LabelEncoder.load(prepared_dir / "label_encoder.json")
    hierarchy = HierarchyMapping.from_index(index, encoder)

    module = BFRBClassificationModule(
        input_dim=int(cfg.model.input_dim),
        hidden_dim=int(cfg.model.hidden_dim),
        num_classes=int(cfg.model.num_classes),
        dropout=float(cfg.model.dropout),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
        hierarchy=hierarchy,
    )

    mlf_logger = MLFlowLogger(
        experiment_name=str(cfg.mlflow.experiment_name),
        tracking_uri=str(cfg.mlflow.tracking_uri),
    )
    state = git_state()
    mlf_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    mlf_logger.log_hyperparams({"git_sha": state["sha"], "git_dirty": state["dirty"]})
    logger.info("Training at git sha %s (dirty=%s)", state["sha"], state["dirty"])

    checkpoint = make_checkpoint_callback(
        Path(cfg.training.checkpoint_dir),
        monitor=str(cfg.training.monitor),
        mode=str(cfg.training.monitor_mode),
    )

    trainer = pl.Trainer(
        max_epochs=int(cfg.training.max_epochs),
        accelerator=str(cfg.training.accelerator),
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        overfit_batches=float(cfg.training.overfit_batches),
        logger=mlf_logger,
        callbacks=[checkpoint],
        deterministic=True,
    )
    logger.info("Starting training for %d epochs", int(cfg.training.max_epochs))
    trainer.fit(module, datamodule=dm)
    logger.info("Training complete; best checkpoint: %s", checkpoint.best_model_path)
