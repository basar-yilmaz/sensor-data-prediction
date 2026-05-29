"""Training orchestration entry point."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import requests
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from bfrb_sensors.data.datamodule import BFRBDataModule, DataModuleConfig
from bfrb_sensors.data.download import download_data
from bfrb_sensors.data.label_encoder import LabelEncoder
from bfrb_sensors.models.factory import build_model
from bfrb_sensors.training.class_weights import compute_class_weights
from bfrb_sensors.training.metrics import HierarchyMapping
from bfrb_sensors.training.module import BFRBClassificationModule
from bfrb_sensors.training.plots import write_training_plots

logger = logging.getLogger(__name__)

_HISTORY_KEYS = (
    "train_loss",
    "val_loss",
    "val_accuracy",
    "val_macro_f1_18",
    "val_binary_f1",
    "val_macro_f1_collapsed",
    "val_hierarchical_f1",
)


class MetricsHistory(Callback):
    """Record epoch-level metrics so we can plot training curves after fit."""

    def __init__(self) -> None:
        self.history: dict[str, list[float]] = {key: [] for key in _HISTORY_KEYS}

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        for key in _HISTORY_KEYS:
            value = metrics.get(key)
            if value is not None:
                self.history[key].append(float(value))


@torch.no_grad()
def _collect_val_predictions(module: pl.LightningModule, dataloader) -> tuple[list[int], list[int]]:
    module.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for batch in dataloader:
        logits = module(batch).logits
        y_pred.extend(logits.argmax(dim=1).tolist())
        y_true.extend(batch["label"].tolist())
    return y_true, y_pred


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

    splits = json.loads((prepared_dir / "splits.json").read_text())
    train_sequence_ids = splits[str(int(cfg.training.fold))]["train"]
    class_weights = compute_class_weights(
        index,
        encoder,
        train_sequence_ids,
        scheme=str(cfg.training.class_weighting),
        num_classes=int(cfg.model.num_classes),
    )
    logger.info("Class weighting scheme: %s", cfg.training.class_weighting)

    model = build_model(cfg.model)
    logger.info("Built model %r", str(cfg.model.name))
    module = BFRBClassificationModule(
        model=model,
        num_classes=int(cfg.model.num_classes),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
        hierarchy=hierarchy,
        class_weights=class_weights,
    )

    mlf_logger = MLFlowLogger(
        experiment_name=str(cfg.mlflow.experiment_name),
        tracking_uri=str(cfg.mlflow.tracking_uri),
    )
    state = git_state()
    mlf_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    mlf_logger.log_hyperparams({"git_sha": state["sha"], "git_dirty": state["dirty"]})
    logger.info("Training at git sha %s (dirty=%s)", state["sha"], state["dirty"])

    run_checkpoint_dir = Path(cfg.training.checkpoint_dir) / mlf_logger.run_id
    logger.info("Checkpoints for this run: %s", run_checkpoint_dir)
    checkpoint = make_checkpoint_callback(
        run_checkpoint_dir,
        monitor=str(cfg.training.monitor),
        mode=str(cfg.training.monitor_mode),
    )

    history = MetricsHistory()
    trainer = pl.Trainer(
        max_epochs=int(cfg.training.max_epochs),
        accelerator=str(cfg.training.accelerator),
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        overfit_batches=float(cfg.training.overfit_batches),
        logger=mlf_logger,
        callbacks=[checkpoint, history],
        deterministic=True,
    )
    logger.info("Starting training for %d epochs", int(cfg.training.max_epochs))
    trainer.fit(module, datamodule=dm)
    logger.info("Training complete; best checkpoint: %s", checkpoint.best_model_path)

    _log_artifacts(cfg, mlf_logger, module, dm, history)


def _log_artifacts(
    cfg: DictConfig,
    mlf_logger: MLFlowLogger,
    module: pl.LightningModule,
    dm: BFRBDataModule,
    history: MetricsHistory,
) -> None:
    plots_dir = Path(cfg.training.plots_dir)
    y_true, y_pred = _collect_val_predictions(module.to("cpu"), dm.val_dataloader())
    if not y_true:
        logger.warning("No validation samples available; skipping plot generation")
        return

    paths = write_training_plots(plots_dir, history.history, y_true=y_true, y_pred=y_pred)
    repo_root = Path(__file__).resolve().parents[2]
    dvc_lock = repo_root / "dvc.lock"
    if dvc_lock.exists():
        paths = [*paths, dvc_lock]

    logger.info("Wrote %d plots to %s", len(paths) - int(dvc_lock.exists()), plots_dir)

    run_id = mlf_logger.run_id
    for path in paths:
        try:
            mlf_logger.experiment.log_artifact(run_id, str(path))
            logger.info("Logged artifact to MLflow: %s", path)
        except Exception:
            logger.exception(
                "Failed to log artifact %s to MLflow. The local copy is preserved under %s. "
                "Check that the server serves artifacts (`--serve-artifacts`).",
                path,
                plots_dir,
            )
