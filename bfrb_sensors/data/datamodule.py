"""Lightning DataModule for prepared BFRB sensor sequences."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from bfrb_sensors.data.collate import ModalityDropout, pad_collate
from bfrb_sensors.data.dataset import BFRBDataset
from bfrb_sensors.data.demographics import (
    DemographicsLookup,
    demographics_stats_path,
    fit_demographics_stats,
    load_demographics_stats,
)
from bfrb_sensors.data.label_encoder import LabelEncoder, build_label_encoder
from bfrb_sensors.data.scaler import ScalerConfig, fit_scaler, load_scaler, scaler_path


@dataclass(frozen=True)
class DataModuleConfig:
    prepared_dir: Path
    artifacts_dir: Path
    fold_idx: int = 0
    batch_size: int = 32
    num_workers: int = 4
    p_thm: float = 0.5
    p_tof: float = 0.5
    pin_memory: bool = True
    persistent_workers: bool = True


def _seed_worker(worker_id: int) -> None:
    import numpy as np

    np.random.seed((torch.initial_seed() + worker_id) % 2**32)


class _BatchTransform:
    def __init__(self, dropout: ModalityDropout | None = None) -> None:
        self.dropout = dropout

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        collated = pad_collate(batch)
        if self.dropout is not None:
            return self.dropout(collated)
        return collated


class BFRBDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DataModuleConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_dataset: BFRBDataset | None = None
        self.val_dataset: BFRBDataset | None = None

    def prepare_data(self) -> None:
        prepared_dir = Path(self.cfg.prepared_dir)
        for filename in ("index.parquet", "splits.json"):
            path = prepared_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"required prepared artifact not found: {path}")

        scaler_cfg = self._scaler_config()
        if not scaler_path(scaler_cfg).exists():
            splits = self._load_splits()
            fit_scaler(scaler_cfg, splits["train"])

        demographics_parquet = self._demographics_parquet()
        if demographics_parquet.exists():
            stats_path = demographics_stats_path(self.cfg.artifacts_dir, self.cfg.fold_idx)
            if not stats_path.exists():
                splits = self._load_splits()
                index = pd.read_parquet(prepared_dir / "index.parquet")
                fit_demographics_stats(
                    demographics_parquet,
                    index,
                    splits["train"],
                    fold_idx=self.cfg.fold_idx,
                    artifacts_dir=self.cfg.artifacts_dir,
                )

    def setup(self, stage: str | None = None) -> None:
        splits = self._load_splits()
        prepared_dir = Path(self.cfg.prepared_dir)
        index = pd.read_parquet(prepared_dir / "index.parquet")
        label_encoder_path = prepared_dir / "label_encoder.json"
        if label_encoder_path.exists():
            label_encoder = LabelEncoder.load(label_encoder_path)
        else:
            label_encoder = build_label_encoder(index["gesture"].astype(str))

        scaler = load_scaler(scaler_path(self._scaler_config()))

        demographics_parquet = self._demographics_parquet()
        demographics_lookup = None
        if demographics_parquet.exists():
            stats_path = demographics_stats_path(self.cfg.artifacts_dir, self.cfg.fold_idx)
            if not stats_path.exists():
                raise FileNotFoundError(
                    f"demographics stats not found at {stats_path}; "
                    "call prepare_data() before setup() to fit fold-wise stats."
                )
            stats = load_demographics_stats(stats_path)
            demographics_lookup = DemographicsLookup(demographics_parquet, stats)

        if stage in (None, "fit", "train", "validate"):
            self.train_dataset = BFRBDataset(
                prepared_dir=prepared_dir,
                sequence_ids=splits["train"],
                scaler=scaler,
                label_encoder=label_encoder,
                demographics_lookup=demographics_lookup,
            )
            self.val_dataset = BFRBDataset(
                prepared_dir=prepared_dir,
                sequence_ids=splits["val"],
                scaler=scaler,
                label_encoder=label_encoder,
                demographics_lookup=demographics_lookup,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("setup('fit') must be called before train_dataloader")
        dropout = ModalityDropout(p_thm=self.cfg.p_thm, p_tof=self.cfg.p_tof)
        return self._dataloader(
            self.train_dataset, shuffle=True, collate_fn=_BatchTransform(dropout)
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("setup('fit') must be called before val_dataloader")
        return self._dataloader(self.val_dataset, shuffle=False, collate_fn=pad_collate)

    def _dataloader(self, dataset: BFRBDataset, shuffle: bool, collate_fn) -> DataLoader:
        kwargs: dict[str, Any] = {
            "batch_size": self.cfg.batch_size,
            "shuffle": shuffle,
            "num_workers": self.cfg.num_workers,
            "pin_memory": self.cfg.pin_memory and torch.cuda.is_available(),
            "collate_fn": collate_fn,
        }
        if self.cfg.num_workers > 0:
            kwargs["worker_init_fn"] = _seed_worker
            kwargs["persistent_workers"] = self.cfg.persistent_workers
        return DataLoader(dataset, **kwargs)

    def _demographics_parquet(self) -> Path:
        return Path(self.cfg.prepared_dir) / "demographics.parquet"

    def _scaler_config(self) -> ScalerConfig:
        return ScalerConfig(
            prepared_dir=Path(self.cfg.prepared_dir),
            artifacts_dir=Path(self.cfg.artifacts_dir),
            fold_idx=self.cfg.fold_idx,
        )

    def _load_splits(self) -> dict[str, list[str]]:
        path = Path(self.cfg.prepared_dir) / "splits.json"
        splits = json.loads(path.read_text())
        try:
            return splits[str(self.cfg.fold_idx)]
        except KeyError as exc:
            raise KeyError(f"fold {self.cfg.fold_idx} not found in {path}") from exc
