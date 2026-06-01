"""Lightning DataModule for prepared BFRB sensor sequences."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from bfrb_sensors.data.collate import ModalityDropout, pad_collate
from bfrb_sensors.data.dataset import BFRBDataset
from bfrb_sensors.data.label_encoder import LabelEncoder, build_label_encoder
from bfrb_sensors.data.scaler import ScalerConfig, fit_scaler, load_scaler, scaler_path
from bfrb_sensors.data.splits import load_splits


@dataclass(frozen=True)
class DataModuleConfig:
    prepared_dir: Path
    artifacts_dir: Path
    batch_size: int = 32
    num_workers: int = 4
    p_thm: float = 0.5
    p_tof: float = 0.5
    pin_memory: bool = True
    persistent_workers: bool = True
    load_tof_raw: bool = True


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
        self.test_dataset: BFRBDataset | None = None

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

        def _make(sequence_ids: list[str]) -> BFRBDataset:
            return BFRBDataset(
                prepared_dir=prepared_dir,
                sequence_ids=sequence_ids,
                scaler=scaler,
                label_encoder=label_encoder,
                load_tof_raw=self.cfg.load_tof_raw,
            )

        if stage in (None, "fit", "train", "validate"):
            self.train_dataset = _make(splits["train"])
            self.val_dataset = _make(splits["val"])
        if stage in (None, "fit", "test"):
            # Held-out test split, scored once at the end of training.
            self.test_dataset = _make(splits["test"])

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

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("setup('fit') or setup('test') must run before test_dataloader")
        return self._dataloader(self.test_dataset, shuffle=False, collate_fn=pad_collate)

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

    def _scaler_config(self) -> ScalerConfig:
        return ScalerConfig(
            prepared_dir=Path(self.cfg.prepared_dir),
            artifacts_dir=Path(self.cfg.artifacts_dir),
        )

    def _load_splits(self) -> dict[str, list[str]]:
        return load_splits(self.cfg.prepared_dir)
