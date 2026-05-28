"""PyTorch Dataset for prepared BFRB sensor sequences."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


class BFRBDataset(Dataset):
    def __init__(
        self,
        prepared_dir: Path,
        sequence_ids: list[str],
        scaler,
        label_encoder,
        transform=None,
    ):
        self.prepared_dir = Path(prepared_dir)
        self.sequence_ids = list(sequence_ids)
        self.scaler = scaler
        self.label_encoder = label_encoder
        self.transform = transform

        index = pd.read_parquet(self.prepared_dir / "index.parquet")
        self._rows = index.set_index("sequence_id").loc[self.sequence_ids].to_dict(orient="index")

    def __len__(self) -> int:
        return len(self.sequence_ids)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        sequence_id = self.sequence_ids[i]
        row = self._rows[sequence_id]
        payload = pq.read_table(
            self.prepared_dir / "sequences" / f"{sequence_id}.parquet"
        ).to_pydict()

        imu = np.asarray(payload["imu"][0], dtype=np.float32)
        thm = np.asarray(payload["thm"][0], dtype=np.float32)
        tof = np.asarray(payload["tof"][0], dtype=np.float32)

        has_thm = bool(row["has_thm"])
        has_tof = bool(row["has_tof"])

        imu = (imu - self._stat("imu_mean")) / self._stat("imu_std")
        thm = (thm - self._stat("thm_mean")) / self._stat("thm_std")

        sample: dict[str, Any] = {
            "imu": torch.as_tensor(imu, dtype=torch.float32),
            "thm": torch.as_tensor(thm, dtype=torch.float32),
            "tof": torch.as_tensor(tof, dtype=torch.float32),
            "label": torch.tensor(self.label_encoder.encode(str(row["gesture"])), dtype=torch.long),
            "has_thm": torch.tensor(has_thm, dtype=torch.bool),
            "has_tof": torch.tensor(has_tof, dtype=torch.bool),
            "length": torch.tensor(int(row["length"]), dtype=torch.long),
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def _stat(self, key: str) -> np.ndarray:
        return np.asarray(self.scaler[key], dtype=np.float32)
