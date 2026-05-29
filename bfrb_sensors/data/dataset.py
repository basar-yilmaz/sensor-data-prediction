"""PyTorch Dataset for prepared BFRB sensor sequences."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
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
        demographics_lookup=None,
        load_tof_raw: bool = True,
    ):
        self.prepared_dir = Path(prepared_dir)
        self.sequence_ids = list(sequence_ids)
        self.scaler = scaler
        self.label_encoder = label_encoder
        self.transform = transform
        self.demographics_lookup = demographics_lookup
        self.load_tof_raw = load_tof_raw

        index = pd.read_parquet(self.prepared_dir / "index.parquet")
        self._rows = index.set_index("sequence_id").loc[self.sequence_ids].to_dict(orient="index")

    def __len__(self) -> int:
        return len(self.sequence_ids)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        sequence_id = self.sequence_ids[i]
        row = self._rows[sequence_id]
        columns = ["imu", "imu_derived", "thm", "tof_stats"]
        if self.load_tof_raw:
            columns.append("tof")
        try:
            payload = pq.read_table(
                self.prepared_dir / "sequences" / f"{sequence_id}.parquet", columns=columns
            ).to_pydict()
        except pa.ArrowInvalid as exc:
            if "imu_derived" not in str(exc) and "tof_stats" not in str(exc):
                raise
            raise KeyError(
                "Prepared sequence is missing imu_derived/tof_stats. "
                "Refresh prepared data with `uv run bfrb download` or "
                "`uv run dvc repro prepare`."
            ) from exc

        imu = np.asarray(payload["imu"][0], dtype=np.float32)
        try:
            imu_derived = np.asarray(payload["imu_derived"][0], dtype=np.float32)
            tof_stats = np.asarray(payload["tof_stats"][0], dtype=np.float32)
        except KeyError as exc:
            raise KeyError(
                "Prepared sequence is missing imu_derived/tof_stats. "
                "Refresh prepared data with `uv run bfrb download` or "
                "`uv run dvc repro prepare`."
            ) from exc
        thm = np.asarray(payload["thm"][0], dtype=np.float32)
        tof = np.asarray(payload["tof"][0], dtype=np.float32) if self.load_tof_raw else None

        expected_shapes = {
            "imu": imu.ndim == 2 and imu.shape[-1] == 7,
            "imu_derived": imu_derived.ndim == 2 and imu_derived.shape[-1] == 7,
            "thm": thm.ndim == 2 and thm.shape[-1] == 5,
            "tof_stats": tof_stats.ndim == 2 and tof_stats.shape[-1] == 20,
        }
        if tof is not None:
            expected_shapes["tof"] = tof.ndim == 4 and tof.shape[1:] == (5, 8, 8)
        invalid = [name for name, is_valid in expected_shapes.items() if not is_valid]
        if not invalid:
            expected_length = int(row["length"])
            lengths = {
                "imu": imu.shape[0],
                "imu_derived": imu_derived.shape[0],
                "thm": thm.shape[0],
                "tof_stats": tof_stats.shape[0],
            }
            if tof is not None:
                lengths["tof"] = tof.shape[0]
            invalid.extend(name for name, length in lengths.items() if length != expected_length)
        if invalid:
            raise ValueError(
                f"Invalid sensor shape for sequence {sequence_id}: {', '.join(invalid)}"
            )

        has_thm = bool(row["has_thm"])
        has_tof = bool(row["has_tof"])

        imu = (imu - self._stat("imu_mean")) / self._stat("imu_std")
        thm = (thm - self._stat("thm_mean")) / self._stat("thm_std")

        sample: dict[str, Any] = {
            "imu": torch.as_tensor(imu, dtype=torch.float32),
            "imu_derived": torch.as_tensor(imu_derived, dtype=torch.float32),
            "thm": torch.as_tensor(thm, dtype=torch.float32),
            "tof_stats": torch.as_tensor(tof_stats, dtype=torch.float32),
            "label": torch.tensor(self.label_encoder.encode(str(row["gesture"])), dtype=torch.long),
            "has_thm": torch.tensor(has_thm, dtype=torch.bool),
            "has_tof": torch.tensor(has_tof, dtype=torch.bool),
            "length": torch.tensor(int(row["length"]), dtype=torch.long),
        }
        if tof is not None:
            sample["tof"] = torch.as_tensor(tof, dtype=torch.float32)
        if self.demographics_lookup is not None:
            sample["demographics"] = torch.as_tensor(
                self.demographics_lookup.vector(str(row["subject_id"])),
                dtype=torch.float32,
            )
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def _stat(self, key: str) -> np.ndarray:
        return np.asarray(self.scaler[key], dtype=np.float32)
