"""Runtime settings for the deploy service.

All values are loaded from environment variables (or a ``.env`` file) via
pydantic-settings. None of the paths are required to exist at import time; the
service fails loudly at startup if the configured checkpoint / scaler /
label-encoder cannot be found.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BFRB_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = 8000
    log_level: Literal["debug", "info", "warning", "error"] = "info"
    reload: bool = False
    # Inference device. Defaults to CPU; set BFRB_DEVICE=cuda to use the GPU.
    device: Literal["cpu", "cuda"] = "cpu"

    model_checkpoint: Path = Field(
        default=PROJECT_ROOT / "models" / "temporal_conv_gru_tof.ckpt",
        description=(
            "Path to a Lightning checkpoint file or to a directory; the newest "
            "matching *.ckpt is auto-selected when a directory is given. When the "
            "configured file is missing it is pulled from the model DVC remote."
        ),
    )
    scaler_path: Path = PROJECT_ROOT / "artifacts" / "scaler.joblib"
    label_encoder_path: Path = PROJECT_ROOT / "data" / "prepared" / "label_encoder.json"
    sample_data_path: Path = PROJECT_ROOT / "deploy" / "sample_data" / "demo_sequence.csv"

    repo_root: Path = PROJECT_ROOT
    # DVC remotes used to restore artifacts that are not on local disk.
    data_remote: str = "bfrb-data"  # label encoder (prepare stage output)
    model_remote: str = "bfrb-models"  # trained checkpoint

    hidden_dim: int = 128
    num_conv_blocks: int = 2
    gru_layers: int = 1
    dropout: float = 0.2
    use_tof_raw: bool = True
    tof_embed_dim: int = 32
    num_classes: int = 18
    input_dim: int = 39

    max_seq_length: int = 4096
    min_seq_length: int = 10
    top_k: int = 5
    nan_threshold: float = 0.5

    @field_validator("model_checkpoint", "scaler_path", "label_encoder_path", mode="after")
    @classmethod
    def _ensure_absolute(cls, value: Path) -> Path:
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @field_validator("sample_data_path", mode="after")
    @classmethod
    def _ensure_absolute_sample(cls, value: Path) -> Path:
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
