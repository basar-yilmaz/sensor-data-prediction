"""CLI entry point. Uses Hydra Compose API to load configs, then dispatches via fire."""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import fire
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from bfrb_sensors.data.datamodule import BFRBDataModule, DataModuleConfig
from bfrb_sensors.data.download import download_data
from bfrb_sensors.data.prepare import PrepareConfig, prepare
from bfrb_sensors.data.splits import SplitsConfig, make_splits


def _load_config(overrides: list[str] | None = None) -> DictConfig:
    config_dir = (Path(__file__).resolve().parent.parent / "configs").resolve()
    overrides = overrides or []
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name="config", overrides=overrides)


def _configure_logging(cfg: DictConfig) -> None:
    log_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
    logging.config.dictConfig(log_cfg)


def _prepare_config_from_hydra(cfg: DictConfig) -> PrepareConfig:
    return PrepareConfig(
        raw_csv=Path(cfg.data.prepare.raw_csv),
        prepared_dir=Path(cfg.data.prepare.prepared_dir),
        min_length=int(cfg.data.prepare.min_length),
        nan_threshold=float(cfg.data.prepare.nan_threshold),
        verbose=bool(cfg.data.prepare.verbose),
        sequence_id_col=str(cfg.data.prepare.sequence_id_col),
        subject_id_col=str(cfg.data.prepare.subject_id_col),
        gesture_col=str(cfg.data.prepare.gesture_col),
        step_col=str(cfg.data.prepare.step_col),
        orientation_col=str(cfg.data.prepare.orientation_col),
        sequence_type_col=str(cfg.data.prepare.sequence_type_col),
        expected_n_classes=int(cfg.data.prepare.expected_n_classes),
    )


def _splits_config_from_hydra(cfg: DictConfig) -> SplitsConfig:
    return SplitsConfig(
        prepared_dir=Path(cfg.data.splits.prepared_dir),
        n_folds=int(cfg.data.splits.n_folds),
        seed=int(cfg.data.splits.seed),
        group_col=str(cfg.data.splits.group_col),
        stratify_col=str(cfg.data.splits.stratify_col),
    )


def _datamodule_config_from_hydra(cfg: DictConfig) -> DataModuleConfig:
    return DataModuleConfig(
        prepared_dir=Path(cfg.data.datamodule.prepared_dir),
        artifacts_dir=Path(cfg.data.datamodule.artifacts_dir),
        fold_idx=int(cfg.data.datamodule.fold_idx),
        batch_size=int(cfg.data.datamodule.batch_size),
        num_workers=int(cfg.data.datamodule.num_workers),
        p_thm=float(cfg.data.datamodule.p_thm),
        p_tof=float(cfg.data.datamodule.p_tof),
        pin_memory=bool(cfg.data.datamodule.pin_memory),
        persistent_workers=bool(cfg.data.datamodule.persistent_workers),
    )


class Commands:
    """`bfrb <command> [hydra-overrides ...]`"""

    def download(self, *overrides: str) -> None:
        cfg = _load_config(list(overrides))
        _configure_logging(cfg)
        repo_root = Path(__file__).resolve().parent.parent
        download_data(repo_root=repo_root)

    def prepare(self, *overrides: str) -> None:
        cfg = _load_config(list(overrides))
        _configure_logging(cfg)
        prepare(_prepare_config_from_hydra(cfg))

    def splits(self, *overrides: str) -> None:
        cfg = _load_config(list(overrides))
        _configure_logging(cfg)
        make_splits(_splits_config_from_hydra(cfg))

    def warm_scaler(self, *overrides: str) -> None:
        """Eagerly fit the scaler for the configured fold (otherwise lazy-fit on first train)."""
        cfg = _load_config(list(overrides))
        _configure_logging(cfg)
        dm = BFRBDataModule(_datamodule_config_from_hydra(cfg))
        dm.prepare_data()


def main() -> None:
    fire.Fire(Commands)


if __name__ == "__main__":
    main()
