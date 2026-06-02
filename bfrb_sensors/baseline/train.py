"""Baseline training orchestration (parallel to bfrb_sensors.training.train)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from bfrb_sensors.baseline.features import FeatureConfig, build_feature_matrix, feature_names
from bfrb_sensors.baseline.model import build_baseline_model
from bfrb_sensors.baseline.plots import write_baseline_plots
from bfrb_sensors.data.download import ensure_prepared_data, ensure_raw_data
from bfrb_sensors.data.label_encoder import LabelEncoder
from bfrb_sensors.data.splits import load_split_file, load_splits
from bfrb_sensors.training.class_weights import compute_class_weights
from bfrb_sensors.training.metrics import HierarchyMapping, evaluate_predictions
from bfrb_sensors.training.plots import write_confusion_matrix
from bfrb_sensors.training.train import check_mlflow_server, git_state, persist_best_model_to_dvc

logger = logging.getLogger(__name__)


def _feature_config(cfg: DictConfig) -> FeatureConfig:
    features = cfg.baseline.features
    return FeatureConfig(
        use_imu_raw=bool(features.use_imu_raw),
        use_imu_derived=bool(features.use_imu_derived),
        stats=tuple(str(stat) for stat in features.stats),
    )


def _sample_weights(
    index: pd.DataFrame,
    encoder: LabelEncoder,
    train_sequence_ids: list[str],
    y_train: np.ndarray,
    scheme: str,
    num_classes: int,
) -> np.ndarray | None:
    """Per-sample weights from the shared class-weighting scheme, or None."""
    class_weights = compute_class_weights(
        index, encoder, train_sequence_ids, scheme=scheme, num_classes=num_classes
    )
    if class_weights is None:
        return None
    return class_weights.numpy()[y_train]


def _split_hyperparams(metadata: dict[str, Any]) -> dict[str, Any]:
    return {f"split_{key}": value for key, value in metadata.items()}


def train_baseline_from_config(cfg: DictConfig) -> None:
    if bool(cfg.mlflow.require_server):
        logger.info("Checking MLflow server at %s", cfg.mlflow.tracking_uri)
        check_mlflow_server(str(cfg.mlflow.tracking_uri))

    repo_root = Path(__file__).resolve().parents[2]
    prepared_dir = Path(cfg.data.datamodule.prepared_dir)
    logger.info("Ensuring raw data is available (DVC remote or dataset download)")
    ensure_raw_data(repo_root, Path(cfg.data.prepare.raw_csv), str(cfg.data.download.url))
    if bool(cfg.data.auto_prepare):
        ensure_prepared_data(repo_root, prepared_dir)

    split_payload = load_split_file(prepared_dir)
    splits = load_splits(prepared_dir)

    index = pd.read_parquet(prepared_dir / "index.parquet")
    encoder = LabelEncoder.load(prepared_dir / "label_encoder.json")
    hierarchy = HierarchyMapping.from_index(index, encoder)
    gesture_by_sequence = dict(
        zip(index["sequence_id"].astype(str), index["gesture"].astype(str), strict=True)
    )

    feature_cfg = _feature_config(cfg)
    num_classes = int(cfg.baseline.num_classes)
    logger.info("Extracting IMU-only features")
    X_train, y_train = build_feature_matrix(
        prepared_dir, splits["train"], encoder, gesture_by_sequence, feature_cfg
    )
    X_val, y_val = build_feature_matrix(
        prepared_dir, splits["val"], encoder, gesture_by_sequence, feature_cfg
    )
    X_test, y_test = build_feature_matrix(
        prepared_dir, splits["test"], encoder, gesture_by_sequence, feature_cfg
    )

    logger.info("Class weighting scheme: %s", cfg.baseline.class_weighting)
    sample_weight = _sample_weights(
        index,
        encoder,
        splits["train"],
        y_train,
        scheme=str(cfg.baseline.class_weighting),
        num_classes=num_classes,
    )

    model = build_baseline_model(cfg.baseline, num_classes=num_classes, seed=int(cfg.baseline.seed))
    logger.info("Built baseline model %r", str(cfg.baseline.name))
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )
    logger.info(
        "Baseline training complete; best_iteration=%s", getattr(model, "best_iteration", None)
    )

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)
    scores = evaluate_predictions(
        y_val, y_pred, hierarchy, num_classes=num_classes, prefix="val", y_proba=y_proba
    )
    logger.info(
        "VAL: hierarchical_f1=%.4f accuracy=%.4f macro_f1_18=%.4f "
        "binary_precision=%.4f binary_recall=%.4f binary_f1=%.4f "
        "macro_f1_collapsed=%.4f log_loss=%.4f",
        scores["val_hierarchical_f1"],
        scores["val_accuracy"],
        scores["val_macro_f1_18"],
        scores["val_binary_precision"],
        scores["val_binary_recall"],
        scores["val_binary_f1"],
        scores["val_macro_f1_collapsed"],
        scores["val_log_loss"],
    )

    # Held-out test split, scored once at the end with the same metrics.
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)
    test_scores = evaluate_predictions(
        y_test,
        y_test_pred,
        hierarchy,
        num_classes=num_classes,
        prefix="test",
        y_proba=y_test_proba,
    )
    logger.info(
        "TEST: hierarchical_f1=%.4f accuracy=%.4f macro_f1_18=%.4f "
        "binary_precision=%.4f binary_recall=%.4f binary_f1=%.4f "
        "macro_f1_collapsed=%.4f log_loss=%.4f",
        test_scores["test_hierarchical_f1"],
        test_scores["test_accuracy"],
        test_scores["test_macro_f1_18"],
        test_scores["test_binary_precision"],
        test_scores["test_binary_recall"],
        test_scores["test_binary_f1"],
        test_scores["test_macro_f1_collapsed"],
        test_scores["test_log_loss"],
    )
    scores.update(test_scores)

    _log_run(
        cfg,
        model,
        scores,
        split_payload,
        X_train.shape[1],
        y_val,
        y_pred,
        y_test,
        y_test_pred,
        feature_cfg,
    )


def _log_run(
    cfg: DictConfig,
    model,
    scores: dict[str, float],
    split_payload: dict[str, Any],
    n_features: int,
    y_val: np.ndarray,
    y_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    feature_cfg: FeatureConfig,
) -> None:
    import mlflow

    mlflow.set_tracking_uri(str(cfg.mlflow.tracking_uri))
    mlflow.set_experiment(str(cfg.mlflow.experiment_name))
    names = feature_names(feature_cfg)

    with mlflow.start_run(run_name=f"baseline-{cfg.baseline.name}") as run:
        state = git_state()
        mlflow.log_params(OmegaConf.to_container(cfg.baseline, resolve=True))
        mlflow.log_params(
            {"git_sha": state["sha"], "git_dirty": state["dirty"], "n_features": n_features}
        )
        mlflow.log_params(_split_hyperparams(split_payload["metadata"]))
        mlflow.log_metrics(scores)
        logger.info("Logged baseline run %s (git sha %s)", run.info.run_id, state["sha"])

        model_path = Path(cfg.baseline.model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "feature_names": names}, model_path)
        logger.info("Saved baseline model to %s", model_path)

        repo_root = Path(__file__).resolve().parents[2]
        dvc_model_path = persist_best_model_to_dvc(
            str(model_path),
            repo_root=repo_root,
            model_registry_dir=Path(cfg.baseline.model_registry_dir),
            model_artifact_name=str(cfg.baseline.model_artifact_name),
            remote=str(cfg.baseline.model_dvc_remote),
            enabled=bool(cfg.baseline.push_model_to_dvc),
        )

        plots_dir = Path(cfg.baseline.plots_dir)
        plot_paths = write_baseline_plots(
            plots_dir,
            model.evals_result(),
            y_true=y_val.tolist(),
            y_pred=y_pred.tolist(),
            feature_names=names,
            importances=model.feature_importances_,
        )
        test_cm = write_confusion_matrix(
            plots_dir, y_test.tolist(), y_test_pred.tolist(), "confusion_matrix_test.png"
        )
        artifact_paths = [model_path, *plot_paths, test_cm]
        if dvc_model_path is not None:
            artifact_paths.append(dvc_model_path)
        for path in artifact_paths:
            try:
                mlflow.log_artifact(str(path))
                logger.info("Logged artifact to MLflow: %s", path)
            except Exception:
                logger.exception("Failed to log artifact %s to MLflow", path)
