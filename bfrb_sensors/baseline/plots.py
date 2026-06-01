"""Baseline plot generation (parallel to bfrb_sensors.training.plots)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bfrb_sensors.training.plots import write_confusion_matrix  # noqa: E402


def write_baseline_plots(
    plots_dir: Path,
    evals_result: dict[str, dict[str, list[float]]],
    y_true: list[int],
    y_pred: list[int],
    feature_names: list[str],
    importances: np.ndarray,
    top_k: int = 25,
) -> list[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    # Boosting curve: train vs. validation mlogloss over rounds.
    curve_path = plots_dir / "loss.png"
    plt.figure()
    label_by_key = {"validation_0": "train_mlogloss", "validation_1": "val_mlogloss"}
    for key, metrics in evals_result.items():
        mlogloss = metrics.get("mlogloss")
        if mlogloss:
            plt.plot(mlogloss, label=label_by_key.get(key, key))
    plt.xlabel("boosting round")
    plt.ylabel("mlogloss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_path)
    plt.close()
    paths.append(curve_path)

    # Confusion matrix (row-normalized), same as the neural pipeline.
    paths.append(write_confusion_matrix(plots_dir, y_true, y_pred))

    # Top-k feature importances.
    importance_path = plots_dir / "feature_importance.png"
    order = np.argsort(importances)[::-1][:top_k]
    plt.figure(figsize=(8, max(4, 0.3 * len(order))))
    plt.barh([feature_names[i] for i in order][::-1], importances[order][::-1])
    plt.xlabel("importance (gain)")
    plt.tight_layout()
    plt.savefig(importance_path)
    plt.close()
    paths.append(importance_path)

    return paths
