"""Training plot generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix  # noqa: E402


def write_confusion_matrix(
    plots_dir: Path,
    y_true: list[int],
    y_pred: list[int],
    filename: str = "confusion_matrix.png",
) -> Path:
    """Row-normalized confusion matrix; shared by every model and split."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / filename
    matrix = confusion_matrix(y_true, y_pred, normalize="true")
    ConfusionMatrixDisplay(matrix).plot(values_format=".2f")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def write_training_plots(
    plots_dir: Path,
    history: dict[str, list[float]],
    y_true: list[int],
    y_pred: list[int],
) -> list[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    loss_path = plots_dir / "loss.png"
    plt.figure()
    plt.plot(history.get("train_loss", []), label="train_loss")
    plt.plot(history.get("val_loss", []), label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path)
    plt.close()
    paths.append(loss_path)

    metric_path = plots_dir / "validation_metrics.png"
    plt.figure()
    for key in (
        "val_accuracy",
        "val_macro_f1_18",
        "val_binary_precision",
        "val_binary_recall",
        "val_binary_f1",
        "val_hierarchical_f1",
    ):
        plt.plot(history.get(key, []), label=key)
    plt.xlabel("epoch")
    plt.ylabel("score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(metric_path)
    plt.close()
    paths.append(metric_path)

    paths.append(write_confusion_matrix(plots_dir, y_true, y_pred))

    return paths
