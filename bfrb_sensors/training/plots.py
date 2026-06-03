"""Training plot generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

_MULTICLASS_METRICS = (
    ("val_accuracy", "Accuracy"),
    ("val_macro_f1_18", "Macro F1 (18 classes)"),
    ("val_macro_f1_collapsed", "Macro F1 (collapsed)"),
    ("val_hierarchical_f1", "Hierarchical F1"),
)
_BINARY_METRICS = (
    ("val_binary_precision", "Binary precision"),
    ("val_binary_recall", "Binary recall"),
    ("val_binary_f1", "Binary F1"),
)


def write_confusion_matrix(
    plots_dir: Path,
    y_true: list[int],
    y_pred: list[int],
    filename: str = "confusion_matrix.png",
) -> Path:
    """Row-normalized confusion matrix; shared by every model and split."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / filename
    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    fig_width = max(9.0, 0.55 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, fig_width))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("Normalized confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.tick_params(axis="both", labelsize=8)

    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            color = "white" if value >= 0.5 else "black"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=7,
            )

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_metric_plot(
    plots_dir: Path,
    filename: str,
    title: str,
    ylabel: str,
    history: dict[str, list[float]],
    metrics: tuple[tuple[str, str], ...],
) -> Path | None:
    available = [(key, label) for key, label in metrics if history.get(key)]
    if not available:
        return None

    path = plots_dir / filename
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label in available:
        ax.plot(history[key], label=label, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_metric_group(
    ax: plt.Axes,
    history: dict[str, list[float]],
    metrics: tuple[tuple[str, str], ...],
    title: str,
) -> None:
    for key, label in metrics:
        if history.get(key):
            ax.plot(history[key], label=label, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("score")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")


def _write_validation_metrics_overview(
    plots_dir: Path,
    history: dict[str, list[float]],
) -> Path:
    path = plots_dir / "validation_metrics.png"
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    _plot_metric_group(
        axes[0],
        history,
        _MULTICLASS_METRICS,
        "Validation multiclass and hierarchical metrics",
    )
    _plot_metric_group(axes[1], history, _BINARY_METRICS, "Validation binary metrics")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
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
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.get("train_loss", []), label="train loss", linewidth=2)
    ax.plot(history.get("val_loss", []), label="validation loss", linewidth=2)
    if history.get("val_log_loss"):
        ax.plot(history["val_log_loss"], label="validation log loss", linewidth=2)
    ax.set_title("Loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(loss_path, dpi=160)
    plt.close(fig)
    paths.append(loss_path)

    paths.append(_write_validation_metrics_overview(plots_dir, history))

    for metric_path in (
        _write_metric_plot(
            plots_dir,
            "validation_multiclass_metrics.png",
            "Validation multiclass and hierarchical metrics",
            "score",
            history,
            _MULTICLASS_METRICS,
        ),
        _write_metric_plot(
            plots_dir,
            "validation_binary_metrics.png",
            "Validation binary metrics",
            "score",
            history,
            _BINARY_METRICS,
        ),
    ):
        if metric_path is not None:
            paths.append(metric_path)

    paths.append(write_confusion_matrix(plots_dir, y_true, y_pred))

    return paths
