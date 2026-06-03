from __future__ import annotations

from pathlib import Path

from bfrb_sensors.training.plots import write_training_plots


def test_write_training_plots_creates_split_metric_pngs(tmp_path: Path):
    history = {
        "train_loss": [1.0, 0.8],
        "val_loss": [1.1, 0.9],
        "val_accuracy": [0.2, 0.3],
        "val_macro_f1_18": [0.1, 0.2],
        "val_macro_f1_collapsed": [0.3, 0.4],
        "val_binary_precision": [0.5, 0.6],
        "val_binary_recall": [0.35, 0.45],
        "val_binary_f1": [0.4, 0.5],
        "val_hierarchical_f1": [0.25, 0.35],
    }
    paths = write_training_plots(tmp_path, history, y_true=[0, 1, 1], y_pred=[0, 0, 1])
    assert {path.name for path in paths} == {
        "loss.png",
        "validation_metrics.png",
        "validation_multiclass_metrics.png",
        "validation_binary_metrics.png",
        "confusion_matrix.png",
    }
    assert all(path.exists() and path.suffix == ".png" for path in paths)
