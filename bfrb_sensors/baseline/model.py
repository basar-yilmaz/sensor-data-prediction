"""XGBoost baseline model construction from the Hydra baseline config."""

from __future__ import annotations

from typing import Any

from xgboost import XGBClassifier


def build_baseline_model(cfg, num_classes: int, seed: int) -> XGBClassifier:
    """Build an ``XGBClassifier`` from ``cfg.baseline`` settings.

    Mirrors :func:`bfrb_sensors.models.factory.build_model`: it dispatches on the
    baseline ``name`` and translates the config into constructor kwargs.
    """
    name = str(cfg.name)
    if name != "xgboost":
        raise ValueError(f"unknown baseline {name!r}")

    xgb = cfg.xgboost
    kwargs: dict[str, Any] = {
        "objective": "multi:softprob",
        "num_class": int(num_classes),
        "eval_metric": "mlogloss",
        "n_estimators": int(xgb.n_estimators),
        "max_depth": int(xgb.max_depth),
        "learning_rate": float(xgb.learning_rate),
        "subsample": float(xgb.subsample),
        "colsample_bytree": float(xgb.colsample_bytree),
        "min_child_weight": float(xgb.min_child_weight),
        "reg_lambda": float(xgb.reg_lambda),
        "reg_alpha": float(xgb.reg_alpha),
        "gamma": float(xgb.gamma),
        "tree_method": str(xgb.tree_method),
        "n_jobs": int(xgb.n_jobs),
        "random_state": int(seed),
    }
    early_stopping_rounds = int(xgb.early_stopping_rounds)
    if early_stopping_rounds > 0:
        kwargs["early_stopping_rounds"] = early_stopping_rounds
    return XGBClassifier(**kwargs)
