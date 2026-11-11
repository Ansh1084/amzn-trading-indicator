"""Leakage-conscious ML and gross-return evaluation helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, log_loss, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score)


def classification_metrics(y_true, y_pred, p_up, train_y=None) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    p_up = np.asarray(p_up, dtype=float)
    both_classes = len(np.unique(y_true)) == 2
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    precision_by_class = precision_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    recall_by_class = recall_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    f1_by_class = f1_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    metrics = {
        "n": int(len(y_true)), "up_actual": int((y_true == 1).sum()), "down_actual": int((y_true == 0).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 and both_classes else 0.0,
        "up_precision": float(precision_by_class[1]), "up_recall": float(recall_by_class[1]), "up_f1": float(f1_by_class[1]),
        "down_precision": float(precision_by_class[0]), "down_recall": float(recall_by_class[0]), "down_f1": float(f1_by_class[0]),
        "confusion_matrix_labels_down_up": cm.tolist(),
        "roc_auc": float(roc_auc_score(y_true, p_up)) if both_classes else None,
        "pr_auc_up": float(average_precision_score(y_true, p_up)) if both_classes else None,
        "log_loss": float(log_loss(y_true, np.column_stack([1-p_up, p_up]), labels=[0, 1])) if both_classes else None,
        "brier_score_raw": float(brier_score_loss(y_true, p_up)) if both_classes else None,
    }
    if train_y is not None:
        majority = int(np.mean(np.asarray(train_y, dtype=int)) >= 0.5)
        metrics["majority_baseline_accuracy"] = float(np.mean(y_true == majority))
        metrics["majority_class_in_train"] = "UP" if majority else "DOWN"
    return metrics


def trading_metrics(frame: pd.DataFrame, y_pred) -> dict:
    """Gross long-UP/short-DOWN returns on non-overlapping 10-minute starts."""
    pred = np.asarray(y_pred, dtype=int)
    local = pd.to_datetime(frame.timestamp, utc=True).dt.tz_convert("America/New_York")
    minute_of_day = local.dt.hour * 60 + local.dt.minute
    minutes_since_open = minute_of_day - (9 * 60 + 30)
    mask = (minutes_since_open >= 0) & (minutes_since_open % 10 == 0)
    selected = frame.loc[mask].copy()
    selected_pred = pred[mask.to_numpy()]
    raw = selected["forward_return_10m"].to_numpy(dtype=float) / 100.0
    signed = np.where(selected_pred == 1, raw, -raw)
    signed = signed[np.isfinite(signed)]
    if not len(signed):
        return {"n_nonoverlap": 0, "cumulative_gross_return": None, "max_drawdown": None,
                "profit_factor": None, "win_rate": None, "average_strategy_return": None}
    wealth = np.cumprod(1.0 + signed)
    running_peak = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    drawdown = wealth / running_peak - 1.0
    gains, losses = signed[signed > 0], signed[signed < 0]
    return {
        "n_nonoverlap": int(len(signed)),
        "cumulative_gross_return": float(wealth[-1] - 1.0),
        "max_drawdown": float(drawdown.min()),
        "profit_factor": float(gains.sum() / abs(losses.sum())) if len(losses) else None,
        "win_rate": float(np.mean(signed > 0)),
        "average_strategy_return": float(np.mean(signed)),
        "average_actual_return": float(np.mean(raw)),
        "average_absolute_return": float(np.mean(np.abs(raw))),
        "correct_count": int((selected_pred == (selected["target_10m"].to_numpy(dtype=int))).sum()),
    }


def combine_metrics(parts: list[dict]) -> dict:
    """Aggregate fold metrics; averages are unweighted across folds except n."""
    if not parts:
        return {}
    keys = [k for k, v in parts[0].items() if isinstance(v, (int, float)) and v is not None]
    combined = {f"mean_{key}": float(np.mean([p[key] for p in parts if p.get(key) is not None]))
                for key in keys if any(p.get(key) is not None for p in parts)}
    combined["fold_count"] = len(parts)
    combined["total_samples"] = int(sum(p.get("n", 0) for p in parts))
    return combined
