"""Classification and signal-quality metrics.

This module provides evaluation utilities for the AMZN intraday trading model,
computing standard classification metrics and formatting results for comparison.

Key design principles:
  - Consistent metric computation across all models.
  - ROC-AUC only computed when probability predictions are available.
  - Confusion matrix and classification report for every model.
  - Structured results for easy comparison and serialization.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    *,
    model_name: str = "model",
    split_name: str = "test",
) -> dict[str, Any]:
    """Compute standard classification metrics for a binary classifier.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels (0 or 1).
    y_pred : np.ndarray
        Predicted labels (0 or 1).
    y_prob : np.ndarray or None
        Predicted probabilities for the positive class (class 1). Used for ROC-AUC.
        If None, ROC-AUC is not computed.
    model_name : str
        Name of the model for reporting.
    split_name : str
        Name of the data split (e.g. ``"val"``, ``"test"``).

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
          - model_name, split_name
          - accuracy, precision, recall, f1_score
          - roc_auc (or None if y_prob not provided)
          - confusion_matrix (as nested list)
          - classification_report (as string)
          - n_samples, n_positive, n_negative
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # ROC-AUC requires probability predictions and both classes present
    roc = None
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        unique_classes = np.unique(y_true)
        if len(unique_classes) == 2:
            try:
                roc = float(roc_auc_score(y_true, y_prob))
            except ValueError:
                logger.warning(
                    "ROC-AUC computation failed for %s on %s split.",
                    model_name, split_name,
                )

    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, zero_division=0)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())

    result = {
        "model_name": model_name,
        "split_name": split_name,
        "accuracy": round(acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1_score": round(f1, 6),
        "roc_auc": round(roc, 6) if roc is not None else None,
        "confusion_matrix": cm,
        "classification_report": report,
        "n_samples": len(y_true),
        "n_positive": n_pos,
        "n_negative": n_neg,
    }

    logger.info(
        "%s [%s] — Acc: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f | AUC: %s",
        model_name, split_name, acc, prec, rec, f1,
        f"{roc:.4f}" if roc is not None else "N/A",
    )

    return result


def build_comparison_table(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a comparison DataFrame from a list of metric dicts.

    Parameters
    ----------
    results : list[dict]
        List of dicts returned by ``compute_classification_metrics``.

    Returns
    -------
    pd.DataFrame
        Comparison table with one row per model-split combination.
    """
    rows = []
    for r in results:
        rows.append({
            "model": r["model_name"],
            "split": r["split_name"],
            "accuracy": r["accuracy"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1_score": r["f1_score"],
            "roc_auc": r["roc_auc"],
            "n_samples": r["n_samples"],
            "n_positive": r["n_positive"],
            "n_negative": r["n_negative"],
        })
    return pd.DataFrame(rows)
