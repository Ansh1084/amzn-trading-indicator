"""Research-only retraining cadence and champion/challenger guardrails.

This module intentionally does not schedule jobs or modify production artifacts.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal

import numpy as np
import pandas as pd


Cadence = Literal["daily", "weekly", "monthly"]


def retraining_due(last_training_date: date | None, today: date, cadence: Cadence) -> bool:
    """Return whether a scheduled research retrain is due (calendar cadence)."""
    if cadence not in {"daily", "weekly", "monthly"}:
        raise ValueError(f"unsupported cadence: {cadence}")
    if last_training_date is None:
        return True
    if today <= last_training_date:
        return False
    elapsed = (today - last_training_date).days
    if cadence == "daily":
        return elapsed >= 1
    if cadence == "weekly":
        return elapsed >= 7
    return today.year * 12 + today.month - (last_training_date.year * 12 + last_training_date.month) >= 1


def mature_training_rows(timestamps, target_valid, now: datetime, label_completion_minutes: int = 15) -> np.ndarray:
    """Indices with completed labels; t+10 target candle completes at t+15.

    Timestamps are candle starts and must be timezone-aware. The latest row can
    only train once its t+10 exit candle has itself completed (5m duration).
    """
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    ts = pd.to_datetime(pd.Series(timestamps), utc=True, errors="raise")
    valid = np.asarray(target_valid, dtype=bool)
    if len(ts) != len(valid):
        raise ValueError("timestamps and target_valid must have the same length")
    now_utc = pd.Timestamp(now).tz_convert("UTC")
    mature = ts + pd.Timedelta(minutes=label_completion_minutes) <= now_utc
    return np.flatnonzero(mature.to_numpy() & valid)


def population_stability_index(reference, recent, bins: int = 10) -> float:
    """Compute PSI using quantile bins defined only by the reference sample."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(recent, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) < bins or len(cur) == 0:
        raise ValueError("PSI requires at least `bins` finite reference values and recent data")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.linspace(ref.min() - 1e-9, ref.max() + 1e-9, max(3, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.histogram(ref, bins=edges)[0].astype(float)
    cur_counts = np.histogram(cur, bins=edges)[0].astype(float)
    ref_share = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
    cur_share = np.clip(cur_counts / cur_counts.sum(), 1e-6, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def drift_triggered(psi_values, threshold: float = 0.20) -> bool:
    """Flag drift if any monitored feature exceeds a documented PSI threshold."""
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    values = np.asarray(list(psi_values), dtype=float)
    finite = values[np.isfinite(values)]
    return bool(len(finite) and np.any(finite >= threshold))


def promotion_review(champion: dict, challenger: dict, *, minimum_samples: int = 250,
                     minimum_macro_f1_gain: float = 0.01,
                     maximum_drawdown_degradation: float = 0.01) -> dict:
    """Return a manual-review recommendation; never auto-promote a model."""
    required = ("n", "f1_macro", "balanced_accuracy", "trade_max_drawdown", "fold_f1_std")
    if any(key not in champion or key not in challenger for key in required):
        raise ValueError(f"both metrics must contain {required}")
    checks = {
        "minimum_sample_count": int(challenger["n"]) >= minimum_samples,
        "macro_f1_gain": float(challenger["f1_macro"] - champion["f1_macro"]) >= minimum_macro_f1_gain,
        "balanced_accuracy_not_lower": float(challenger["balanced_accuracy"]) >= float(champion["balanced_accuracy"]),
        "drawdown_not_materially_worse": float(challenger["trade_max_drawdown"]) >=
            float(champion["trade_max_drawdown"]) - maximum_drawdown_degradation,
        "fold_stability_not_materially_worse": float(challenger["fold_f1_std"]) <=
            float(champion["fold_f1_std"]) + 0.02,
    }
    return {"eligible_for_human_review": all(checks.values()), "automatic_promotion": False,
            "checks": checks, "note": "A passing challenger still requires review on a separate untouched period and explicit deployment approval."}
