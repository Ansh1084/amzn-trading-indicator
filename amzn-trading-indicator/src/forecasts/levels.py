"""Causal swing levels and historical support-hold frequency estimation."""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def _confirmed_swings(history: pd.DataFrame, left: int, right: int) -> tuple[list[float], list[float]]:
    lows, highs = [], []
    low = history["low"].to_numpy(float)
    high = history["high"].to_numpy(float)
    for i in range(left, len(history) - right):
        # The right-side candles are all before the forecast anchor, so each pivot is known.
        if low[i] == np.min(low[i-left:i+right+1]):
            lows.append(float(low[i]))
        if high[i] == np.max(high[i-left:i+right+1]):
            highs.append(float(high[i]))
    return lows, highs


def find_levels(history: pd.DataFrame, reference: float, atr: float,
                lookback: int = 20, left: int = 2, right: int = 2,
                max_atr_distance: float = 3.0) -> dict[str, Any]:
    if history.empty or not np.isfinite(reference) or reference <= 0:
        return {"support": None, "resistance": None, "method": "confirmed_swing_2x2_atr_filtered"}
    days = pd.to_datetime(history["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.date
    selected_days = sorted(days.unique())[-lookback:]
    hist = history.loc[days.isin(selected_days)].reset_index(drop=True)
    lows, highs = _confirmed_swings(hist, left, right)
    distance = max(float(atr) if np.isfinite(atr) else 0.0, reference * 0.001)
    support_candidates = [x for x in lows if x < reference and reference - x <= max_atr_distance * distance]
    resistance_candidates = [x for x in highs if x > reference and x - reference <= max_atr_distance * distance]
    support = max(support_candidates) if support_candidates else None
    resistance = min(resistance_candidates) if resistance_candidates else None
    return {
        "support": support,
        "support_method": "nearest_confirmed_prior_swing_low_within_atr_limit",
        "support_distance": (reference - support) if support is not None else None,
        "resistance": resistance,
        "resistance_method": "nearest_confirmed_prior_swing_high_within_atr_limit",
        "resistance_distance": (resistance - reference) if resistance is not None else None,
        "reference_price": reference,
        "atr": float(atr) if np.isfinite(atr) else None,
    }


def support_hold_probability(samples: list[dict[str, Any]], support_distance: float | None,
                             atr: float | None, minimum_samples: int = 30) -> dict[str, Any]:
    if support_distance is None or atr is None or not np.isfinite(atr) or atr <= 0:
        return {"probability": None, "sample_count": 0, "status": "unavailable_no_support_or_atr"}
    ratio = support_distance / atr
    comparable = [s for s in samples if s.get("break") is not None and s.get("distance_atr") is not None
                  and abs(s["distance_atr"] - ratio) <= 0.5]
    if len(comparable) < minimum_samples:
        return {"probability": None, "sample_count": len(comparable), "status": "insufficient_calibration_samples"}
    # Jeffreys smoothing prevents empirical 0%/100% certainty from finite samples.
    holds = sum(not bool(s["break"]) for s in comparable)
    probability = (holds + 0.5) / (len(comparable) + 1.0)
    return {"probability": round(probability, 4), "sample_count": len(comparable),
            "status": "historical_conditional_frequency", "distance_atr": round(ratio, 4)}


def downside_scenario(history: pd.DataFrame, support: float | None, reference: float,
                      atr: float | None, levels: dict[str, Any]) -> dict[str, Any]:
    lower_lows = [x for x in _confirmed_swings(history, 2, 2)[0] if support is not None and x < support]
    next_support = max(lower_lows) if lower_lows else None
    if next_support is not None:
        lower, upper = next_support, support
        method = "next_confirmed_swing_low_to_current_support"
    else:
        width = max(float(atr or reference * 0.01), reference * 0.005)
        lower, upper = max(0.01, (support or reference) - width), (support or reference)
        method = "one_atr_below_support_fallback"
    return {"trigger": "5m_close_below_support", "next_support": next_support,
            "estimated_range": {"lower": float(lower), "upper": float(upper)},
            "method": method, "estimate_only": True}
