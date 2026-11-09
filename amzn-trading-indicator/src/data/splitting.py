"""Chronological splitting and temporal purging.

This module provides session-aware chronological splitting for time-series data,
designed to prevent data leakage in the AMZN intraday trading model pipeline.

Key design principles:
  - Chronological order: Training always precedes validation, which precedes test.
  - Session integrity: Complete trading sessions (all candles on a given trading date)
    are kept together in the same split. No random candle-level splitting.
  - Temporal purge/embargo: A configurable number of candles at the boundary between
    splits are dropped to prevent leakage from forward-looking targets that span
    across split boundaries.
  - No future information: All split boundaries are determined by trading dates, not
    by peeking at target values.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_TRAIN_FRACTION: float = 0.70
DEFAULT_VAL_FRACTION: float = 0.15
DEFAULT_TEST_FRACTION: float = 0.15
DEFAULT_PURGE_CANDLES: int = 2  # 2 candles × 5 min = 10 min (matches horizon)
DEFAULT_TIMEZONE: str = "America/New_York"


# ── Session-Aware Chronological Splitting ────────────────────────────────────


def get_trading_dates(
    df: pd.DataFrame,
    *,
    tz: str = DEFAULT_TIMEZONE,
) -> list:
    """Extract sorted unique trading dates from a timestamp column.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a ``timestamp`` column (UTC-aware or timezone-aware string).
    tz : str
        Timezone for local date extraction (default ``America/New_York``).

    Returns
    -------
    list
        Sorted list of unique ``datetime.date`` objects.
    """
    ts = pd.to_datetime(df["timestamp"], utc=True)
    local_dates = ts.dt.tz_convert(tz).dt.date
    unique_dates = sorted(local_dates.unique())
    return unique_dates


def split_by_sessions(
    df: pd.DataFrame,
    *,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    purge_candles: int = DEFAULT_PURGE_CANDLES,
    tz: str = DEFAULT_TIMEZONE,
) -> dict[str, pd.DataFrame]:
    """Split a DataFrame chronologically by complete trading sessions with purge.

    The split operates on trading dates (sessions), not individual candles:
      1. Unique trading dates are extracted and sorted chronologically.
      2. Dates are assigned to train/val/test based on the specified fractions.
      3. A purge buffer of ``purge_candles`` candles is dropped from the END of the
         training set and the END of the validation set to prevent forward-label
         leakage across split boundaries.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a ``timestamp`` column.
    train_fraction : float
        Fraction of sessions for training (default 0.70).
    val_fraction : float
        Fraction of sessions for validation (default 0.15).
    test_fraction : float
        Fraction of sessions for test (default 0.15).
    purge_candles : int
        Number of candles to drop from the end of train and val sets as a
        temporal embargo buffer (default 2, i.e. 10 minutes for 5-min candles).
    tz : str
        Timezone for trading date extraction (default ``America/New_York``).

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: ``"train"``, ``"val"``, ``"test"``, each a copy of the relevant rows.

    Raises
    ------
    ValueError
        If fractions are invalid, DataFrame is empty, or too few sessions exist.
    """
    # ── Validate inputs ──────────────────────────────────────────────────────
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    total = train_fraction + val_fraction + test_fraction
    if not (0.99 <= total <= 1.01):
        raise ValueError(
            f"Split fractions must sum to ~1.0, got {total:.4f} "
            f"(train={train_fraction}, val={val_fraction}, test={test_fraction})."
        )

    if any(f <= 0 for f in [train_fraction, val_fraction, test_fraction]):
        raise ValueError("All split fractions must be positive.")

    if purge_candles < 0:
        raise ValueError(f"purge_candles must be non-negative, got {purge_candles}.")

    # ── Extract trading dates ────────────────────────────────────────────────
    trading_dates = get_trading_dates(df, tz=tz)
    n_dates = len(trading_dates)

    if n_dates < 3:
        raise ValueError(
            f"Need at least 3 trading sessions for a 3-way split, got {n_dates}."
        )

    # ── Assign dates to splits ───────────────────────────────────────────────
    n_train = max(1, int(round(n_dates * train_fraction)))
    n_val = max(1, int(round(n_dates * val_fraction)))
    n_test = n_dates - n_train - n_val

    if n_test < 1:
        # Ensure at least 1 test session
        n_val = max(1, n_val - 1)
        n_test = n_dates - n_train - n_val
        if n_test < 1:
            n_train = max(1, n_train - 1)
            n_test = n_dates - n_train - n_val

    train_dates = set(trading_dates[:n_train])
    val_dates = set(trading_dates[n_train : n_train + n_val])
    test_dates = set(trading_dates[n_train + n_val :])

    # ── Assign rows by their trading date ────────────────────────────────────
    ts = pd.to_datetime(df["timestamp"], utc=True)
    local_dates = ts.dt.tz_convert(tz).dt.date

    train_mask = local_dates.isin(train_dates)
    val_mask = local_dates.isin(val_dates)
    test_mask = local_dates.isin(test_dates)

    df_train = df.loc[train_mask].copy().reset_index(drop=True)
    df_val = df.loc[val_mask].copy().reset_index(drop=True)
    df_test = df.loc[test_mask].copy().reset_index(drop=True)

    # ── Apply purge/embargo ──────────────────────────────────────────────────
    # Drop the last ``purge_candles`` rows from train and val to prevent the
    # forward-looking target from leaking into the next split.
    n_purged_train = 0
    n_purged_val = 0

    if purge_candles > 0 and len(df_train) > purge_candles:
        n_purged_train = purge_candles
        df_train = df_train.iloc[:-purge_candles].reset_index(drop=True)

    if purge_candles > 0 and len(df_val) > purge_candles:
        n_purged_val = purge_candles
        df_val = df_val.iloc[:-purge_candles].reset_index(drop=True)

    # ── Logging ──────────────────────────────────────────────────────────────
    logger.info(
        "Session-aware chronological split: "
        "%d train sessions (%d candles, %d purged), "
        "%d val sessions (%d candles, %d purged), "
        "%d test sessions (%d candles).",
        len(train_dates), len(df_train), n_purged_train,
        len(val_dates), len(df_val), n_purged_val,
        len(test_dates), len(df_test),
    )

    return {
        "train": df_train,
        "val": df_val,
        "test": df_test,
        "metadata": {
            "n_train_sessions": len(train_dates),
            "n_val_sessions": len(val_dates),
            "n_test_sessions": len(test_dates),
            "n_train_candles": len(df_train),
            "n_val_candles": len(df_val),
            "n_test_candles": len(df_test),
            "n_purged_train": n_purged_train,
            "n_purged_val": n_purged_val,
            "train_date_range": (
                min(train_dates).isoformat(),
                max(train_dates).isoformat(),
            ),
            "val_date_range": (
                min(val_dates).isoformat(),
                max(val_dates).isoformat(),
            ),
            "test_date_range": (
                min(test_dates).isoformat(),
                max(test_dates).isoformat(),
            ),
        },
    }
