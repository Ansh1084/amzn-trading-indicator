"""Ten-minute forward target generation for AMZN 5-minute OHLCV candles.

This module computes leakage-safe forward price returns and binary directional targets:
  - Exact timestamp matching: Matches each candle at t with the candle at exactly
    t + horizon_minutes. Missing candles never cause an incorrect future match.
  - Session and day boundaries: Targets never cross trading dates or regular trading
    session boundaries (using America/New_York local exchange time).
  - Configurable neutral threshold: Forward returns falling within [-threshold, +threshold]
    are classified as neutral (target = NaN). Default neutral threshold is 0.0 bps.
  - Strict leakage prevention: Target columns (future_close, forward_return, target)
    are diagnostic and supervision labels ONLY. They must NEVER be included in feature
    sets for model training or live inference.

Target Definition:
  forward_return = ((future_close - current_close) / current_close) * 100.0
  target = 1.0 when forward_return > positive_threshold
  target = 0.0 when forward_return < -negative_threshold
  target = NaN when within neutral zone or future candle is unavailable
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from src.data.preprocessing import DEFAULT_US_EARLY_CLOSES

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_HORIZON_MINUTES: int = 10
DEFAULT_NEUTRAL_THRESHOLD_BPS: float = 0.0
DEFAULT_TIMEZONE: str = "America/New_York"
RTH_START: str = "09:30"
RTH_END: str = "16:00"

REQUIRED_INPUT_COLUMNS: list[str] = ["timestamp", "close"]

DEFAULT_LEAKAGE_HORIZONS: list[int] = [5, 10, 15, 20, 30, 60]


def get_target_columns(horizon_minutes: int = 10) -> list[str]:
    """Return explicit target column names for a given prediction horizon."""
    if isinstance(horizon_minutes, bool) or not isinstance(horizon_minutes, int) or horizon_minutes <= 0:
        raise ValueError(
            f"horizon_minutes must be a positive integer (not boolean), got: {horizon_minutes!r}"
        )
    return [
        f"future_close_{horizon_minutes}m",
        f"forward_return_{horizon_minutes}m",
        f"target_{horizon_minutes}m",
        f"target_valid_{horizon_minutes}m",
    ]


def get_leakage_target_columns(horizons: Sequence[int] | int | None = None) -> list[str]:
    """Return all target supervision columns to exclude from model features across horizons.

    Parameters
    ----------
    horizons : int, sequence of int, or None
        Horizon(s) in minutes. If None, uses standard horizons [5, 10, 15, 20, 30, 60].

    Returns
    -------
    list[str]
        Deduplicated list of target column names.
    """
    if horizons is None:
        h_list = DEFAULT_LEAKAGE_HORIZONS
    elif isinstance(horizons, int):
        h_list = [horizons]
    else:
        h_list = list(horizons)

    cols: list[str] = []
    for h in h_list:
        for c in get_target_columns(h):
            if c not in cols:
                cols.append(c)
    return cols


def is_target_column(column_name: str) -> bool:
    """Return True if column_name is a forward-looking target supervision column."""
    prefixes = ("future_close_", "forward_return_", "target_", "target_valid_")
    return any(
        column_name.startswith(p) and column_name.endswith("m")
        for p in prefixes
    )


# Catalog of leakage columns across standard horizons
LEAKAGE_TARGET_COLUMNS: list[str] = get_leakage_target_columns(DEFAULT_LEAKAGE_HORIZONS)


# ── Internal Validation Helpers ──────────────────────────────────────────────


def _validate_threshold(val: object, name: str) -> float:
    """Validate that a threshold is numeric, finite, non-negative, and not boolean."""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"{name} must be numeric, got: {val!r}")
    f_val = float(val)
    if np.isnan(f_val):
        raise ValueError(f"{name} cannot be NaN.")
    if not np.isfinite(f_val):
        raise ValueError(f"{name} cannot be infinite.")
    if f_val < 0.0:
        raise ValueError(f"{name} must be non-negative, got: {f_val}.")
    return f_val


def _parse_and_validate_timestamps(
    series: pd.Series,
    naive_tz_policy: str = "reject",
) -> pd.Series:
    """Parse timestamp series into UTC-aware datetime, rejecting naive timestamps by default.

    Parameters
    ----------
    series : pd.Series
        Raw timestamp series (strings or datetime).
    naive_tz_policy : str
        Policy for timezone-naive timestamps:
          - 'reject' (default): raise ValueError if any naive timestamp is encountered.
          - 'assume_utc': assume naive timestamps are UTC, with a logged warning.
          - <tz string>: localize to this timezone and convert to UTC.

    Returns
    -------
    pd.Series
        UTC-aware datetime64[ns, UTC] Series.
    """
    if series.empty:
        raise ValueError("Timestamp column is empty.")

    # 1. If series is already a datetime dtype
    if pd.api.types.is_datetime64_any_dtype(series):
        if series.dt.tz is None:
            if naive_tz_policy == "reject":
                raise ValueError(
                    f"Found {len(series)} timezone-naive timestamps. Naive timestamps "
                    "are rejected by default. Set naive_tz_policy='assume_utc' or specify a source timezone."
                )
            elif naive_tz_policy == "assume_utc":
                logger.warning("Localizing %d naive timestamps to UTC.", len(series))
                ts_utc = series.dt.tz_localize("UTC")
            else:
                logger.warning(
                    "Localizing %d naive timestamps to %s and converting to UTC.",
                    len(series),
                    naive_tz_policy,
                )
                try:
                    ts_utc = series.dt.tz_localize(naive_tz_policy).dt.tz_convert("UTC")
                except Exception as exc:
                    raise ValueError(
                        f"Failed to localize naive timestamps to timezone '{naive_tz_policy}': {exc}"
                    ) from exc
        else:
            ts_utc = series.dt.tz_convert("UTC")
    else:
        # 2. String or object series
        raw_strings = series.dropna().astype(str).str.strip()
        has_tz = raw_strings.str.contains(r"[+\-]\d{2}:?\d{2}|Z$", regex=True)
        n_naive = int((~has_tz).sum())

        if n_naive > 0:
            if naive_tz_policy == "reject":
                raise ValueError(
                    f"Found {n_naive} timezone-naive timestamps. Naive timestamps are "
                    "rejected by default. Set naive_tz_policy='assume_utc' or specify a source timezone."
                )
            elif naive_tz_policy == "assume_utc":
                logger.warning(
                    "%d timestamps had no timezone info and were assumed UTC.", n_naive
                )
                ts_utc = pd.to_datetime(series, utc=True, format="mixed", errors="coerce")
            else:
                logger.warning(
                    "%d timestamps had no timezone info and were localized to %s.",
                    n_naive,
                    naive_tz_policy,
                )
                res = pd.Series(index=series.index, dtype="datetime64[ns, UTC]")
                if has_tz.any():
                    res[has_tz] = pd.to_datetime(
                        series[has_tz], utc=True, format="mixed", errors="coerce"
                    )
                if (~has_tz).any():
                    naive_dt = pd.to_datetime(
                        series[~has_tz], format="mixed", errors="coerce"
                    )
                    try:
                        localized = naive_dt.dt.tz_localize(naive_tz_policy)
                    except Exception as exc:
                        raise ValueError(
                            f"Failed to localize naive timestamps to timezone '{naive_tz_policy}': {exc}"
                        ) from exc
                    res[~has_tz] = localized.dt.tz_convert("UTC")
                ts_utc = res
        else:
            ts_utc = pd.to_datetime(series, utc=True, format="mixed", errors="coerce")

    # Check for unparseable or null/NaT timestamps
    if ts_utc.isna().any():
        n_invalid = int(ts_utc.isna().sum())
        raise ValueError(
            f"Timestamp column contains {n_invalid} invalid, unparseable, or null/NaT value(s)."
        )

    # Check duplicates
    if ts_utc.duplicated().any():
        n_dups = int(ts_utc.duplicated().sum())
        raise ValueError(f"Input contains {n_dups} duplicate timestamp(s).")

    # Check chronologically increasing
    if not ts_utc.is_monotonic_increasing:
        raise ValueError(
            "Timestamps are not chronologically increasing. Silent sorting is forbidden."
        )

    return ts_utc


# ── Core Target Generation Function ──────────────────────────────────────────


def generate_targets(
    df: pd.DataFrame,
    *,
    horizon_minutes: int = DEFAULT_HORIZON_MINUTES,
    neutral_threshold_bps: float = DEFAULT_NEUTRAL_THRESHOLD_BPS,
    positive_threshold_bps: float | None = None,
    negative_threshold_bps: float | None = None,
    naive_tz_policy: str = "reject",
    tz: str = DEFAULT_TIMEZONE,
    rth_start: str = RTH_START,
    rth_end: str = RTH_END,
    early_close_schedule: dict[str, str] | None = None,
    enforce_session_bounds: bool = True,
    include_validity_col: bool = True,
) -> pd.DataFrame:
    """Generate forward returns and binary classification targets on 5-minute candles.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least 'timestamp' and 'close'.
    horizon_minutes : int
        Prediction horizon in minutes (default 10). Must be a positive integer (not boolean).
    neutral_threshold_bps : float
        Symmetric neutral zone half-width in basis points (1 bps = 0.01% = 0.0001).
        Default 0.0 bps (0.0%). Must be numeric, finite, and non-negative.
    positive_threshold_bps : float or None
        Optional explicit positive threshold in bps. Overrides neutral_threshold_bps
        for the upper boundary if provided. Must be non-negative and finite.
    negative_threshold_bps : float or None
        Optional explicit negative threshold magnitude in bps. Overrides neutral_threshold_bps
        for the lower boundary if provided. Must be non-negative and finite.
    naive_tz_policy : str
        Policy for timezone-naive timestamps: 'reject' (default), 'assume_utc', or a tz name.
    tz : str
        Exchange timezone for session and date boundaries (default 'America/New_York').
    rth_start : str
        Session open time string 'HH:MM' (default '09:30').
    rth_end : str
        Session close time string 'HH:MM' (default '16:00').
    early_close_schedule : dict[str, str] or None
        Mapping of 'YYYY-MM-DD' dates to early close time 'HH:MM'. If None, uses
        DEFAULT_US_EARLY_CLOSES.
    enforce_session_bounds : bool
        If True (default), target timestamps must strictly fall within regular
        trading hours (start <= time < session_end). End-of-session candles that
        lack a future candle within the same session receive NaN targets.
    include_validity_col : bool
        If True (default), appends boolean column 'target_valid_{h}m' (True if
        target is not NaN, False otherwise).

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with preserved row order and index, plus:
          - future_close_{h}m : Exact close price at t + horizon_minutes (or NaN)
          - forward_return_{h}m : Percentage return ((future - current) / current) * 100
          - target_{h}m : 1.0 (UP), 0.0 (DOWN), or NaN (neutral / unavailable)
          - target_valid_{h}m : Boolean validity indicator (if include_validity_col=True)

    Raises
    ------
    ValueError
        If required columns are missing, DataFrame is empty, close prices are
        invalid, horizon is invalid, thresholds are negative/non-finite, or
        timestamps are invalid, naive, duplicate, or unsorted.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    missing_cols = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame is missing required column(s): {missing_cols}")

    # Validate horizon_minutes
    if isinstance(horizon_minutes, bool) or not isinstance(horizon_minutes, int) or horizon_minutes <= 0:
        raise ValueError(
            f"horizon_minutes must be a positive integer (not boolean), got: {horizon_minutes!r}"
        )

    # ── 1. Validate Thresholds (bps -> percentage return units) ──────────────
    val_neutral = _validate_threshold(neutral_threshold_bps, "neutral_threshold_bps")

    if positive_threshold_bps is not None:
        pos_thresh_pct = _validate_threshold(positive_threshold_bps, "positive_threshold_bps") / 100.0
    else:
        pos_thresh_pct = val_neutral / 100.0

    if negative_threshold_bps is not None:
        neg_thresh_pct = _validate_threshold(negative_threshold_bps, "negative_threshold_bps") / 100.0
    else:
        neg_thresh_pct = val_neutral / 100.0

    # ── 2. Strict Timestamp Validation ───────────────────────────────────────
    ts_utc = _parse_and_validate_timestamps(df["timestamp"], naive_tz_policy=naive_tz_policy)

    # ── 3. Strict Close Price Validation ─────────────────────────────────────
    try:
        close_series = pd.to_numeric(df["close"], errors="coerce")
    except Exception as err:
        raise ValueError(f"Invalid close column: {err}") from err

    if close_series.isna().any() or not np.isfinite(close_series.to_numpy(dtype=float)).all():
        raise ValueError("Close column contains NaN, null, or non-finite values.")

    if (close_series <= 0).any():
        raise ValueError("Close column contains zero or negative prices.")

    current_close = close_series.to_numpy(dtype=float)

    # ── 4. Session & Trading Date Determination (America/New_York) ───────────
    out = df.copy()

    ts_local = ts_utc.dt.tz_convert(tz)
    date_local = ts_local.dt.date

    # Exact target timestamp
    target_ts = ts_utc + pd.Timedelta(minutes=horizon_minutes)
    target_local = target_ts.dt.tz_convert(tz)
    target_date = target_local.dt.date

    # Condition A: Future timestamp must be on the SAME local exchange date
    same_date_mask = (target_date == date_local).to_numpy()

    # Condition B: Enforce regular trading session boundaries
    if enforce_session_bounds:
        schedule = dict(DEFAULT_US_EARLY_CLOSES)
        if early_close_schedule:
            schedule.update(early_close_schedule)

        date_str = date_local.astype(str)
        start_time_val = pd.Timestamp(rth_start).time()
        end_times_map = {
            d: pd.Timestamp(schedule.get(d, rth_end)).time()
            for d in date_str.unique()
        }
        end_time_series = date_str.map(end_times_map)

        cur_t = ts_local.dt.time
        tgt_t = target_local.dt.time

        cur_in_sess = ((cur_t >= start_time_val) & (cur_t < end_time_series)).to_numpy()
        tgt_in_sess = ((tgt_t >= start_time_val) & (tgt_t < end_time_series)).to_numpy()
        valid_session_mask = same_date_mask & cur_in_sess & tgt_in_sess
    else:
        valid_session_mask = same_date_mask

    # ── 5. Exact Timestamp Matching (Causal & Non-Blind) ─────────────────────
    close_by_ts = pd.Series(current_close, index=ts_utc)
    future_close_raw = target_ts.map(close_by_ts).to_numpy(dtype=float)

    # Disallow matches crossing session boundaries or dates
    future_close = np.where(valid_session_mask, future_close_raw, np.nan)

    # ── 6. Forward Return Calculation ────────────────────────────────────────
    forward_return = np.where(
        np.isfinite(future_close) & (current_close > 0),
        ((future_close - current_close) / current_close) * 100.0,
        np.nan,
    )

    # ── 7. Binary Target Labeling with Neutral Zone ──────────────────────────
    target = np.full(len(df), np.nan, dtype=float)
    valid_ret = np.isfinite(forward_return)

    up_mask = valid_ret & (forward_return > pos_thresh_pct)
    down_mask = valid_ret & (forward_return < -neg_thresh_pct)

    target[up_mask] = 1.0
    target[down_mask] = 0.0

    # ── 8. Assign Output Columns ─────────────────────────────────────────────
    col_fc = f"future_close_{horizon_minutes}m"
    col_fr = f"forward_return_{horizon_minutes}m"
    col_tgt = f"target_{horizon_minutes}m"

    out[col_fc] = future_close
    out[col_fr] = forward_return
    out[col_tgt] = target

    if include_validity_col:
        col_valid = f"target_valid_{horizon_minutes}m"
        out[col_valid] = ~np.isnan(target)

    return out
