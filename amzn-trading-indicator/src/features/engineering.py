"""Shared causal feature engineering for training and inference.

This module computes backward-looking technical, price-action, volume,
and time-of-day features on 5-minute OHLCV candles for AMZN.

Key design principles:
  - Strict causality: All features at candle t use only data available at or before t.
    No future information, centered rolling windows, or look-ahead biases.
  - No target leakage: Target labels or future close prices are never computed or
    referenced in this module.
  - Zero-division & infinity safety: Division operations are explicitly guarded to
    prevent inf, -inf, or ZeroDivisionError.
  - NaN transparency: Initial rolling warm-up periods produce NaN values naturally.
    NaNs are never bfilled or zero-filled. A missing-value inspection helper is provided.
  - Configurable session handling: Supports both continuous rolling indicators
    across overnight sessions and session-reset indicators (re-initialized at market open).
  - Symmetry between training and live inference: Historical batch calculations and
    live rolling-window inference execute identical code with identical column order.

Session Handling Trade-Offs (Continuous vs. Session-Reset):
  1. Continuous mode (`reset_daily=False`):
     - Calculates indicators across session boundaries without resetting at market open.
     - Pros: Avoids discarding the first ~20-35 candles of each trading day during warm-up;
       maintains unbroken multi-candle momentum/trend context.
     - Cons: Treats overnight gap intervals as adjacent 5-minute steps; moving averages
       smooth across overnight price jumps.
  2. Session-reset mode (`reset_daily=True`):
     - Resets all rolling, lag, and exponential indicators at each day's market open.
     - Pros: Intraday indicators reflect only today's price discovery and volume dynamics,
       completely isolated from overnight price jumps.
     - Cons: The first 14-35 candles of each trading session have NaN values due to warm-up,
       reducing usable intraday observations if dropped.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Required Input Columns ───────────────────────────────────────────────────

REQUIRED_INPUT_COLUMNS: list[str] = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
]

# ── Explicit Model Feature Columns ───────────────────────────────────────────

FEATURE_COLUMNS: list[str] = [
    # A. Price & return features
    "return_1",
    "return_2",
    "return_3",
    "log_return_1",
    "price_change",
    "candle_body",
    "candle_range",
    "upper_wick",
    "lower_wick",
    "body_to_range",
    # B. Trend features
    "sma_5",
    "sma_10",
    "sma_20",
    "ema_5",
    "ema_10",
    "ema_20",
    "macd",
    "macd_signal",
    "macd_hist",
    # C. Momentum features
    "rsi_14",
    "roc_5",
    "roc_10",
    # D. Volatility features
    "atr_14",
    "bb_upper",
    "bb_middle",
    "bb_lower",
    "bb_width",
    "bb_position",
    # E. Volume & VWAP features
    "volume_change",
    "volume_sma_5",
    "volume_sma_20",
    "volume_ratio_5",
    "volume_ratio_20",
    "volume_ratio",
    "vwap_distance",
    # F. Time features
    "hour",
    "minute",
    "day_of_week",
    "time_of_day_sin",
    "time_of_day_cos",
]


def get_feature_columns() -> list[str]:
    """Return an immutable copy of the explicit model feature column names."""
    return list(FEATURE_COLUMNS)


def get_missing_feature_counts(df: pd.DataFrame) -> pd.Series:
    """Return count of NaN values for each engineered feature column present in df.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing engineered features.

    Returns
    -------
    pd.Series
        Missing value counts indexed by feature column name.
    """
    present_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not present_cols:
        return pd.Series(dtype=int)
    return df[present_cols].isna().sum()


# ── Internal Calculation Helpers ─────────────────────────────────────────────


def _safe_divide(
    num: np.ndarray | pd.Series,
    den: np.ndarray | pd.Series,
    *,
    fill_value: float = np.nan,
    min_den: float = 1e-12,
) -> np.ndarray:
    """Safe masked element-wise division avoiding zero-division warnings.

    Only evaluates division at positions where denominator is finite and > min_den.
    All other positions are filled with `fill_value`.
    Prevents ZeroDivisionError, RuntimeWarning, and infinite values.
    """
    n = np.asarray(num, dtype=float)
    d = np.asarray(den, dtype=float)
    out = np.full(n.shape, fill_value, dtype=float)
    valid_mask = np.isfinite(n) & np.isfinite(d) & (d > min_den)
    np.divide(n, d, out=out, where=valid_mask)
    return out


def _apply_grouped_or_series(
    series: pd.Series,
    group_col: pd.Series | None,
    func,
) -> pd.Series:
    """Apply a transformation either continuously or grouped by trading_date."""
    if group_col is None:
        return func(series)
    return series.groupby(group_col, group_keys=False).apply(func)


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute mathematically correct Wilder's RSI on a price series.

    Formula and Initialization:
      1. Price changes: Delta[t] = close[t] - close[t-1]
      2. Gains and Losses: Gain = max(Delta, 0), Loss = max(-Delta, 0)
      3. First 14 valid price changes (from 15 prices) initialize average gain and loss
         using their simple arithmetic mean:
           AvgGain[14] = sum(Gain[1..14]) / 14
           AvgLoss[14] = sum(Loss[1..14]) / 14
      4. For subsequent candles t > 14, apply Wilder's recursive smoothing:
           AvgGain[t] = (AvgGain[t-1] * 13 + Gain[t]) / 14
           AvgLoss[t] = (AvgLoss[t-1] * 13 + Loss[t]) / 14
      5. RSI calculation:
           - Total <= 1e-12 -> RSI = 50.0 (completely flat prices)
           - AvgLoss <= 1e-14 and AvgGain > 0 -> RSI = 100.0 (strictly rising)
           - AvgGain <= 1e-14 and AvgLoss > 0 -> RSI = 0.0 (strictly falling)
           - Otherwise: RSI = 100.0 * AvgGain / (AvgGain + AvgLoss)
      6. Warm-up periods (< 14 valid price changes) remain NaN.
    """
    n = len(series)
    out = np.full(n, np.nan, dtype=float)
    if n <= period:
        return pd.Series(out, index=series.index, dtype=float)

    prices = series.to_numpy(dtype=float)
    deltas = np.diff(prices)  # length n - 1

    valid_deltas = np.isfinite(deltas)
    valid_indices = np.where(valid_deltas)[0]

    if len(valid_indices) < period:
        return pd.Series(out, index=series.index, dtype=float)

    first_14_idx = valid_indices[:period]
    first_14_deltas = deltas[first_14_idx]

    gains_14 = np.maximum(first_14_deltas, 0.0)
    losses_14 = np.maximum(-first_14_deltas, 0.0)

    avg_gain = float(np.mean(gains_14))
    avg_loss = float(np.mean(losses_14))

    def _calc_rsi_val(ag: float, al: float) -> float:
        total = ag + al
        if total <= 1e-12:
            return 50.0
        if al <= 1e-14:
            return 100.0
        if ag <= 1e-14:
            return 0.0
        return 100.0 * ag / total

    init_candle_idx = first_14_idx[-1] + 1
    out[init_candle_idx] = _calc_rsi_val(avg_gain, avg_loss)

    p_minus_1 = period - 1
    for d_idx in range(first_14_idx[-1] + 1, len(deltas)):
        candle_idx = d_idx + 1
        d = deltas[d_idx]
        if not np.isfinite(d):
            out[candle_idx] = np.nan
            continue
        g = max(d, 0.0)
        l = max(-d, 0.0)
        avg_gain = (avg_gain * p_minus_1 + g) / period
        avg_loss = (avg_loss * p_minus_1 + l) / period
        out[candle_idx] = _calc_rsi_val(avg_gain, avg_loss)

    return pd.Series(out, index=series.index, dtype=float)


def _compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Compute mathematically correct Wilder's Average True Range (ATR).

    First-Candle True Range Convention:
      On the first candle of a series (or session when reset_daily=True), no prior
      closing price exists. Under standard technical analysis convention
      (Wilder 1978), the True Range for candle 0 is defined as:
          TR_0 = high_0 - low_0
      For all subsequent candles (t >= 1):
          TR_t = max(high_t - low_t, abs(high_t - close_{t-1}), abs(low_t - close_{t-1}))

    Initialization and Recursive Smoothing:
      1. First 14 True Range values (candles 0..13) initialize ATR using their simple arithmetic mean:
           ATR_13 = sum(TR[0..13]) / 14
      2. For all subsequent candles t >= 14, apply Wilder's recursive smoothing:
           ATR_t = (ATR_{t-1} * 13 + TR_t) / 14
      3. Warm-up candles (t < 13) remain NaN.
      4. Calculation is strictly causal: only past and present prices are referenced.
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=float)
    if n < period:
        return pd.Series(out, index=close.index, dtype=float)

    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    c = close.to_numpy(dtype=float)

    tr = np.empty(n, dtype=float)
    # First candle convention: no previous close exists
    tr[0] = h[0] - l[0]

    h_l = h[1:] - l[1:]
    h_cp = np.abs(h[1:] - c[:-1])
    l_cp = np.abs(l[1:] - c[:-1])
    tr[1:] = np.maximum(h_l, np.maximum(h_cp, l_cp))

    valid_tr_mask = np.isfinite(tr)
    valid_tr_idx = np.where(valid_tr_mask)[0]

    if len(valid_tr_idx) < period:
        return pd.Series(out, index=close.index, dtype=float)

    first_14_idx = valid_tr_idx[:period]
    atr_val = float(np.mean(tr[first_14_idx]))

    init_candle_idx = first_14_idx[-1]
    out[init_candle_idx] = atr_val

    p_minus_1 = period - 1
    for idx in range(init_candle_idx + 1, n):
        tr_val = tr[idx]
        if not np.isfinite(tr_val):
            out[idx] = np.nan
            continue
        atr_val = (atr_val * p_minus_1 + tr_val) / period
        out[idx] = atr_val

    return pd.Series(out, index=close.index, dtype=float)


# ── Core Feature Engineering Function ────────────────────────────────────────


def generate_features(
    df: pd.DataFrame,
    *,
    reset_daily: bool = False,
    tz: str = "America/New_York",
) -> pd.DataFrame:
    """Generate all engineered features on 5-minute AMZN OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least: timestamp, open, high, low, close, volume, vwap.
    reset_daily : bool
        If False (default), indicators and rolling windows run continuously across sessions.
        If True, indicators and returns reset at the beginning of each trading day
        (using America/New_York local date).
    tz : str
        Timezone for time-of-day features and session grouping (default "America/New_York").

    Returns
    -------
    pd.DataFrame
        A new DataFrame with all original columns preserved plus all engineered features.
        The input DataFrame is not modified in-place.
        Original index and row count are preserved.

    Raises
    ------
    ValueError
        If required input columns are missing, if df is empty, or if timestamps
        are invalid, unparseable, duplicate, or not chronologically increasing.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    missing_cols = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame is missing required column(s): {missing_cols}")

    # ── Strict Timestamp Validation ──────────────────────────────────────────
    # Reject invalid, unparseable, duplicate, or unsorted timestamps.
    # Silent sorting is strictly forbidden to preserve input order and index integrity.
    try:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    except Exception as err:
        raise ValueError(f"Failed to parse timestamp column: {err}") from err

    if ts.isna().any():
        n_invalid = int(ts.isna().sum())
        raise ValueError(
            f"Timestamp column contains {n_invalid} invalid, unparseable, or null/NaT value(s)."
        )

    if ts.duplicated().any():
        n_dups = int(ts.duplicated().sum())
        raise ValueError(f"Input contains {n_dups} duplicate timestamp(s).")

    if not ts.is_monotonic_increasing:
        raise ValueError(
            "Timestamps are not chronologically increasing. Silent sorting is forbidden."
        )

    # Copy to ensure the input DataFrame is never modified in-place
    out = df.copy()

    # Timezone conversion for time-of-day features and session grouping
    ts_local = ts.dt.tz_convert(tz)
    group_col = ts_local.dt.date if reset_daily else None

    open_p = out["open"].astype(float)
    high_p = out["high"].astype(float)
    low_p = out["low"].astype(float)
    close_p = out["close"].astype(float)
    volume_p = out["volume"].astype(float)
    vwap_p = out["vwap"].astype(float)

    # ── A. Price and Return Features ─────────────────────────────────────────

    # Shifted closes (session-isolated if reset_daily=True)
    def _calc_returns(s: pd.Series) -> pd.DataFrame:
        c1 = s.shift(1).to_numpy(dtype=float)
        c2 = s.shift(2).to_numpy(dtype=float)
        c3 = s.shift(3).to_numpy(dtype=float)
        s_arr = s.to_numpy(dtype=float)

        r1 = _safe_divide(s_arr - c1, c1, fill_value=np.nan, min_den=1e-12)
        r2 = _safe_divide(s_arr - c2, c2, fill_value=np.nan, min_den=1e-12)
        r3 = _safe_divide(s_arr - c3, c3, fill_value=np.nan, min_den=1e-12)

        # Log return with safe masked evaluation
        log_r1 = np.full(len(s_arr), np.nan, dtype=float)
        mask_log = np.isfinite(s_arr) & (s_arr > 0) & np.isfinite(c1) & (c1 > 0)
        log_s = np.zeros_like(s_arr)
        log_c1 = np.zeros_like(c1)
        np.log(s_arr, out=log_s, where=mask_log)
        np.log(c1, out=log_c1, where=mask_log)
        log_r1[mask_log] = log_s[mask_log] - log_c1[mask_log]

        p_chg = np.where(np.isfinite(c1) & np.isfinite(s_arr), s_arr - c1, np.nan)

        return pd.DataFrame(
            {
                "return_1": r1,
                "return_2": r2,
                "return_3": r3,
                "log_return_1": log_r1,
                "price_change": p_chg,
            },
            index=s.index,
        )

    if group_col is not None:
        ret_df = close_p.groupby(group_col, group_keys=False).apply(_calc_returns)
    else:
        ret_df = _calc_returns(close_p)

    out["return_1"] = ret_df["return_1"]
    out["return_2"] = ret_df["return_2"]
    out["return_3"] = ret_df["return_3"]
    out["log_return_1"] = ret_df["log_return_1"]
    out["price_change"] = ret_df["price_change"]

    # Intra-candle geometry (independent of session boundary)
    c_body = close_p - open_p
    c_range = high_p - low_p
    out["candle_body"] = c_body
    out["candle_range"] = c_range
    out["upper_wick"] = high_p - np.maximum(open_p, close_p)
    out["lower_wick"] = np.minimum(open_p, close_p) - low_p
    out["body_to_range"] = _safe_divide(
        c_body.to_numpy(dtype=float),
        c_range.to_numpy(dtype=float),
        fill_value=0.0,
        min_den=1e-12,
    )

    # ── B. Trend Features (SMA, EMA, MACD) ───────────────────────────────────

    for span in [5, 10, 20]:
        out[f"sma_{span}"] = _apply_grouped_or_series(
            close_p, group_col, lambda s, w=span: s.rolling(w, min_periods=w).mean()
        )
        out[f"ema_{span}"] = _apply_grouped_or_series(
            close_p, group_col, lambda s, sp=span: s.ewm(span=sp, adjust=False).mean()
        )

    # MACD (12, 26, 9)
    ema_12 = _apply_grouped_or_series(
        close_p, group_col, lambda s: s.ewm(span=12, adjust=False).mean()
    )
    ema_26 = _apply_grouped_or_series(
        close_p, group_col, lambda s: s.ewm(span=26, adjust=False).mean()
    )
    macd = ema_12 - ema_26
    macd_signal = _apply_grouped_or_series(
        macd, group_col, lambda s: s.ewm(span=9, adjust=False).mean()
    )
    out["macd"] = macd
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd - macd_signal

    # ── C. Momentum Features (RSI 14, ROC 5, ROC 10) ─────────────────────────

    out["rsi_14"] = _apply_grouped_or_series(
        close_p, group_col, lambda s: _compute_rsi(s, period=14)
    )

    for p in [5, 10]:

        def _calc_roc(s: pd.Series, lag=p) -> pd.Series:
            c_lag = s.shift(lag).to_numpy(dtype=float)
            s_arr = s.to_numpy(dtype=float)
            roc_vals = _safe_divide(s_arr - c_lag, c_lag, fill_value=np.nan, min_den=1e-12)
            roc_vals = np.where(np.isnan(roc_vals), np.nan, roc_vals * 100.0)
            return pd.Series(roc_vals, index=s.index, dtype=float)

        out[f"roc_{p}"] = _apply_grouped_or_series(close_p, group_col, _calc_roc)

    # ── D. Volatility Features (ATR 14, Bollinger Bands) ─────────────────────

    if group_col is not None:
        def _calc_grp_atr(grp: pd.DataFrame) -> pd.Series:
            return _compute_atr(grp["high"], grp["low"], grp["close"], period=14)

        out["atr_14"] = out.groupby(group_col, group_keys=False).apply(_calc_grp_atr)
    else:
        out["atr_14"] = _compute_atr(high_p, low_p, close_p, period=14)

    # Bollinger Bands (20-period, 2 std)
    bb_mid = _apply_grouped_or_series(
        close_p, group_col, lambda s: s.rolling(20, min_periods=20).mean()
    )
    bb_std = _apply_grouped_or_series(
        close_p, group_col, lambda s: s.rolling(20, min_periods=20).std(ddof=0)
    )
    bb_up = bb_mid + 2.0 * bb_std
    bb_low = bb_mid - 2.0 * bb_std
    bb_diff = bb_up - bb_low

    out["bb_upper"] = bb_up
    out["bb_middle"] = bb_mid
    out["bb_lower"] = bb_low
    out["bb_width"] = _safe_divide(
        bb_diff.to_numpy(dtype=float),
        bb_mid.to_numpy(dtype=float),
        fill_value=np.nan,
        min_den=1e-12,
    )

    # Bollinger Band position (%B)
    bb_pos = np.full(len(close_p), np.nan, dtype=float)
    diff_arr = bb_diff.to_numpy(dtype=float)
    close_arr = close_p.to_numpy(dtype=float)
    low_arr = bb_low.to_numpy(dtype=float)
    mask_calc = np.isfinite(diff_arr) & (diff_arr > 1e-12)
    np.divide(close_arr - low_arr, diff_arr, out=bb_pos, where=mask_calc)
    mask_flat = np.isfinite(diff_arr) & (diff_arr <= 1e-12)
    bb_pos[mask_flat] = 0.5
    out["bb_position"] = bb_pos

    # ── E. Volume and VWAP Features ──────────────────────────────────────────

    def _calc_vol_change(s: pd.Series) -> pd.Series:
        v_prev = s.shift(1).to_numpy(dtype=float)
        s_arr = s.to_numpy(dtype=float)
        out_vc = np.full(len(s_arr), np.nan, dtype=float)
        mask_pos = np.isfinite(v_prev) & (v_prev > 0) & np.isfinite(s_arr)
        np.divide(s_arr - v_prev, v_prev, out=out_vc, where=mask_pos)
        zero_to_zero = (
            np.isfinite(v_prev) & (v_prev == 0) & np.isfinite(s_arr) & (s_arr == 0)
        )
        out_vc[zero_to_zero] = 0.0
        return pd.Series(out_vc, index=s.index, dtype=float)

    out["volume_change"] = _apply_grouped_or_series(volume_p, group_col, _calc_vol_change)

    out["volume_sma_5"] = _apply_grouped_or_series(
        volume_p, group_col, lambda s: s.rolling(5, min_periods=5).mean()
    )
    out["volume_sma_20"] = _apply_grouped_or_series(
        volume_p, group_col, lambda s: s.rolling(20, min_periods=20).mean()
    )

    v_arr = volume_p.to_numpy(dtype=float)
    v_sma5_arr = out["volume_sma_5"].to_numpy(dtype=float)
    v_sma20_arr = out["volume_sma_20"].to_numpy(dtype=float)

    out["volume_ratio_5"] = _safe_divide(v_arr, v_sma5_arr, fill_value=np.nan, min_den=1e-12)
    mask_v5_zero = np.isfinite(v_sma5_arr) & (v_sma5_arr <= 1e-12)
    out.loc[mask_v5_zero, "volume_ratio_5"] = 0.0

    out["volume_ratio_20"] = _safe_divide(v_arr, v_sma20_arr, fill_value=np.nan, min_den=1e-12)
    mask_v20_zero = np.isfinite(v_sma20_arr) & (v_sma20_arr <= 1e-12)
    out.loc[mask_v20_zero, "volume_ratio_20"] = 0.0

    # volume_ratio default alias to 20-period
    out["volume_ratio"] = out["volume_ratio_20"]

    # VWAP distance: (close - vwap) / vwap
    vwap_arr = vwap_p.to_numpy(dtype=float)
    out["vwap_distance"] = _safe_divide(
        close_arr - vwap_arr,
        vwap_arr,
        fill_value=np.nan,
        min_den=1e-12,
    )
    mask_vwap_zero = np.isfinite(vwap_arr) & (vwap_arr <= 1e-12)
    out.loc[mask_vwap_zero, "vwap_distance"] = 0.0

    # ── F. Time Features (in America/New_York) ────────────────────────────────

    h = ts_local.dt.hour
    m = ts_local.dt.minute
    total_mins = h * 60 + m

    out["hour"] = h
    out["minute"] = m
    out["day_of_week"] = ts_local.dt.day_of_week
    out["time_of_day_sin"] = np.sin(2.0 * np.pi * total_mins / 1440.0)
    out["time_of_day_cos"] = np.cos(2.0 * np.pi * total_mins / 1440.0)

    return out


# Convenience alias for function discovery
compute_features = generate_features
