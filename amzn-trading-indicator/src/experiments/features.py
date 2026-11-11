"""Causal experimental features; production feature pipelines are not modified."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.engineering import FEATURE_COLUMNS as CURRENT_FEATURES
from src.features.engineering import generate_features as generate_current_features
from src.forecasts.calendar import ET, session_close


ADVANCED_GROUPS: dict[str, list[str]] = {
    "advanced_trend": [
        "sma_ratio_50", "ema_ratio_50", "ema_slope_5", "ema_slope_10", "ema_slope_20",
        "ema_slope_50", "sma_slope_20", "sma_slope_50", "ema_spread_5_20",
        "ema_spread_10_50", "trend_strength_atr",
    ],
    "advanced_momentum": [
        "rsi_7", "macd_hist_change_1", "roc_20", "momentum_acceleration",
        "consecutive_bullish", "consecutive_bearish",
    ],
    "advanced_volatility": [
        "atr_pct", "return_std_20", "return_std_60", "volatility_ratio_20_60",
        "volatility_high_regime", "volatility_expansion",
    ],
    "advanced_volume": [
        "relative_volume_60", "volume_acceleration", "price_volume_interaction",
        "volume_spike_2sigma",
    ],
    "advanced_candle": [
        "body_pct", "upper_wick_pct", "lower_wick_pct", "range_pct",
        "close_position", "bullish_candle", "bearish_candle", "gap_from_previous_close_pct",
    ],
    "advanced_structure": [
        "distance_prior_high_12", "distance_prior_low_12", "distance_prior_high_36",
        "distance_prior_low_36", "breakout_above_prior_high_12", "breakdown_below_prior_low_12",
        "range_compression_12_atr",
    ],
    "advanced_time": [
        "minutes_since_open", "minutes_to_close", "opening_period", "closing_period",
        "session_phase_sin", "session_phase_cos",
    ],
    "advanced_regime": ["trend_regime", "sideways_regime"],
}

ADVANCED_FEATURES = [column for group in ADVANCED_GROUPS.values() for column in group]


def generate_experimental_features(candles: pd.DataFrame) -> pd.DataFrame:
    """Return existing current features plus a causal, versioned research block.

    Row t may use OHLCV through the completed candle at t. Every rolling
    structure level is shifted one candle, and no centered/backward fill is used.
    """
    if candles.empty:
        raise ValueError("candles cannot be empty")
    required = {"timestamp", "open", "high", "low", "close", "volume", "vwap"}
    missing = required - set(candles.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    out = candles.copy().reset_index(drop=True)
    out["timestamp"] = pd.to_datetime(out.timestamp, utc=True, errors="raise")
    if out.timestamp.duplicated().any() or not out.timestamp.is_monotonic_increasing:
        raise ValueError("timestamps must be unique and chronological")
    for column in ("open", "high", "low", "close", "volume", "vwap"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if out[["open", "high", "low", "close", "volume"]].isna().any().any():
        raise ValueError("OHLCV values must be present and numeric")

    # Current 36-feature production-window pipeline remains unmodified.
    base = generate_current_features(out, reset_daily=False)
    for column in CURRENT_FEATURES:
        out[column] = base[column].to_numpy()

    local = out.timestamp.dt.tz_convert(ET)
    day = local.dt.date
    close, high, low, open_, volume = (out[x].astype(float) for x in ("close", "high", "low", "open", "volume"))
    groups = lambda s: s.groupby(day, sort=False, group_keys=False)
    pct1 = close.groupby(day, sort=False).pct_change(fill_method=None)

    # Trend levels and slopes are normalized by price and use current/past bars.
    for span in (20, 50):
        sma = close.rolling(span, min_periods=span).mean()
        ema = close.ewm(span=span, adjust=False, min_periods=span).mean()
        out[f"sma_ratio_{span}"] = (close - sma) / sma
        out[f"ema_ratio_{span}"] = (close - ema) / ema
        out[f"sma_slope_{span}"] = sma.pct_change(5, fill_method=None)
        out[f"ema_slope_{span}"] = ema.pct_change(5, fill_method=None)
    for span in (5, 10):
        ema = close.ewm(span=span, adjust=False).mean()
        out[f"ema_slope_{span}"] = ema.pct_change(3, fill_method=None)
    out["ema_spread_5_20"] = (close.ewm(span=5, adjust=False).mean() - close.ewm(span=20, adjust=False).mean()) / close
    out["ema_spread_10_50"] = (close.ewm(span=10, adjust=False).mean() - close.ewm(span=50, adjust=False).mean()) / close
    out["trend_strength_atr"] = (close.ewm(span=5, adjust=False).mean() - close.ewm(span=50, adjust=False).mean()).abs() / out.atr_14

    # Session reset momentum; opening return uses open rather than overnight close gap.
    delta = close.groupby(day, sort=False).diff()
    first = groups(close).cumcount().eq(0)
    delta = delta.where(~first, close - open_)
    gain, loss = delta.clip(lower=0), (-delta).clip(lower=0)
    avg_gain = gain.groupby(day, sort=False).transform(lambda s: s.ewm(alpha=1/7, adjust=False, min_periods=7).mean())
    avg_loss = loss.groupby(day, sort=False).transform(lambda s: s.ewm(alpha=1/7, adjust=False, min_periods=7).mean())
    out["rsi_7"] = 100 - 100 / (1 + avg_gain / avg_loss.replace(0, np.nan))
    out.loc[(avg_loss == 0) & (avg_gain > 0), "rsi_7"] = 100.0
    out.loc[(avg_loss == 0) & (avg_gain == 0), "rsi_7"] = 50.0
    out["macd_hist_change_1"] = out.macd_hist.diff()
    out["roc_20"] = close.groupby(day, sort=False).transform(lambda s: s.pct_change(20, fill_method=None) * 100)
    out["momentum_acceleration"] = pct1.groupby(day, sort=False).diff()
    candle_sign = np.sign(close - open_)
    bull_streak = np.zeros(len(out), dtype=float)
    bear_streak = np.zeros(len(out), dtype=float)
    b = r = 0
    prior_day = None
    for i, (d, sign) in enumerate(zip(day, candle_sign)):
        if d != prior_day:
            b = r = 0
            prior_day = d
        b = b + 1 if sign > 0 else 0
        r = r + 1 if sign < 0 else 0
        bull_streak[i], bear_streak[i] = b, r
    out["consecutive_bullish"], out["consecutive_bearish"] = bull_streak, bear_streak

    # Volatility features and rolling causal regime boundaries.
    out["atr_pct"] = out.atr_14 / close
    out["return_std_20"] = pct1.groupby(day, sort=False).transform(lambda s: s.rolling(20, min_periods=10).std())
    out["return_std_60"] = pct1.groupby(day, sort=False).transform(lambda s: s.rolling(60, min_periods=20).std())
    out["volatility_ratio_20_60"] = out.return_std_20 / out.return_std_60.replace(0, np.nan)
    vol_median = out.atr_pct.groupby(day, sort=False).transform(lambda s: s.rolling(60, min_periods=20).median())
    out["volatility_high_regime"] = (out.atr_pct > vol_median).astype(float).where(vol_median.notna())
    out["volatility_expansion"] = out.return_std_20.groupby(day, sort=False).diff()

    # Volume dynamics are normalized against strictly trailing within-session means.
    vol_mean60 = volume.groupby(day, sort=False).transform(lambda s: s.rolling(60, min_periods=20).mean())
    vol_std60 = volume.groupby(day, sort=False).transform(lambda s: s.rolling(60, min_periods=20).std())
    out["relative_volume_60"] = volume / vol_mean60.replace(0, np.nan)
    out["volume_acceleration"] = out.volume_ratio_5.groupby(day, sort=False).diff()
    out["price_volume_interaction"] = pct1 * out.volume_ratio_5
    out["volume_spike_2sigma"] = ((volume - vol_mean60) > 2 * vol_std60).astype(float).where(vol_mean60.notna())

    # Candle geometry is scaled by price/range. Gap uses the previous observed close.
    candle_range = (high - low).replace(0, np.nan)
    out["body_pct"] = (close - open_) / close
    out["upper_wick_pct"] = (high - np.maximum(open_, close)) / close
    out["lower_wick_pct"] = (np.minimum(open_, close) - low) / close
    out["range_pct"] = (high - low) / close
    out["close_position"] = ((close - low) / candle_range).fillna(0.5)
    out["bullish_candle"], out["bearish_candle"] = (close > open_).astype(float), (close < open_).astype(float)
    out["gap_from_previous_close_pct"] = (open_ / close.shift(1) - 1).where(~first, np.nan)

    # Prior swing references exclude the current bar, so the current bar cannot
    # help define the boundary it is being tested against.
    for window in (12, 36):
        prior_high = high.groupby(day, sort=False).transform(lambda s: s.rolling(window, min_periods=window).max().shift(1))
        prior_low = low.groupby(day, sort=False).transform(lambda s: s.rolling(window, min_periods=window).min().shift(1))
        out[f"distance_prior_high_{window}"] = (close - prior_high) / close
        out[f"distance_prior_low_{window}"] = (close - prior_low) / close
        if window == 12:
            out["breakout_above_prior_high_12"] = (close > prior_high).astype(float).where(prior_high.notna())
            out["breakdown_below_prior_low_12"] = (close < prior_low).astype(float).where(prior_low.notna())
    out["range_compression_12_atr"] = (high.groupby(day, sort=False).transform(lambda s: s.rolling(12, min_periods=12).max().shift(1)) -
                                         low.groupby(day, sort=False).transform(lambda s: s.rolling(12, min_periods=12).min().shift(1))) / out.atr_14

    # Exchange-local time features including early-close aware session phase.
    mins = local.dt.hour * 60 + local.dt.minute
    open_min = 9 * 60 + 30
    close_min = pd.Series([session_close(d).hour * 60 + session_close(d).minute for d in day], index=out.index)
    elapsed = mins - open_min
    duration = close_min - open_min
    out["minutes_since_open"] = elapsed.astype(float)
    out["minutes_to_close"] = (close_min - mins).astype(float)
    out["opening_period"] = (elapsed < 30).astype(float)
    out["closing_period"] = (close_min - mins <= 30).astype(float)
    phase = elapsed / duration.clip(lower=1)
    out["session_phase_sin"] = np.sin(2 * np.pi * phase)
    out["session_phase_cos"] = np.cos(2 * np.pi * phase)

    # A simple causal trend/range regime based on current trend strength.
    out["trend_regime"] = (out.trend_strength_atr >= 1.0).astype(float).where(out.trend_strength_atr.notna())
    out["sideways_regime"] = (out.trend_strength_atr < 1.0).astype(float).where(out.trend_strength_atr.notna())
    return out


def get_advanced_feature_columns() -> list[str]:
    return list(ADVANCED_FEATURES)


def get_feature_groups(include_base: bool = True) -> dict[str, list[str]]:
    groups = {name: list(columns) for name, columns in ADVANCED_GROUPS.items()}
    if include_base:
        groups["base_36"] = list(CURRENT_FEATURES)
    return groups
