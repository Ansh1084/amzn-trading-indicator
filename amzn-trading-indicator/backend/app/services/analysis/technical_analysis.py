"""Causal technical indicators for completed 5-minute AMZN candles."""
from __future__ import annotations

import numpy as np
import pandas as pd
from backend.app.services.analysis.market_structure import confirmed_pivots

from backend.app.services.analysis.config import TIMEZONE, SWING_RIGHT


def _rsi_by_session(frame: pd.DataFrame, period: int) -> pd.Series:
    close = frame.close.astype(float)
    open_ = frame.open.astype(float)
    local_day = frame.timestamp.dt.tz_convert(TIMEZONE).dt.date
    delta = close.groupby(local_day, sort=False).diff()
    first = close.groupby(local_day, sort=False).cumcount().eq(0)
    delta.loc[first] = (close - open_).loc[first]
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.groupby(local_day, sort=False).transform(
        lambda x: x.ewm(alpha=1 / period, min_periods=period, adjust=False).mean())
    avg_loss = loss.groupby(local_day, sort=False).transform(
        lambda x: x.ewm(alpha=1 / period, min_periods=period, adjust=False).mean())
    den = avg_gain + avg_loss
    result = 100 * avg_gain / den.replace(0, np.nan)
    result = result.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    result = result.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    result = result.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    return result


def calculate_indicators(candles: pd.DataFrame) -> pd.DataFrame:
    """Return indicators aligned to each input row; no centered/future windows."""
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(candles.columns)
    if missing:
        raise ValueError(f"missing candle columns: {sorted(missing)}")
    if candles.empty:
        raise ValueError("at least one candle is required")
    frame = candles.copy().reset_index(drop=True)
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="raise")
    if frame.timestamp.duplicated().any() or not frame.timestamp.is_monotonic_increasing:
        raise ValueError("candles must have unique chronological timestamps")
    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if frame[["open", "high", "low", "close", "volume"]].isna().any().any():
        raise ValueError("OHLCV values must be finite and present")
    if not np.isfinite(frame[["open", "high", "low", "close", "volume"]].to_numpy(float)).all():
        raise ValueError("OHLCV values must be finite")
    if (frame.volume < 0).any() or (frame.low <= 0).any():
        raise ValueError("volume must be nonnegative and prices positive")
    invalid = (frame.high < frame.low) | (frame.high < frame[["open", "close"]].max(axis=1)) | (
        frame.low > frame[["open", "close"]].min(axis=1))
    if invalid.any():
        raise ValueError("invalid OHLC relationships")

    close, high, low = frame.close, frame.high, frame.low
    for span in (9, 20, 50):
        frame[f"ema_{span}"] = close.ewm(span=span, adjust=False, min_periods=span).mean()
    for window in (20, 50):
        frame[f"sma_{window}"] = close.rolling(window, min_periods=window).mean()
    frame["ema_9_slope_5"] = frame.ema_9.pct_change(5, fill_method=None)
    frame["ema_20_slope_5"] = frame.ema_20.pct_change(5, fill_method=None)
    frame["ema_50_slope_10"] = frame.ema_50.pct_change(10, fill_method=None)
    frame["ema_cross_9_20"] = np.sign(frame.ema_9 - frame.ema_20)
    frame["ema_cross_20_50"] = np.sign(frame.ema_20 - frame.ema_50)
    frame["ema_cross_9_20_event"] = np.select(
        [(frame.ema_cross_9_20.shift(1) <= 0) & (frame.ema_cross_9_20 > 0),
         (frame.ema_cross_9_20.shift(1) >= 0) & (frame.ema_cross_9_20 < 0)],
        ["BULLISH_CROSS", "BEARISH_CROSS"], default="NONE")
    frame["ema_cross_20_50_event"] = np.select(
        [(frame.ema_cross_20_50.shift(1) <= 0) & (frame.ema_cross_20_50 > 0),
         (frame.ema_cross_20_50.shift(1) >= 0) & (frame.ema_cross_20_50 < 0)],
        ["BULLISH_CROSS", "BEARISH_CROSS"], default="NONE")
    frame["price_distance_ema20_pct"] = (close / frame.ema_20 - 1) * 100
    frame["price_distance_ema50_pct"] = (close / frame.ema_50 - 1) * 100
    aligned_up = (frame.ema_9 > frame.ema_20) & (frame.ema_20 > frame.ema_50)
    aligned_down = (frame.ema_9 < frame.ema_20) & (frame.ema_20 < frame.ema_50)
    slope_up = (frame.ema_20_slope_5 > 0) & (frame.ema_50_slope_10 > 0)
    slope_down = (frame.ema_20_slope_5 < 0) & (frame.ema_50_slope_10 < 0)
    frame["trend_alignment"] = np.select([aligned_up & slope_up, aligned_up, aligned_down & slope_down, aligned_down],
                                          [2, 1, -2, -1], default=0)
    frame["trend_classification"] = np.select(
        [aligned_up & slope_up, aligned_up, aligned_down & slope_down, aligned_down],
        ["STRONG_BULLISH", "BULLISH", "STRONG_BEARISH", "BEARISH"], default="NEUTRAL")

    local_day = frame.timestamp.dt.tz_convert(TIMEZONE).dt.date
    frame["rsi_7"] = _rsi_by_session(frame, 7)
    frame["rsi_14"] = _rsi_by_session(frame, 14)
    prev_close = close.groupby(local_day, sort=False).shift()
    tr = pd.concat([(high - low).rename("hl"), (high - prev_close).abs().rename("hc"),
                    (low - prev_close).abs().rename("lc")], axis=1).max(axis=1)
    first = close.groupby(local_day, sort=False).cumcount().eq(0)
    tr.loc[first] = (high - low).loc[first]
    frame["atr_14"] = tr.groupby(local_day, sort=False).transform(
        lambda x: x.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean())
    frame["atr_pct"] = frame.atr_14 / close * 100
    frame["macd"] = close.ewm(span=12, adjust=False, min_periods=26).mean() - close.ewm(
        span=26, adjust=False, min_periods=26).mean()
    frame["macd_signal"] = frame.macd.ewm(span=9, adjust=False, min_periods=9).mean()
    frame["macd_histogram"] = frame.macd - frame.macd_signal
    frame["macd_histogram_change"] = frame.macd_histogram.diff()
    frame["momentum_acceleration"] = frame.macd_histogram_change
    frame["roc_5_pct"] = close.pct_change(5, fill_method=None) * 100
    frame["roc_20_pct"] = close.pct_change(20, fill_method=None) * 100
    frame["momentum_direction"] = np.select(
        [(frame.rsi_7 > 50) & (frame.macd_histogram > 0),
         (frame.rsi_7 < 50) & (frame.macd_histogram < 0)], ["POSITIVE", "NEGATIVE"], default="MIXED")
    frame["overbought"] = frame.rsi_14 >= 70
    frame["oversold"] = frame.rsi_14 <= 30

    mid = close.rolling(20, min_periods=20).mean()
    sd = close.rolling(20, min_periods=20).std(ddof=0)
    frame["bollinger_middle"] = mid
    frame["bollinger_upper"] = mid + 2 * sd
    frame["bollinger_lower"] = mid - 2 * sd
    frame["bollinger_width_pct"] = (frame.bollinger_upper - frame.bollinger_lower) / mid * 100
    frame["bollinger_position"] = (close - frame.bollinger_lower) / (
        frame.bollinger_upper - frame.bollinger_lower).replace(0, np.nan)
    frame["rolling_volatility_20"] = close.pct_change(fill_method=None).rolling(20, min_periods=15).std() * np.sqrt(78)
    frame["volatility_percentile"] = frame.atr_pct.rolling(252, min_periods=60).rank(pct=True)
    frame["volatility_classification"] = np.select(
        [frame.volatility_percentile >= .95, frame.volatility_percentile >= .80,
         frame.volatility_percentile <= .20], ["EXTREME", "HIGH", "LOW"], default="NORMAL")

    frame["volume_sma_5"] = frame.volume.groupby(local_day, sort=False).transform(
        lambda x: x.rolling(5, min_periods=5).mean())
    frame["volume_sma_20"] = frame.volume.groupby(local_day, sort=False).transform(
        lambda x: x.rolling(20, min_periods=15).mean())
    # Audit metadata for the existing same-session rolling ratio. This does not
    # change the denominator or any decision/model calculation.
    frame["volume_observation_count_20"] = frame.volume.groupby(local_day, sort=False).transform(
        lambda x: x.rolling(20, min_periods=1).count())
    frame["relative_volume"] = frame.volume / frame.volume_sma_20.replace(0, np.nan)
    frame["volume_acceleration"] = frame.volume_sma_5 / frame.volume_sma_20.replace(0, np.nan)
    session_delta=close.groupby(local_day,sort=False).diff()
    first_session=frame.groupby(local_day,sort=False).cumcount().eq(0)
    session_delta.loc[first_session]=(close-frame.open).loc[first_session]
    frame["price_volume_relationship"] = np.sign(session_delta) * frame.relative_volume
    frame["unusually_high_volume"] = frame.relative_volume >= 1.5
    frame["volume_confirmation"] = (frame.relative_volume >= 1.2) & (
        np.sign(close - frame.open) == np.sign(frame.roc_5_pct))
    # Derived after structure detection; NaN means not evaluated rather than false.
    frame["weak_volume_breakout"] = np.nan
    frame["buying_pressure"] = np.where(close > frame.open, frame.relative_volume, 0.0)
    frame["selling_pressure"] = np.where(close < frame.open, frame.relative_volume, 0.0)

    # Divergence is emitted on the bar that confirms the latest 2x2 pivot.
    # It compares consecutive confirmed same-type pivots; no marker appears before
    # the right-side confirmation candles have completed.
    frame["bullish_divergence"] = False
    frame["bearish_divergence"] = False
    pivots=confirmed_pivots(frame)
    for kind, flag, oscillator, price_condition, oscillator_condition in (
        ("LOW","bullish_divergence","rsi_7",lambda a,b:b["price"]<a["price"],lambda a,b:b["oscillator"]>a["oscillator"]),
        ("HIGH","bearish_divergence","rsi_7",lambda a,b:b["price"]>a["price"],lambda a,b:b["oscillator"]<a["oscillator"]),
    ):
        typed=sorted((p for p in pivots if p["type"]==kind),key=lambda p:p["index"])
        for prior,current in zip(typed,typed[1:]):
            first_osc=float(frame[oscillator].iloc[prior["index"]]) if pd.notna(frame[oscillator].iloc[prior["index"]]) else np.nan
            next_osc=float(frame[oscillator].iloc[current["index"]]) if pd.notna(frame[oscillator].iloc[current["index"]]) else np.nan
            if np.isfinite(first_osc) and np.isfinite(next_osc) and price_condition(prior,current) and oscillator_condition({"oscillator":first_osc},{"oscillator":next_osc}):
                confirmation=current["index"]+SWING_RIGHT
                if confirmation<len(frame): frame.loc[confirmation,flag]=True
    return frame


def analyze_latest(indicators: pd.DataFrame) -> dict:
    if indicators.empty:
        raise ValueError("no indicator rows")
    last = indicators.iloc[-1]
    required = ["ema_9", "ema_20", "ema_50", "rsi_7", "rsi_14", "macd", "macd_signal", "atr_14"]
    missing = [key for key in required if pd.isna(last.get(key))]
    if missing:
        return {"status": "INSUFFICIENT_WARMUP", "missing_indicators": missing,
                "minimum_history_bars": 60, "current_candle_timestamp": last.timestamp.isoformat()}
    result = {}
    for key in indicators.columns:
        value = last[key]
        if isinstance(value, (np.bool_, bool)):
            value = bool(value)
        elif isinstance(value, (np.integer, int)):
            value = int(value)
        elif isinstance(value, (np.floating, float)):
            value = float(value) if np.isfinite(value) else None
        if pd.isna(value) if not isinstance(value, str) else False:
            value = None
        result[key] = value
    result["status"] = "READY"
    return result
