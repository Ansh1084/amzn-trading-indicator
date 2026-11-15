"""No-lookahead market context derived only from complete regular-session candles."""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.forecasts.calendar import ET, is_trading_day, previous_trading_day, session_close

_CONFIG_PATH = Path(__file__).resolve().parents[4] / "configs" / "market_intelligence.json"
_DEFAULTS = {"rvol_history_sessions": 20, "rvol_minimum_samples": 3, "rvol_low_below": .75,
             "rvol_normal_below": 1.2, "rvol_elevated_below": 1.8}
_CONFIG = {**_DEFAULTS, **(json.loads(_CONFIG_PATH.read_text(encoding="utf-8")) if _CONFIG_PATH.exists() else {})}
if (int(_CONFIG["rvol_minimum_samples"]) < 1 or int(_CONFIG["rvol_history_sessions"]) < int(_CONFIG["rvol_minimum_samples"]) or
    not 0 < float(_CONFIG["rvol_low_below"]) < float(_CONFIG["rvol_normal_below"]) < float(_CONFIG["rvol_elevated_below"])):
    raise ValueError("Relative-volume configuration has invalid sample counts or thresholds")


def _completed_rth(candles: pd.DataFrame, as_of: datetime | None = None) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if required - set(candles.columns):
        raise ValueError(f"Market intelligence requires columns: {sorted(required)}")
    cutoff = pd.Timestamp(as_of or datetime.now(tz=ET))
    if cutoff.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    frame = candles.copy()
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="raise")
    frame = frame.loc[frame.timestamp + pd.Timedelta(minutes=5) <= cutoff.tz_convert("UTC")].copy()
    local = frame.timestamp.dt.tz_convert(ET)
    local_dates = local.dt.date
    session_closes = {day: session_close(day) if is_trading_day(day) else None for day in local_dates.unique()}
    keep = []
    for stamp in local:
        close = session_closes.get(stamp.date())
        keep.append(bool(close and (stamp.hour, stamp.minute) >= (9, 30) and
                         (stamp.minute - 30) % 5 == 0 and stamp.second == 0 and stamp.microsecond == 0 and
                         (stamp + timedelta(minutes=5)).time().replace(tzinfo=None) <= close))
    frame = frame.loc[keep].sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    numeric = frame[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not all(math.isfinite(float(v)) for v in numeric.to_numpy().ravel()):
        raise ValueError("Market intelligence input contains non-finite OHLCV values")
    if ((numeric.volume < 0) | (numeric.low <= 0) | (numeric.high < numeric[["open", "close"]].max(axis=1)) |
        (numeric.low > numeric[["open", "close"]].min(axis=1))).any():
        raise ValueError("Market intelligence input contains invalid OHLCV relationships")
    frame[list(numeric.columns)] = numeric
    return frame


def _aggregate(frame: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    if minutes == 5:
        return frame[["timestamp", "open", "high", "low", "close", "volume"]].copy().reset_index(drop=True)
    local = frame.timestamp.dt.tz_convert(ET)
    offsets = ((local.dt.hour * 60 + local.dt.minute) - (9 * 60 + 30)).astype(int)
    work = frame.copy()
    work["_day"] = local.dt.date
    work["_minute_offset"] = offsets
    work["_bucket"] = offsets // minutes
    expected = minutes // 5
    rows = []
    for (_, _), group in work.groupby(["_day", "_bucket"], sort=True):
        group = group.sort_values("timestamp")
        mins = group["_minute_offset"].tolist()
        first = mins[0] if mins else -1
        if len(group) != expected or mins != list(range(first, first + expected * 5, 5)):
            continue
        rows.append({"timestamp": group.timestamp.iloc[0], "open": float(group.open.iloc[0]),
                     "high": float(group.high.max()), "low": float(group.low.min()),
                     "close": float(group.close.iloc[-1]), "volume": float(group.volume.sum())})
    return pd.DataFrame(rows)


def _trend(frame: pd.DataFrame, label: str) -> dict:
    if len(frame) < 21:
        return {"timeframe": label, "trend": "NEUTRAL", "status": "INSUFFICIENT_DATA", "bars": len(frame)}
    close = frame.close.astype(float)
    fast = close.ewm(span=9, adjust=False).mean()
    slow = close.ewm(span=21, adjust=False).mean()
    slope = slow.iloc[-1] - slow.iloc[-4]
    if close.iloc[-1] > fast.iloc[-1] > slow.iloc[-1] and slope > 0:
        trend = "BULLISH"
    elif close.iloc[-1] < fast.iloc[-1] < slow.iloc[-1] and slope < 0:
        trend = "BEARISH"
    else:
        trend = "NEUTRAL"
    return {"timeframe": label, "trend": trend, "status": "READY", "bars": len(frame),
            "close": float(close.iloc[-1]), "ema_9": float(fast.iloc[-1]), "ema_21": float(slow.iloc[-1]),
            "ema_21_slope_3": float(slope)}


def _orb(frame: pd.DataFrame, reference: float | None) -> dict:
    if frame.empty:
        return {"status": "UNAVAILABLE", "reason": "No completed candles for the active session."}
    local = frame.timestamp.dt.tz_convert(ET)
    day = local.iloc[-1].date()
    today = frame.loc[local.dt.date == day].copy()
    today_local = today.timestamp.dt.tz_convert(ET)
    first = today.loc[today_local.dt.strftime("%H:%M").isin(["09:30", "09:35", "09:40"])]
    if len(first) != 3 or list(first.timestamp.dt.tz_convert(ET).dt.strftime("%H:%M")) != ["09:30", "09:35", "09:40"]:
        return {"status": "INSUFFICIENT_DATA", "reason": "All three completed 5-minute opening-range candles are required."}
    high, low = float(first.high.max()), float(first.low.min())
    current = float(today.close.iloc[-1])
    after_opening = today.loc[today.timestamp >= first.timestamp.iloc[-1] + pd.Timedelta(minutes=5)]
    if after_opening.empty:
        state = "OPENING_RANGE_COMPLETE"
    else:
        recent = after_opening.tail(12)
        previous = recent.iloc[-2] if len(recent) > 1 else None
        latest = recent.iloc[-1]
        if previous is not None and previous.close > high and latest.close <= high:
            state = "FAILED_BULLISH_BREAKOUT"
        elif previous is not None and previous.close < low and latest.close >= low:
            state = "FAILED_BEARISH_BREAKDOWN"
        elif latest.close > high and latest.low <= high:
            state = "BULLISH_BREAKOUT_RETEST"
        elif latest.close < low and latest.high >= low:
            state = "BEARISH_BREAKDOWN_RETEST"
        elif current > high:
            state = "BULLISH_BREAKOUT"
        elif current < low:
            state = "BEARISH_BREAKDOWN"
        else:
            state = "INSIDE_OPENING_RANGE"
    return {"status": "READY", "high": high, "low": low, "midpoint": (high + low) / 2,
            "reference_price": reference, "distance_to_high": reference - high if reference is not None else None,
            "distance_to_low": reference - low if reference is not None else None, "classification": state,
            "confirmation": "completed 5-minute candle close", "opening_range": "09:30–09:45 America/New_York"}


def _rvol(frame: pd.DataFrame, min_samples=None, history_sessions=None) -> dict:
    if frame.empty:
        return {"status": "INSUFFICIENT_DATA", "classification": "INSUFFICIENT_DATA", "sample_count": 0}
    min_samples = int(min_samples if min_samples is not None else _CONFIG["rvol_minimum_samples"])
    history_sessions = int(history_sessions if history_sessions is not None else _CONFIG["rvol_history_sessions"])
    local = frame.timestamp.dt.tz_convert(ET)
    current = frame.iloc[-1]
    current_local = local.iloc[-1]
    slot = current_local.hour * 12 + current_local.minute // 5
    dates = sorted({d for d in local.dt.date if d < current_local.date()})[-history_sessions:]
    sample = []
    for day in dates:
        same = frame.loc[local.dt.date == day]
        same_local = same.timestamp.dt.tz_convert(ET)
        matching = same.loc[(same_local.dt.hour * 12 + same_local.dt.minute // 5) == slot]
        if len(matching):
            sample.append(float(matching.volume.iloc[-1]))
    if len(sample) < min_samples or sum(sample) <= 0:
        return {"status": "INSUFFICIENT_DATA", "classification": "INSUFFICIENT_DATA", "sample_count": len(sample),
                "slot_et": current_local.strftime("%H:%M"), "required_samples": min_samples}
    avg = sum(sample) / len(sample)
    value = float(current.volume) / avg if avg > 0 else None
    low_below = float(_CONFIG["rvol_low_below"])
    normal_below = float(_CONFIG["rvol_normal_below"])
    elevated_below = float(_CONFIG["rvol_elevated_below"])
    classification = "LOW PARTICIPATION" if value < low_below else "NORMAL" if value < normal_below else "ELEVATED" if value < elevated_below else "HIGH PARTICIPATION"
    return {"status": "READY", "metric_name": "same-time-slot historical RVOL",
            "formula": "latest completed 5-minute candle volume / mean volume for the same ET 5-minute slot on prior trading sessions",
            "candle_timestamp": current.timestamp.isoformat(), "rvol": value, "average_same_slot_volume": avg,
            "current_volume": float(current.volume), "sample_count": len(sample), "slot_et": current_local.strftime("%H:%M"),
            "classification": classification, "thresholds": {"low_below": low_below, "normal_below": normal_below,
                                                               "elevated_below": elevated_below}}


def _previous_day_levels(frame: pd.DataFrame, reference: float | None) -> dict:
    if frame.empty:
        return {"status": "UNAVAILABLE", "reason": "No completed regular-session candles."}
    latest_date = frame.timestamp.iloc[-1].tz_convert(ET).date()
    previous = previous_trading_day(latest_date)
    local_dates = frame.timestamp.dt.tz_convert(ET).dt.date
    day = frame.loc[local_dates == previous]
    if day.empty:
        return {"status": "INSUFFICIENT_DATA", "session_date": previous.isoformat(),
                "reason": "Previous scheduled trading session is not present in the available completed candles."}
    high, low, close = float(day.high.max()), float(day.low.min()), float(day.close.iloc[-1])
    return {"status": "READY", "session_date": previous.isoformat(), "high": high, "low": low, "close": close,
            "reference_price": reference,
            "distance": {"high": reference - high if reference is not None else None,
                         "low": reference - low if reference is not None else None,
                         "close": reference - close if reference is not None else None},
            "position": {"high": "ABOVE" if reference > high else "BELOW" if reference < high else "TESTING" if reference is not None else "UNKNOWN",
                         "low": "ABOVE" if reference > low else "BELOW" if reference < low else "TESTING" if reference is not None else "UNKNOWN",
                         "close": "ABOVE" if reference > close else "BELOW" if reference < close else "TESTING" if reference is not None else "UNKNOWN"}}


def analyze_market_intelligence(candles: pd.DataFrame, as_of: datetime | None = None) -> dict:
    """Return trend alignment and explainable session context, without touching decision weights."""
    frame = _completed_rth(candles, as_of)
    reference = float(frame.close.iloc[-1]) if len(frame) else None
    trend5 = _trend(_aggregate(frame, 5), "5M")
    trend15 = _trend(_aggregate(frame, 15), "15M")
    trend60 = _trend(_aggregate(frame, 60), "1H")
    values = [trend60["trend"], trend15["trend"], trend5["trend"]]
    directionals = {v for v in values if v != "NEUTRAL"}
    alignment = "BULLISH" if directionals == {"BULLISH"} else "BEARISH" if directionals == {"BEARISH"} else "MIXED" if len(directionals) > 1 else "NEUTRAL"
    if alignment == "MIXED":
        conflict = [f"{item['timeframe']} {item['trend']}" for item in (trend60, trend15, trend5)]
        explanation = f"Mixed timeframe evidence: {', '.join(conflict)}. Conflicting non-neutral directions are retained; EMA 9/20/50 structure is reported separately."
    elif alignment in {"BULLISH", "BEARISH"}:
        explanation = f"Completed 1-hour, 15-minute and 5-minute trends align {alignment.lower()}. This is context only, not an entry signal."
    else:
        explanation = "Timeframes are neutral or do not establish a common direction."
    orb = _orb(frame, reference)
    rvol = _rvol(frame)
    if orb.get("status") == "READY" and rvol.get("status") == "READY":
        orb["rvol_evidence"] = {"value": rvol["rvol"], "classification": rvol["classification"],
                                "supports_confirmed_breakout": bool(rvol["rvol"] >= float(_CONFIG["rvol_normal_below"])),
                                "interpretation": "Breakout volume context only; never an independent trade signal."}
    return {"as_of": pd.Timestamp(as_of or datetime.now(tz=ET)).tz_convert("UTC").isoformat(),
            "latest_completed_candle": frame.timestamp.iloc[-1].isoformat() if len(frame) else None,
            "trend_alignment": {"overall": alignment, "timeframes": [trend60, trend15, trend5], "explanation": explanation,
                                "conflict_warning": alignment == "MIXED"},
            "opening_range": orb, "rvol": rvol,
            "previous_day_levels": _previous_day_levels(frame, reference),
            "decision_weight_changed": False, "lookahead_policy": "Only complete 5-minute candles ending at or before as_of are used."}
