"""Confirmed pivots, ATR-relative levels, and close-confirmed price structure."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from backend.app.services.analysis.config import (
    BREAKOUT_PENETRATION_ATR, BREAKOUT_RELATIVE_VOLUME, MAX_LEVEL_SESSIONS,
    MIN_TOUCH_SESSIONS, RETEST_DISTANCE_ATR, SWING_LEFT, SWING_RIGHT,
    TIMEZONE, ZONE_TOLERANCE_ATR, RETEST_LOOKBACK_BARS,
)


def confirmed_pivots(candles: pd.DataFrame, left: int = SWING_LEFT,
                     right: int = SWING_RIGHT) -> list[dict]:
    """Return pivots only after all right-confirmation bars are present."""
    if left < 1 or right < 1:
        raise ValueError("pivot left/right windows must be positive")
    high, low = candles.high.to_numpy(float), candles.low.to_numpy(float)
    stamps = pd.to_datetime(candles.timestamp, utc=True)
    local_days = stamps.dt.tz_convert(TIMEZONE).dt.date
    pivots = []
    for index in range(left, len(candles) - right):
        # A swing is not allowed to span a session boundary.
        session = local_days.iloc[index-left:index+right+1]
        if session.nunique() != 1:
            continue
        high_window = high[index-left:index+right+1]
        low_window = low[index-left:index+right+1]
        if high[index] == high_window.max() and np.count_nonzero(high_window == high[index]) == 1:
            pivots.append({"type": "HIGH", "index": index, "timestamp": stamps.iloc[index].isoformat(),
                           "confirmed_at": stamps.iloc[index+right].isoformat(), "price": float(high[index])})
        if low[index] == low_window.min() and np.count_nonzero(low_window == low[index]) == 1:
            pivots.append({"type": "LOW", "index": index, "timestamp": stamps.iloc[index].isoformat(),
                           "confirmed_at": stamps.iloc[index+right].isoformat(), "price": float(low[index])})
    return pivots


def _level_zones(candles: pd.DataFrame, pivots: list[dict], atr: float,
                 current_price: float, level_type: str) -> list[dict]:
    if not np.isfinite(atr) or atr <= 0:
        return []
    relevant = [p for p in pivots if p["type"] == ("LOW" if level_type == "SUPPORT" else "HIGH")]
    session_dates = sorted(set(pd.to_datetime(candles.timestamp, utc=True).dt.tz_convert(TIMEZONE).dt.date))
    if not session_dates:
        return []
    keep_days = set(session_dates[-MAX_LEVEL_SESSIONS:])
    relevant = [p for p in relevant
                if pd.Timestamp(p["confirmed_at"]).tz_convert(TIMEZONE).date() in keep_days]
    tolerance = max(0.01, atr * ZONE_TOLERANCE_ATR)
    clusters: list[list[dict]] = []
    # Price-sorted clustering with a running center prevents unrelated pivots
    # from being merged through a chain of neighboring prices.
    for pivot in sorted(relevant, key=lambda p: p["price"]):
        if not clusters or abs(pivot["price"] - np.median([p["price"] for p in clusters[-1]])) > tolerance:
            clusters.append([pivot])
        else:
            clusters[-1].append(pivot)

    zones = []
    day_position = {day: i for i, day in enumerate(session_dates)}
    current_day = session_dates[-1]
    for cluster in clusters:
        by_day: dict[object, dict] = {}
        for pivot in cluster:
            day = pd.Timestamp(pivot["timestamp"]).tz_convert(TIMEZONE).date()
            # Keep one point per session for a zone: repeated nearby extrema
            # in one session do not inflate the independent touch count.
            existing = by_day.get(day)
            if existing is None or abs(pivot["price"] - current_price) < abs(existing["price"] - current_price):
                by_day[day] = pivot
        points = list(by_day.values())
        if not points:
            continue
        prices = [p["price"] for p in points]
        center = float(np.median(prices))
        lower, upper = max(0.01, min(prices) - tolerance / 2), max(prices) + tolerance / 2
        distance = (current_price - center) if level_type == "SUPPORT" else (center - current_price)
        zone_status = "BROKEN" if (current_price < lower if level_type == "SUPPORT" else current_price > upper) else "INTACT"
        if zone_status == "INTACT":
            window = candles.tail(1)
            if level_type == "SUPPORT" and float(window.low.iloc[-1]) <= upper:
                zone_status = "TESTED"
            elif level_type == "RESISTANCE" and float(window.high.iloc[-1]) >= lower:
                zone_status = "TESTED"
        last_touch_day = max(by_day)
        age = max(0, day_position.get(current_day, 0) - day_position.get(last_touch_day, 0))
        reactions = []
        for p in points:
            start, end = p["index"] + SWING_RIGHT + 1, min(len(candles), p["index"] + SWING_RIGHT + 7)
            if start >= end:
                continue
            future_window = candles.iloc[start:end]
            reaction = (float(future_window.high.max()) - p["price"] if level_type == "SUPPORT"
                        else p["price"] - float(future_window.low.min()))
            reactions.append(max(0.0, reaction) / atr)
        reaction_score = min(1.0, float(np.mean(reactions)) / 1.5) if reactions else 0.0
        touch_score = min(1.0, len(points) / 5)
        age_score = float(np.exp(-age / 20))
        strength = 100 * (0.55 * touch_score + 0.20 * age_score + 0.25 * reaction_score)
        zones.append({
            "lower": float(lower), "upper": float(upper), "midpoint": center,
            "type": level_type, "touches": len(points),
            "touch_sessions": sorted(day.isoformat() for day in by_day),
            "strength_score": round(strength, 2), "distance_from_price": float(distance),
            "distance_atr": float(distance / atr), "age_sessions": age,
            "status": zone_status, "method": "confirmed_2x2_swings_clustered_with_0.35_atr_tolerance",
        })
    return zones


def analyze_structure(candles: pd.DataFrame, atr: float, known_pivots: list[dict] | None = None,
                      index_offset: int = 0) -> dict:
    if candles.empty:
        return {"status": "INSUFFICIENT_DATA", "pivots": [], "zones": []}
    pivots = confirmed_pivots(candles) if known_pivots is None else known_pivots
    zone_pivots=[]
    for pivot in pivots:
        if pivot["index"]>=index_offset:
            zone_pivots.append({**pivot,"index":pivot["index"]-index_offset})
    current = float(candles.close.iloc[-1])
    support_zones = _level_zones(candles, zone_pivots, atr, current, "SUPPORT")
    resistance_zones = _level_zones(candles, zone_pivots, atr, current, "RESISTANCE")
    support = sorted([z for z in support_zones if z["midpoint"] < current], key=lambda z: current-z["midpoint"])
    resistance = sorted([z for z in resistance_zones if z["midpoint"] > current], key=lambda z: z["midpoint"]-current)
    lows = [p for p in pivots if p["type"] == "LOW"]
    highs = [p for p in pivots if p["type"] == "HIGH"]
    structure = {"higher_high": None, "higher_low": None, "lower_high": None, "lower_low": None}
    if len(highs) >= 2:
        structure["higher_high"] = highs[-1]["price"] > highs[-2]["price"]
        structure["lower_high"] = highs[-1]["price"] < highs[-2]["price"]
    if len(lows) >= 2:
        structure["higher_low"] = lows[-1]["price"] > lows[-2]["price"]
        structure["lower_low"] = lows[-1]["price"] < lows[-2]["price"]

    latest = candles.iloc[-1]
    previous_close = float(candles.close.iloc[-2]) if len(candles) >= 2 else current
    rel_vol = latest.get("relative_volume", np.nan)
    rel_vol = float(rel_vol) if pd.notna(rel_vol) else None
    breakout, breakdown = _crossing_events(resistance_zones, support_zones,
        previous_close, current, atr, rel_vol, candles.close.tail(RETEST_LOOKBACK_BARS+1).to_list())
    return {"status": "READY", "pivots": pivots, "support_zones": support_zones,
            "resistance_zones": resistance_zones, "nearest_support": support[0] if support else None,
            "nearest_resistance": resistance[0] if resistance else None,
            "sequence": structure, "breakout": breakout, "breakdown": breakdown,
            "swing_method": {"left_bars": SWING_LEFT, "right_bars": SWING_RIGHT,
                             "confirmation_rule": "pivot exposed only after right-side bars have completed"},
            "zone_method": {"atr_tolerance": ZONE_TOLERANCE_ATR,
                            "independent_touch_rule": "at most one pivot per zone per trading session",
                            "minimum_touch_sessions_for_strong_zone": MIN_TOUCH_SESSIONS}}


def _crossing_events(resistance_zones: list[dict], support_zones: list[dict],
                     previous_close: float, current: float, atr: float,
                     relative_volume: float | None, recent_closes: list[float] | None = None) -> tuple[dict, dict]:
    """Detect a close crossing using all prior zones, including zones now behind price."""
    breakout={"status":"NONE","zone":None,"volume_confirmed":False}
    breakdown={"status":"NONE","zone":None,"volume_confirmed":False}
    candidates=[]
    for zone in resistance_zones:
        if previous_close<=zone["upper"] and current>zone["upper"]:
            candidates.append((abs(current-zone["midpoint"]),zone,"cross"))
        elif previous_close>zone["upper"] and zone["lower"]<=current<=zone["upper"]:
            candidates.append((abs(current-zone["midpoint"]),zone,"retest"))
        elif previous_close>zone["upper"] and current<zone["lower"] and _recent_cross(recent_closes,zone["upper"],"UP"):
            candidates.append((abs(previous_close-zone["midpoint"]),zone,"failed"))
    if candidates:
        _,zone,kind=min(candidates,key=lambda item:item[0])
        if kind=="cross":
            penetration=(current-zone["upper"])/atr if atr>0 else 0.0
            volume_ok=relative_volume is not None and relative_volume>=BREAKOUT_RELATIVE_VOLUME
            breakout={"status":"CONFIRMED_BREAKOUT" if penetration>=BREAKOUT_PENETRATION_ATR and volume_ok else "POTENTIAL_BREAKOUT",
                      "zone":zone,"penetration_atr":penetration,"volume_confirmed":volume_ok}
        else: breakout={"status":"BREAKOUT_RETEST" if kind=="retest" else "FAILED_BREAKOUT","zone":zone,"volume_confirmed":False}
    candidates=[]
    for zone in support_zones:
        if previous_close>=zone["lower"] and current<zone["lower"]:
            candidates.append((abs(current-zone["midpoint"]),zone,"cross"))
        elif previous_close<zone["lower"] and zone["lower"]<=current<=zone["upper"]:
            candidates.append((abs(current-zone["midpoint"]),zone,"retest"))
        elif previous_close<zone["lower"] and current>zone["upper"] and _recent_cross(recent_closes,zone["lower"],"DOWN"):
            candidates.append((abs(previous_close-zone["midpoint"]),zone,"failed"))
    if candidates:
        _,zone,kind=min(candidates,key=lambda item:item[0])
        if kind=="cross":
            penetration=(zone["lower"]-current)/atr if atr>0 else 0.0
            volume_ok=relative_volume is not None and relative_volume>=BREAKOUT_RELATIVE_VOLUME
            breakdown={"status":"CONFIRMED_BREAKDOWN" if penetration>=BREAKOUT_PENETRATION_ATR and volume_ok else "POTENTIAL_BREAKDOWN",
                       "zone":zone,"penetration_atr":penetration,"volume_confirmed":volume_ok}
        else: breakdown={"status":"BREAKDOWN_RETEST" if kind=="retest" else "FAILED_BREAKDOWN","zone":zone,"volume_confirmed":False}
    return breakout,breakdown


def _recent_cross(closes: list[float] | None, level: float, direction: str) -> bool:
    """Require the initiating close-cross within the configured lookback."""
    if not closes or len(closes)<3: return False
    values=[float(x) for x in closes[-(RETEST_LOOKBACK_BARS+1):]]
    pairs=list(zip(values,values[1:]))[:-1]  # exclude current close; caller tests its failure separately
    if direction=="UP": return any(before<=level and after>level for before,after in pairs)
    return any(before>=level and after<level for before,after in pairs)
