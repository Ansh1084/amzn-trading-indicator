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

ZONE_INVALIDATION_ATR = 0.15
ZONE_CLUSTERING_ATR = 1.8
MAX_DISPLAY_ZONES = 2


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


def _merge_overlapping_zones(zones: list[dict], atr: float, current_price: float, is_support: bool) -> list[dict]:
    """Merge genuinely overlapping zones of the same type to prevent chart clutter."""
    if len(zones) <= 1:
        return zones
    max_h = max(1.00, atr * 2.2) if atr > 0 else 1.50
    sorted_zones = sorted(zones, key=lambda z: z["lower"])
    merged: list[dict] = []
    for z in sorted_zones:
        if not merged:
            merged.append(dict(z))
            continue
        prev = merged[-1]
        combo_lower = min(prev["lower"], z["lower"])
        combo_upper = max(prev["upper"], z["upper"])
        # Overlap test: zone lower boundary sits within or touches previous upper boundary
        # and combined height stays within a natural bounded thickness (not a massive multi-dollar blob)
        if z["lower"] <= prev["upper"] and (combo_upper - combo_lower) <= max_h:
            new_lower = combo_lower
            new_upper = combo_upper
            new_midpoint = float(np.round((new_lower + new_upper) / 2.0, 4))
            combined_sessions = sorted(set(prev.get("touch_sessions", []) + z.get("touch_sessions", [])))
            total_touches = max(len(combined_sessions), prev["touches"] + z["touches"])
            earliest_formation = min(prev["formation_timestamp"], z["formation_timestamp"])
            retests = [ts for ts in [prev.get("latest_retest_timestamp"), z.get("latest_retest_timestamp")] if ts]
            latest_retest = max(retests) if retests else None

            # Lifecycle status resolution for merged zone:
            # If any component is ACTIVE or TESTED, the zone remains alive and respected.
            if prev["status"] in ("ACTIVE", "TESTED") or z["status"] in ("ACTIVE", "TESTED"):
                status = "TESTED" if (prev["status"] == "TESTED" or z["status"] == "TESTED" or total_touches > 1) else "ACTIVE"
            else:
                status = "INVALIDATED"

            merged_strength = max(prev["strength_score"], z["strength_score"])
            distance = (current_price - new_midpoint) if is_support else (new_midpoint - current_price)

            prev["lower"] = new_lower
            prev["upper"] = new_upper
            prev["midpoint"] = new_midpoint
            prev["formation_timestamp"] = earliest_formation
            prev["latest_retest_timestamp"] = latest_retest
            prev["touches"] = total_touches
            prev["touch_sessions"] = combined_sessions
            prev["strength_score"] = merged_strength
            prev["status"] = status
            prev["distance_from_price"] = float(np.round(distance, 4))
            prev["distance_atr"] = float(np.round(distance / atr, 2)) if atr > 0 else 0.0
            prev["id"] = f"zone-{prev['zone_type'].lower()}-{earliest_formation.replace(':', '').replace('-', '')[:15]}-{int(new_midpoint * 100)}"
            if status != "INVALIDATED":
                prev["invalidation_timestamp"] = None
        else:
            merged.append(dict(z))
    return merged


def _level_zones(candles: pd.DataFrame, pivots: list[dict], atr: float,
                 current_price: float, level_type: str) -> list[dict]:
    if not np.isfinite(atr) or atr <= 0:
        return []
    is_support = (level_type == "SUPPORT")
    zone_type = "DEMAND" if is_support else "SUPPLY"
    target_pivot_type = "LOW" if is_support else "HIGH"
    relevant = [p for p in pivots if p["type"] == target_pivot_type]
    session_dates = sorted(set(pd.to_datetime(candles.timestamp, utc=True).dt.tz_convert(TIMEZONE).dt.date))
    if not session_dates:
        return []
    keep_days = set(session_dates[-MAX_LEVEL_SESSIONS:])
    relevant = [p for p in relevant
                if pd.Timestamp(p["confirmed_at"]).tz_convert(TIMEZONE).date() in keep_days]
    if not relevant:
        return []

    min_h = max(0.35, atr * 0.5)
    raw_zones: list[dict] = []

    # Process confirmed pivots according to TradingView Wick-and-Body rule:
    # Supply (Resistance): Top = highest wick, Bottom = real body max(open, close)
    # Demand (Support): Bottom = lowest wick, Top = real body min(open, close)
    for p in relevant:
        idx = p["index"]
        if idx >= len(candles):
            continue
        c_open = float(candles.open.iloc[idx])
        c_close = float(candles.close.iloc[idx])
        c_high = float(candles.high.iloc[idx])
        c_low = float(candles.low.iloc[idx])

        if is_support:
            lower = c_low
            body_top = min(c_open, c_close)
            upper = max(body_top, lower + min_h)
        else:
            upper = c_high
            body_bottom = max(c_open, c_close)
            lower = min(body_bottom, upper - min_h)

        conf_end_idx = idx + SWING_RIGHT + 1
        subsequent_bars = candles.iloc[conf_end_idx:] if conf_end_idx < len(candles) else candles.iloc[0:0]

        retest_count = 0
        latest_retest_timestamp = None
        invalidated = False
        inval_time = None
        invalidation_threshold = lower - ZONE_INVALIDATION_ATR * atr if is_support else upper + ZONE_INVALIDATION_ATR * atr

        if not subsequent_bars.empty:
            closes = subsequent_bars.close.to_numpy(float)
            highs = subsequent_bars.high.to_numpy(float)
            lows = subsequent_bars.low.to_numpy(float)
            sub_stamps = pd.to_datetime(subsequent_bars.timestamp, utc=True)

            for b_idx in range(len(subsequent_bars)):
                c = closes[b_idx]
                h = highs[b_idx]
                l = lows[b_idx]

                if is_support:
                    if c < invalidation_threshold:
                        invalidated = True
                        inval_time = sub_stamps.iloc[b_idx].isoformat()
                        break
                    elif l <= upper:
                        retest_count += 1
                        latest_retest_timestamp = sub_stamps.iloc[b_idx].isoformat()
                else:
                    if c > invalidation_threshold:
                        invalidated = True
                        inval_time = sub_stamps.iloc[b_idx].isoformat()
                        break
                    elif h >= lower:
                        retest_count += 1
                        latest_retest_timestamp = sub_stamps.iloc[b_idx].isoformat()

        # Invalidation check against latest close
        if is_support and current_price < lower:
            invalidated = True
            if inval_time is None:
                inval_time = str(candles.timestamp.iloc[-1])
        elif (not is_support) and current_price > upper:
            invalidated = True
            if inval_time is None:
                inval_time = str(candles.timestamp.iloc[-1])

        if invalidated:
            zone_status = "INVALIDATED"
        elif retest_count > 0:
            zone_status = "TESTED"
        else:
            zone_status = "ACTIVE"

        center = float(np.round((upper + lower) / 2.0, 4))
        distance = (current_price - center) if is_support else (center - current_price)
        total_touches = 1 + retest_count
        strength = min(100.0, 50.0 + min(50.0, retest_count * 10.0))
        formation_timestamp = p["timestamp"]
        zone_id = f"zone-{zone_type.lower()}-{formation_timestamp.replace(':', '').replace('-', '')[:15]}-{int(center * 100)}"

        raw_zones.append({
            "id": zone_id,
            "zone_type": zone_type,
            "type": level_type,
            "lower": float(np.round(lower, 4)),
            "upper": float(np.round(upper, 4)),
            "midpoint": center,
            "formation_timestamp": formation_timestamp,
            "formation_price": float(p["price"]),
            "latest_retest_timestamp": latest_retest_timestamp,
            "invalidation_timestamp": inval_time,
            "touches": total_touches,
            "touch_sessions": [pd.Timestamp(formation_timestamp).tz_convert(TIMEZONE).date().isoformat()],
            "strength_score": round(strength, 2),
            "distance_from_price": float(np.round(distance, 4)),
            "distance_atr": float(np.round(distance / atr, 2)),
            "age_sessions": 0,
            "status": zone_status,
            "method": "tradingview_wick_body_anchored_swings",
        })

    # Merge genuinely overlapping price bands
    merged = _merge_overlapping_zones(raw_zones, atr, current_price, is_support)
    if not merged:
        return []

    # Enforce clear separation between displayed zones
    min_separation = max(1.20, atr * 2.0)

    if is_support:
        # Demand zones floor: strictly below or at current price
        active_candidates = [z for z in merged if z["status"] in ("ACTIVE", "TESTED") and z["upper"] <= current_price + 0.15 * atr]
        active_sorted = sorted(active_candidates, key=lambda z: (
            abs(current_price - z["midpoint"]),  # prioritize nearest floor to current price
            -z["strength_score"],
        ))
        all_fallback = sorted(merged, key=lambda z: (
            0 if z["status"] in ("ACTIVE", "TESTED") else 1,
            abs(current_price - z["midpoint"]),
            -z["strength_score"],
        ))
        candidates = active_sorted if active_sorted else all_fallback
    else:
        # Supply zones ceiling: strictly above or at current price
        active_candidates = [z for z in merged if z["status"] in ("ACTIVE", "TESTED") and z["lower"] >= current_price - 0.15 * atr]
        active_sorted = sorted(active_candidates, key=lambda z: (
            abs(z["midpoint"] - current_price),  # prioritize nearest ceiling to current price
            -z["strength_score"],
        ))
        all_fallback = sorted(merged, key=lambda z: (
            0 if z["status"] in ("ACTIVE", "TESTED") else 1,
            abs(z["midpoint"] - current_price),
            -z["strength_score"],
        ))
        candidates = active_sorted if active_sorted else all_fallback

    # Greedily select top finite, distinct, non-overlapping key zones
    selected = []
    for z in candidates:
        if not any(abs(z["midpoint"] - s["midpoint"]) < min_separation for s in selected):
            selected.append(z)
            if len(selected) >= MAX_DISPLAY_ZONES:
                break

    if not selected and candidates:
        selected.append(candidates[0])

    return sorted(selected, key=lambda z: z["lower"])


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
    return {"status": "READY", "pivots": pivots,
            "support_zones": support_zones, "resistance_zones": resistance_zones,
            "demand_zones": support_zones, "supply_zones": resistance_zones,
            "nearest_support": support[0] if support else None,
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
