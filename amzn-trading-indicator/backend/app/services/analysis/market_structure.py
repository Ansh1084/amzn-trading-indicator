"""Confirmed pivots, multi-timeframe ATR-relative levels, and close-confirmed price structure."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

from backend.app.services.analysis.config import (
    BREAKOUT_PENETRATION_ATR, BREAKOUT_RELATIVE_VOLUME, MAX_LEVEL_SESSIONS,
    MIN_TOUCH_SESSIONS, RETEST_DISTANCE_ATR, SWING_LEFT, SWING_RIGHT,
    TIMEZONE, ZONE_TOLERANCE_ATR, RETEST_LOOKBACK_BARS,
)

ZONE_INVALIDATION_ATR = 0.25
ZONE_CLUSTERING_ATR = 1.8
MAX_DISPLAY_ZONES = 5
MAX_DISPLAY_SUPPLY_ZONES = 3
MAX_DISPLAY_DEMAND_ZONES = 5


def confirmed_pivots(candles: pd.DataFrame, left: int = SWING_LEFT,
                     right: int = SWING_RIGHT) -> list[dict]:
    """Return pivots only after all right-confirmation bars are present."""
    if left < 1 or right < 1:
        raise ValueError("pivot left/right windows must be positive")
    if candles.empty or len(candles) < (left + right + 1):
        return []
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


def extract_mtf_pivots(candles: pd.DataFrame, left: int = SWING_LEFT,
                       right: int = SWING_RIGHT) -> list[dict]:
    """Extract pivots across 1H, 15M, and 5M timeframes derived from 5-minute candles."""
    if candles.empty or len(candles) < (left + right + 1):
        return []

    stamps = pd.to_datetime(candles["timestamp"], utc=True)

    # 1. 5M pivots (tactical intraday levels)
    p_5m = confirmed_pivots(candles, left=left, right=right)
    for p in p_5m:
        p["timeframe"] = "5M"

    pivots_by_key: dict[tuple[str, str], dict] = {}
    for p in p_5m:
        pivots_by_key[(p["type"], p["timestamp"])] = p

    # 2. Derive 15M and 1H pivots if sufficient candles exist
    if len(candles) >= 15:
        try:
            df_indexed = candles[["open", "high", "low", "close", "volume"]].copy()
            df_indexed.index = stamps
            df_indexed = df_indexed.sort_index()

            # 15M Resampling (intermediate swing structure)
            df_15m = df_indexed.resample("15min").agg({
                "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
            }).dropna().reset_index()

            if len(df_15m) >= (left + right + 1):
                p_15m = confirmed_pivots(df_15m, left=left, right=right)
                for p in p_15m:
                    p["timeframe"] = "15M"
                    p_ts = pd.Timestamp(p["timestamp"])
                    m5_mask = stamps >= p_ts
                    p["index"] = int(np.argmax(m5_mask)) if m5_mask.any() else p["index"]
                    # 15M pivots overwrite or enhance 5M
                    pivots_by_key[(p["type"], p["timestamp"])] = p

            # 1H Resampling (macro structural levels)
            if len(candles) >= 30:
                df_1h = df_indexed.resample("1h").agg({
                    "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
                }).dropna().reset_index()

                if len(df_1h) >= (left + right + 1):
                    # For 1H, allow cross-session pivots to capture overnight peaks & multi-day highs/lows
                    high_1h = df_1h.high.to_numpy(float)
                    low_1h = df_1h.low.to_numpy(float)
                    stamps_1h = pd.to_datetime(df_1h.timestamp, utc=True)
                    for i in range(left, len(df_1h) - right):
                        hw = high_1h[i-left:i+right+1]
                        lw = low_1h[i-left:i+right+1]
                        if high_1h[i] == hw.max() and np.count_nonzero(hw == high_1h[i]) == 1:
                            ts = stamps_1h.iloc[i].isoformat()
                            m5_mask = stamps >= stamps_1h.iloc[i]
                            idx_5m = int(np.argmax(m5_mask)) if m5_mask.any() else i
                            pivots_by_key[("HIGH", ts)] = {
                                "type": "HIGH", "index": idx_5m, "timestamp": ts,
                                "confirmed_at": stamps_1h.iloc[i+right].isoformat(),
                                "price": float(high_1h[i]), "timeframe": "1H",
                            }
                        if low_1h[i] == lw.min() and np.count_nonzero(lw == low_1h[i]) == 1:
                            ts = stamps_1h.iloc[i].isoformat()
                            m5_mask = stamps >= stamps_1h.iloc[i]
                            idx_5m = int(np.argmax(m5_mask)) if m5_mask.any() else i
                            pivots_by_key[("LOW", ts)] = {
                                "type": "LOW", "index": idx_5m, "timestamp": ts,
                                "confirmed_at": stamps_1h.iloc[i+right].isoformat(),
                                "price": float(low_1h[i]), "timeframe": "1H",
                            }
        except Exception:
            pass

    return sorted(pivots_by_key.values(), key=lambda p: p["index"])


def _merge_overlapping_zones(zones: list[dict], atr: float, current_price: float, is_support: bool) -> list[dict]:
    """Merge genuinely overlapping zones of the same type to prevent chart clutter."""
    if len(zones) <= 1:
        return zones
    max_h = max(1.20, atr * 2.5) if atr > 0 else 1.50
    sorted_zones = sorted(zones, key=lambda z: (z["lower"], -z["upper"]))
    merged: list[dict] = []
    tf_rank = {"1H": 3, "15M": 2, "5M": 1}

    for z in sorted_zones:
        if not merged:
            merged.append(dict(z))
            continue
        prev = merged[-1]
        combo_lower = min(prev["lower"], z["lower"])
        combo_upper = max(prev["upper"], z["upper"])

        # Overlap test: zone lower boundary sits within or touches previous upper boundary
        # and combined height stays within a natural bounded thickness
        overlap = (z["lower"] <= prev["upper"])
        if overlap and (combo_upper - combo_lower) <= max_h:
            new_lower = combo_lower
            new_upper = combo_upper
            new_midpoint = float(np.round((new_lower + new_upper) / 2.0, 4))
            combined_sessions = sorted(set(prev.get("touch_sessions", []) + z.get("touch_sessions", [])))
            merged_retests = max(prev.get("retest_count", 0), z.get("retest_count", 0))
            total_touches = max(len(combined_sessions), 1 + merged_retests)
            earliest_formation = min(prev["formation_timestamp"], z["formation_timestamp"])
            retests = [ts for ts in [prev.get("latest_retest_timestamp"), z.get("latest_retest_timestamp")] if ts]
            latest_retest = max(retests) if retests else None

            # Lifecycle status resolution for merged zone:
            # If any component is ACTIVE or TESTED, the zone remains alive and respected.
            if prev["status"] in ("ACTIVE", "TESTED") or z["status"] in ("ACTIVE", "TESTED"):
                status = "TESTED" if (prev["status"] == "TESTED" or z["status"] == "TESTED" or total_touches > 1) else "ACTIVE"
            else:
                status = "INVALIDATED"

            # Prefer higher timeframe tag (1H > 15M > 5M)
            if tf_rank.get(z.get("timeframe", "5M"), 1) > tf_rank.get(prev.get("timeframe", "5M"), 1):
                prev["timeframe"] = z.get("timeframe", "5M")

            merged_strength = max(prev["strength_score"], z["strength_score"])
            distance = (current_price - new_midpoint) if is_support else (new_midpoint - current_price)

            invals = [ts for ts in [prev.get("invalidation_timestamp"), z.get("invalidation_timestamp")] if ts]
            latest_inval = max(invals) if invals else None

            prev["lower"] = new_lower
            prev["upper"] = new_upper
            prev["midpoint"] = new_midpoint
            prev["formation_timestamp"] = earliest_formation
            prev["latest_retest_timestamp"] = latest_retest
            prev["touches"] = total_touches
            prev["retest_count"] = merged_retests
            prev["touch_sessions"] = combined_sessions
            prev["strength_score"] = merged_strength
            prev["status"] = status
            prev["distance_from_price"] = float(np.round(distance, 4))
            prev["distance_atr"] = float(np.round(distance / atr, 2)) if atr > 0 else 0.0
            prev["id"] = f"zone-{prev['zone_type'].lower()}-{earliest_formation.replace(':', '').replace('-', '')[:15]}-{int(new_midpoint * 100)}"
            if status != "INVALIDATED":
                prev["invalidation_timestamp"] = None
            else:
                prev["invalidation_timestamp"] = latest_inval
        else:
            merged.append(dict(z))
    return merged


def _select_distributed_zones(merged_zones: list[dict], current_price: float,
                              atr: float, is_support: bool, max_zones: int | None = None) -> list[dict]:
    """Select distinct, non-overlapping zones distributed across the price action.
    For supply/resistance zones, strictly selects at most 3 of the strongest valid zones.
    For demand/support zones, keeps existing selection logic up to 5 zones.
    """
    limit = max_zones if max_zones is not None else (MAX_DISPLAY_DEMAND_ZONES if is_support else MAX_DISPLAY_SUPPLY_ZONES)
    min_separation = max(0.60, atr * 1.2) if atr > 0 else 0.60

    def conflicts(z: dict, existing: list[dict]) -> bool:
        for s in existing:
            # Overlapping price interval
            if max(z["lower"], s["lower"]) <= min(z["upper"], s["upper"]):
                return True
            # Midpoint separation
            if abs(z["midpoint"] - s["midpoint"]) < min_separation:
                return True
        return False

    active = [z for z in merged_zones if z["status"] in ("ACTIVE", "TESTED")]
    invalid = [z for z in merged_zones if z["status"] == "INVALIDATED"]

    if is_support:
        # Existing demand selection logic unchanged:
        active_sorted = sorted(active, key=lambda z: (
            -z["strength_score"],
            -len(z.get("touch_sessions", [])),
            abs(current_price - z["midpoint"]),
        ))
    else:
        # Supply zones: Rank using higher timeframe confirmation (1H > 15M > 5M),
        # zone quality score, independent retests, multi-session rejection, and recency.
        def supply_rank_key(z: dict):
            # Timeframe weighting (1H prioritized over 15M and 5M)
            tf_pts = {"1H": 25.0, "15M": 12.0, "5M": 0.0}.get(z.get("timeframe", "5M"), 0.0)
            quality_pts = float(z.get("strength_score", 0.0)) * 0.40
            retest_pts = min(20.0, float(z.get("retest_count", 0)) * 5.0)
            session_pts = min(15.0, len(z.get("touch_sessions", [])) * 5.0)
            # Active/tested priority (unbroken defense)
            unbroken_bonus = 30.0 if z.get("status") in ("ACTIVE", "TESTED") else 0.0
            composite_strength = unbroken_bonus + tf_pts + quality_pts + retest_pts + session_pts

            # Recency: epoch timestamp of latest retest or formation
            ts_str = z.get("latest_retest_timestamp") or z.get("formation_timestamp", "")
            try:
                epoch_ts = pd.Timestamp(ts_str).timestamp()
            except Exception:
                epoch_ts = 0.0

            return (
                -composite_strength,
                -float(z.get("strength_score", 0.0)),
                -epoch_ts,
            )

        active_sorted = sorted(active, key=supply_rank_key)

    selected: list[dict] = []
    for z in active_sorted:
        if not conflicts(z, selected):
            selected.append(z)
            if len(selected) >= limit:
                break

    # If fewer than limit zones exist:
    # For demand zones, preserve existing backfill with historical invalidated zones.
    # For supply zones, do not force 3 if fewer valid zones exist; only consider historical
    # invalidated zones if absolutely zero valid supply zones exist at all.
    if is_support and len(selected) < limit:
        invalid_sorted = sorted(invalid, key=lambda z: (
            -z["strength_score"],
            -len(z.get("touch_sessions", [])),
            abs(current_price - z["midpoint"]),
        ))
        for z in invalid_sorted:
            if not conflicts(z, selected):
                selected.append(z)
                if len(selected) >= limit:
                    break
    elif not is_support and len(selected) == 0 and invalid:
        invalid_sorted = sorted(invalid, key=lambda z: (
            -z["strength_score"],
            -len(z.get("touch_sessions", [])),
            abs(current_price - z["midpoint"]),
        ))
        for z in invalid_sorted:
            if not conflicts(z, selected):
                selected.append(z)
                if len(selected) >= limit:
                    break

    return sorted(selected, key=lambda z: z["lower"])


def _level_zones(candles: pd.DataFrame, pivots: list[dict], atr: float,
                 current_price: float, level_type: str) -> list[dict]:
    if not np.isfinite(atr) or atr <= 0:
        return []
    is_support = (level_type == "SUPPORT")
    zone_type = "DEMAND" if is_support else "SUPPLY"
    target_pivot_type = "LOW" if is_support else "HIGH"
    relevant = [p for p in pivots if p["type"] == target_pivot_type]
    stamps = pd.to_datetime(candles.timestamp, utc=True)
    session_dates = sorted(set(stamps.dt.tz_convert(TIMEZONE).dt.date))
    if not session_dates:
        return []
    keep_days = set(session_dates[-MAX_LEVEL_SESSIONS:])
    relevant = [p for p in relevant
                if pd.Timestamp(p.get("confirmed_at", p["timestamp"])).tz_convert(TIMEZONE).date() in keep_days]
    if not relevant:
        return []

    raw_zones: list[dict] = []
    candle_times = stamps
    candle_closes = candles.close.to_numpy(float)
    candle_highs = candles.high.to_numpy(float)
    candle_lows = candles.low.to_numpy(float)
    candle_sessions = candle_times.dt.tz_convert(TIMEZONE).dt.date

    # Process confirmed pivots according to TradingView Wick-and-Body rule:
    # Supply (Resistance): Top = highest wick, Bottom = real body max(open, close)
    # Demand (Support): Bottom = lowest wick, Top = real body min(open, close)
    for p in relevant:
        idx = p.get("index", 0)
        tf = p.get("timeframe", "5M")
        min_h = max(0.35, atr * (0.6 if tf == "1H" else (0.5 if tf == "15M" else 0.4)))
        max_h = max(0.90, atr * 2.2)

        if idx < len(candles):
            c_open = float(candles.open.iloc[idx])
            c_close = float(candles.close.iloc[idx])
            c_high = float(candles.high.iloc[idx])
            c_low = float(candles.low.iloc[idx])
        else:
            c_open = c_close = c_high = c_low = float(p["price"])

        if is_support:
            lower = float(p["price"])
            body_top = min(c_open, c_close)
            upper = min(lower + max_h, max(body_top, lower + min_h))
        else:
            upper = float(p["price"])
            body_bottom = max(c_open, c_close)
            lower = max(upper - max_h, min(body_bottom, upper - min_h))

        p_time = pd.Timestamp(p["timestamp"])
        after_mask = candle_times >= p_time
        start_idx = int(np.argmax(after_mask)) if after_mask.any() else idx + SWING_RIGHT + 1
        conf_start = max(start_idx + 1, idx + SWING_RIGHT + 1)

        invalidation_buf = ZONE_INVALIDATION_ATR * atr
        separation_buf = max(0.20, 0.40 * atr)
        independent_retests = 0
        in_zone = True
        latest_retest_timestamp = None
        invalidated = False
        inval_time = None
        touch_sessions = {pd.Timestamp(p["timestamp"]).tz_convert(TIMEZONE).date().isoformat()}
        last_touch_sess = pd.Timestamp(p["timestamp"]).tz_convert(TIMEZONE).date().isoformat()

        # Track if price departed from the zone during confirmation
        pre_start = min(start_idx + 1, len(candles))
        for pre_i in range(pre_start, min(conf_start, len(candles))):
            c_pre = candle_closes[pre_i]
            h_pre = candle_highs[pre_i]
            l_pre = candle_lows[pre_i]
            if is_support:
                if l_pre > upper and c_pre > upper + separation_buf:
                    in_zone = False
            else:
                if h_pre < lower and c_pre < lower - separation_buf:
                    in_zone = False

        if conf_start < len(candles):
            for b_idx in range(conf_start, len(candles)):
                c = candle_closes[b_idx]
                h = candle_highs[b_idx]
                l = candle_lows[b_idx]
                t_str = candle_times.iloc[b_idx].isoformat()
                sess_str = candle_sessions.iloc[b_idx].isoformat()

                if is_support:
                    if c < lower - invalidation_buf:
                        invalidated = True
                        inval_time = t_str
                        break
                    is_touch = (l <= upper and c >= lower - invalidation_buf)
                    is_departed = (l > upper and c > upper + separation_buf)
                    if is_touch:
                        if not in_zone or sess_str != last_touch_sess:
                            independent_retests += 1
                            in_zone = True
                            last_touch_sess = sess_str
                        latest_retest_timestamp = t_str
                        touch_sessions.add(sess_str)
                    elif is_departed:
                        in_zone = False
                else:
                    if c > upper + invalidation_buf:
                        invalidated = True
                        inval_time = t_str
                        break
                    is_touch = (h >= lower and c <= upper + invalidation_buf)
                    is_departed = (h < lower and c < lower - separation_buf)
                    if is_touch:
                        if not in_zone or sess_str != last_touch_sess:
                            independent_retests += 1
                            in_zone = True
                            last_touch_sess = sess_str
                        latest_retest_timestamp = t_str
                        touch_sessions.add(sess_str)
                    elif is_departed:
                        in_zone = False

        # Invalidation check against latest close / current price
        if is_support and current_price < lower - invalidation_buf:
            invalidated = True
            if inval_time is None:
                inval_time = str(candles.timestamp.iloc[-1])
        elif (not is_support) and current_price > upper + invalidation_buf:
            invalidated = True
            if inval_time is None:
                inval_time = str(candles.timestamp.iloc[-1])

        if invalidated:
            zone_status = "INVALIDATED"
        elif independent_retests > 0:
            zone_status = "TESTED"
        else:
            zone_status = "ACTIVE"

        center = float(np.round((upper + lower) / 2.0, 4))
        distance = (current_price - center) if is_support else (center - current_price)
        total_touches = max(len(touch_sessions), 1 + independent_retests)

        # Multi-factor strength score:
        # Timeframe weight: 1H (35 pts), 15M (20 pts), 5M (10 pts)
        tf_pts = {"1H": 35.0, "15M": 20.0, "5M": 10.0}.get(tf, 10.0)
        session_pts = min(40.0, len(touch_sessions) * 12.0)
        retest_pts = min(15.0, independent_retests * 3.0)
        active_pts = 20.0 if zone_status in ("ACTIVE", "TESTED") else 0.0
        strength = min(100.0, tf_pts + session_pts + retest_pts + active_pts)

        formation_timestamp = p["timestamp"]
        zone_id = f"zone-{zone_type.lower()}-{formation_timestamp.replace(':', '').replace('-', '')[:15]}-{int(center * 100)}"

        raw_zones.append({
            "id": zone_id,
            "zone_type": zone_type,
            "type": level_type,
            "timeframe": tf,
            "lower": float(np.round(lower, 4)),
            "upper": float(np.round(upper, 4)),
            "midpoint": center,
            "formation_timestamp": formation_timestamp,
            "formation_price": float(p["price"]),
            "latest_retest_timestamp": latest_retest_timestamp,
            "invalidation_timestamp": inval_time,
            "touches": total_touches,
            "retest_count": independent_retests,
            "touch_sessions": sorted(touch_sessions),
            "strength_score": round(strength, 2),
            "distance_from_price": float(np.round(distance, 4)),
            "distance_atr": float(np.round(distance / atr, 2)),
            "age_sessions": len(touch_sessions),
            "status": zone_status,
            "method": "tradingview_wick_body_anchored_swings",
        })

    # Merge genuinely overlapping price bands
    merged = _merge_overlapping_zones(raw_zones, atr, current_price, is_support)
    if not merged:
        return []

    # Select top distributed non-overlapping zones across historical price action
    max_display = MAX_DISPLAY_DEMAND_ZONES if is_support else MAX_DISPLAY_SUPPLY_ZONES
    return _select_distributed_zones(merged, current_price, atr, is_support, max_zones=max_display)


def analyze_structure(candles: pd.DataFrame, atr: float, known_pivots: list[dict] | None = None,
                      index_offset: int = 0) -> dict:
    if candles.empty:
        return {"status": "INSUFFICIENT_DATA", "pivots": [], "zones": []}
    pivots = extract_mtf_pivots(candles) if known_pivots is None else known_pivots
    zone_pivots = []
    for pivot in pivots:
        if pivot.get("index", 0) >= index_offset:
            zone_pivots.append({**pivot, "index": pivot.get("index", 0) - index_offset})
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
    breakout = {"status": "NONE", "zone": None, "volume_confirmed": False}
    breakdown = {"status": "NONE", "zone": None, "volume_confirmed": False}
    candidates = []
    for zone in resistance_zones:
        if previous_close <= zone["upper"] and current > zone["upper"]:
            candidates.append((abs(current - zone["midpoint"]), zone, "cross"))
        elif previous_close > zone["upper"] and zone["lower"] <= current <= zone["upper"]:
            candidates.append((abs(current - zone["midpoint"]), zone, "retest"))
        elif previous_close > zone["upper"] and current < zone["lower"] and _recent_cross(recent_closes, zone["upper"], "UP"):
            candidates.append((abs(previous_close - zone["midpoint"]), zone, "failed"))
    if candidates:
        _, zone, kind = min(candidates, key=lambda item: item[0])
        if kind == "cross":
            penetration = (current - zone["upper"]) / atr if atr > 0 else 0.0
            volume_ok = relative_volume is not None and relative_volume >= BREAKOUT_RELATIVE_VOLUME
            breakout = {"status": "CONFIRMED_BREAKOUT" if penetration >= BREAKOUT_PENETRATION_ATR and volume_ok else "POTENTIAL_BREAKOUT",
                        "zone": zone, "penetration_atr": penetration, "volume_confirmed": volume_ok}
        elif kind == "retest":
            penetration = (zone["upper"] - current) / atr if atr > 0 else 0.0
            retest_ok = penetration <= RETEST_DISTANCE_ATR
            breakout = {"status": "BREAKOUT_RETEST" if retest_ok else "POTENTIAL_BREAKOUT",
                        "zone": zone, "penetration_atr": penetration, "volume_confirmed": False}
        elif kind == "failed":
            breakout = {"status": "FAILED_BREAKOUT", "zone": zone, "penetration_atr": 0.0, "volume_confirmed": False}

    support_candidates = []
    for zone in support_zones:
        if previous_close >= zone["lower"] and current < zone["lower"]:
            support_candidates.append((abs(current - zone["midpoint"]), zone, "cross"))
        elif previous_close < zone["lower"] and zone["lower"] <= current <= zone["upper"]:
            support_candidates.append((abs(current - zone["midpoint"]), zone, "retest"))
        elif previous_close < zone["lower"] and current > zone["upper"] and _recent_cross(recent_closes, zone["lower"], "DOWN"):
            support_candidates.append((abs(previous_close - zone["midpoint"]), zone, "failed"))
    if support_candidates:
        _, zone, kind = min(support_candidates, key=lambda item: item[0])
        if kind == "cross":
            penetration = (zone["lower"] - current) / atr if atr > 0 else 0.0
            volume_ok = relative_volume is not None and relative_volume >= BREAKOUT_RELATIVE_VOLUME
            breakdown = {"status": "CONFIRMED_BREAKDOWN" if penetration >= BREAKOUT_PENETRATION_ATR and volume_ok else "POTENTIAL_BREAKDOWN",
                         "zone": zone, "penetration_atr": penetration, "volume_confirmed": volume_ok}
        elif kind == "retest":
            penetration = (current - zone["lower"]) / atr if atr > 0 else 0.0
            retest_ok = penetration <= RETEST_DISTANCE_ATR
            breakdown = {"status": "BREAKDOWN_RETEST" if retest_ok else "POTENTIAL_BREAKDOWN",
                         "zone": zone, "penetration_atr": penetration, "volume_confirmed": False}
        elif kind == "failed":
            breakdown = {"status": "FAILED_BREAKDOWN", "zone": zone, "penetration_atr": 0.0, "volume_confirmed": False}

    return breakout, breakdown


def _recent_cross(closes: list[float] | None, level: float, direction: str) -> bool:
    if not closes or len(closes) < 2:
        return False
    for prev, cur in zip(closes[:-1], closes[1:]):
        if direction == "UP" and prev <= level and cur > level:
            return True
        if direction == "DOWN" and prev >= level and cur < level:
            return True
    return False
