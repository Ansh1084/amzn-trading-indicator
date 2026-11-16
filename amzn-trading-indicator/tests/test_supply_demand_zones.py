"""Tests for Supply and Demand Zones (Support and Resistance Bands)."""
import pandas as pd
import numpy as np
import pytest

from backend.app.services.analysis.market_structure import (
    confirmed_pivots, analyze_structure, _level_zones, ZONE_INVALIDATION_ATR
)


def make_candles(n=60, day="2026-11-09", base_price=250.0):
    stamps = pd.date_range(f"{day} 09:30", periods=n, freq="5min", tz="America/New_York").tz_convert("UTC")
    closes = [base_price + (i % 6) * 0.15 for i in range(n)]
    return pd.DataFrame({
        "timestamp": stamps,
        "open": [c - 0.05 for c in closes],
        "high": [c + 0.10 for c in closes],
        "low": [c - 0.10 for c in closes],
        "close": closes,
        "volume": [1000 + i * 10 for i in range(n)],
        "vwap": closes
    })


def test_supply_and_demand_zones_fields_and_schema():
    df = make_candles(70)
    atr = 0.50
    result = analyze_structure(df, atr)

    assert "support_zones" in result
    assert "resistance_zones" in result
    assert "demand_zones" in result
    assert "supply_zones" in result
    assert result["demand_zones"] == result["support_zones"]
    assert result["supply_zones"] == result["resistance_zones"]

    all_zones = result["demand_zones"] + result["supply_zones"]
    assert len(all_zones) > 0

    for zone in all_zones:
        # Check required fields
        assert "id" in zone
        assert zone["id"].startswith("zone-")
        assert zone["zone_type"] in ("SUPPLY", "DEMAND")
        assert zone["type"] in ("SUPPORT", "RESISTANCE")
        assert zone["status"] in ("ACTIVE", "TESTED", "INVALIDATED")
        assert isinstance(zone["upper"], float)
        assert isinstance(zone["lower"], float)
        assert isinstance(zone["midpoint"], float)
        assert zone["upper"] > zone["lower"]
        assert zone["lower"] <= zone["midpoint"] <= zone["upper"]
        assert "formation_timestamp" in zone
        assert "touches" in zone
        assert zone["touches"] >= 1
        assert "strength_score" in zone
        assert 0.0 <= zone["strength_score"] <= 100.0
        assert "distance_from_price" in zone
        assert "distance_atr" in zone


def test_no_lookahead_in_zone_formation():
    # Swing high at index 4 (5th bar) needs left=2, right=2.
    # Therefore, it cannot be confirmed until bar index 6 (7th bar).
    stamps = pd.date_range("2026-11-09 09:30", periods=10, freq="5min", tz="America/New_York").tz_convert("UTC")
    highs = [100, 101, 102, 103, 110, 104, 102, 101, 100, 99]
    lows = [98, 99, 100, 101, 102, 101, 100, 99, 98, 97]
    closes = [99, 100, 101, 102, 105, 102, 101, 100, 99, 98]
    df = pd.DataFrame({
        "timestamp": stamps,
        "open": closes,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": [1000] * 10
    })

    # At bar index 5 (before right=2 is satisfied), swing at 4 must NOT be visible
    pivots_before = confirmed_pivots(df.iloc[:6])
    assert not any(p["index"] == 4 for p in pivots_before)

    # At bar index 6 (right=2 satisfied), swing at 4 IS confirmed
    pivots_after = confirmed_pivots(df.iloc[:7])
    p4 = [p for p in pivots_after if p["index"] == 4 and p["type"] == "HIGH"]
    assert len(p4) == 1
    assert p4[0]["confirmed_at"] == stamps[6].isoformat()


def test_zone_invalidation_when_price_closes_beyond():
    # Construct a supply zone, then have subsequent candle close strongly above it
    stamps = pd.date_range("2026-11-09 09:30", periods=15, freq="5min", tz="America/New_York").tz_convert("UTC")
    # Bar 3 has peak swing high at 105
    highs = [100, 101, 102, 105, 102, 101, 100, 100, 100, 108, 109, 110, 110, 110, 110]
    lows = [98, 99, 100, 101, 100, 99, 98, 98, 98, 106, 107, 108, 108, 108, 108]
    closes = [99, 100, 101, 103, 101, 100, 99, 99, 99, 107, 108, 109, 109, 109, 109]
    df = pd.DataFrame({
        "timestamp": stamps,
        "open": closes,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": [1000] * 15
    })
    atr = 1.0
    pivots = confirmed_pivots(df)
    current_price = float(df.close.iloc[-1])
    supply_zones = _level_zones(df, pivots, atr, current_price, "RESISTANCE")

    assert len(supply_zones) >= 1
    # Since current price is 109 and breakout closed at 107+ well above zone (~105), it must be INVALIDATED
    zone = supply_zones[0]
    assert zone["zone_type"] == "SUPPLY"
    assert zone["status"] == "INVALIDATED"


def test_no_overlapping_zones_of_same_type():
    # Two swing highs with very close prices that overlap
    stamps = pd.date_range("2026-11-09 09:30", periods=20, freq="5min", tz="America/New_York").tz_convert("UTC")
    highs = [100, 101, 105.0, 102, 101, 100, 101, 105.1, 102, 101, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    lows = [98, 99, 101, 100, 99, 98, 99, 101, 100, 99, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98]
    closes = [99, 100, 103, 101, 100, 99, 100, 103, 101, 100, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]
    df = pd.DataFrame({
        "timestamp": stamps,
        "open": closes,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": [1000] * 20
    })
    atr = 1.0
    pivots = confirmed_pivots(df)
    current_price = float(df.close.iloc[-1])
    supply_zones = _level_zones(df, pivots, atr, current_price, "RESISTANCE")

    # Check that no two supply zones overlap in price
    for i in range(len(supply_zones) - 1):
        z1 = supply_zones[i]
        z2 = supply_zones[i + 1]
        assert z1["upper"] < z2["lower"] or z2["upper"] < z1["lower"], f"Zones overlap: {z1} vs {z2}"
