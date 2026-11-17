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


def test_multiple_valid_supply_and_demand_zones_across_historical_candles():
    """Verify multiple supply and demand zones form across distinct historical sessions."""
    from src.data.ingestion import latest_processed_path
    from backend.app.services.analysis.technical_analysis import calculate_indicators
    import os

    path = latest_processed_path()
    if not os.path.exists(path):
        pytest.skip("Market data snapshot not found")

    df = pd.read_csv(path).tail(1170).reset_index(drop=True)
    indicators = calculate_indicators(df)
    atr = float(indicators.atr_14.iloc[-1])
    result = analyze_structure(indicators, atr)

    supplies = result["supply_zones"]
    demands = result["demand_zones"]

    # Must have multiple zones distributed across the chart
    assert len(supplies) >= 2, f"Expected at least 2 supply zones, got {len(supplies)}"
    assert len(demands) >= 2, f"Expected at least 2 demand zones, got {len(demands)}"
    assert len(supplies) <= 3, f"Expected at most 3 supply zones, got {len(supplies)}"
    assert len(demands) <= 5, f"Expected at most 5 demand zones, got {len(demands)}"

    # Check distinct formation timestamps
    supply_formations = {s["formation_timestamp"] for s in supplies}
    demand_formations = {d["formation_timestamp"] for d in demands}
    assert len(supply_formations) >= 2, "Supply zones must have distinct historical formation timestamps"
    assert len(demand_formations) >= 2, "Demand zones must have distinct historical formation timestamps"


def test_higher_timeframe_metadata_present():
    """Verify zones include higher-timeframe metadata (1H, 15M, 5M)."""
    from src.data.ingestion import latest_processed_path
    from backend.app.services.analysis.technical_analysis import calculate_indicators

    df = pd.read_csv(latest_processed_path()).tail(1170).reset_index(drop=True)
    indicators = calculate_indicators(df)
    atr = float(indicators.atr_14.iloc[-1])
    result = analyze_structure(indicators, atr)

    all_zones = result["supply_zones"] + result["demand_zones"]
    timeframes = {z.get("timeframe") for z in all_zones}
    assert any(tf in ("1H", "15M") for tf in timeframes), f"Expected higher-timeframe zones, got {timeframes}"
    for z in all_zones:
        assert z.get("timeframe") in ("1H", "15M", "5M")


def test_preservation_of_separate_price_levels():
    """Check that distinct price tiers (not clustered within single dollar) are preserved."""
    from src.data.ingestion import latest_processed_path
    from backend.app.services.analysis.technical_analysis import calculate_indicators

    df = pd.read_csv(latest_processed_path()).tail(1170).reset_index(drop=True)
    indicators = calculate_indicators(df)
    atr = float(indicators.atr_14.iloc[-1])
    result = analyze_structure(indicators, atr)

    supplies = result["supply_zones"]
    if len(supplies) >= 2:
        price_spread = supplies[-1]["midpoint"] - supplies[0]["midpoint"]
        assert price_spread >= 2.0, f"Expected supply zones to span at least $2 price range, got {price_spread:.2f}"


def test_consecutive_candles_do_not_inflate_touch_count():
    """Verify that 20 consecutive candles staying inside a zone do not produce 20 touches."""
    # Pivot at index 2 (high=105), confirmed at bar 4.
    # Bars 5 to 25 remain inside [103.5, 105.0].
    stamps = pd.date_range("2026-11-09 09:30", periods=30, freq="5min", tz="America/New_York").tz_convert("UTC")
    closes = [100.0, 102.0, 104.5, 102.0, 100.0] + [104.0] * 20 + [100.0] * 5
    highs = [c + 0.5 for c in closes]
    highs[2] = 105.0 # swing high
    lows = [c - 0.5 for c in closes]

    df = pd.DataFrame({"timestamp": stamps, "open": closes, "high": highs, "low": lows, "close": closes, "volume": [1000]*30})
    atr = 1.0
    result = _level_zones(df, [{"index": 2, "type": "HIGH", "price": 105.0, "timestamp": stamps[2].isoformat(), "timeframe": "5M"}], atr, 100.0, "RESISTANCE")

    assert len(result) >= 1
    zone = result[0]
    # Despite 20 consecutive candles in the zone, it should count as 1 continuous retest interaction (touches <= 2)
    assert zone["touches"] <= 2, f"Expected <= 2 touches, got {zone['touches']}"
    assert zone["retest_count"] == 1, f"Expected exactly 1 retest, got {zone['retest_count']}"


def test_price_departure_and_return_counts_as_independent_touch():
    """Verify that leaving the zone and returning counts as separate independent touches."""
    # Pivot at index 2 (high=105)
    # Bars 5..8 in zone (Touch 1)
    # Bars 9..14 drop away to 98 (departure > 0.40 ATR)
    # Bars 15..18 return to 104.5 (Touch 2)
    # Bars 19..22 drop away to 98
    # Bars 23..25 return to 104.5 (Touch 3)
    closes = (
        [100.0, 102.0, 104.5, 102.0, 100.0] # formation
        + [104.0, 104.2, 104.1, 104.0]     # interaction 1
        + [98.0, 97.5, 97.0, 97.5, 98.0]    # departure > 0.40 ATR below lower
        + [104.0, 104.2, 104.1, 104.0]     # interaction 2
        + [98.0, 97.5, 97.0, 97.5, 98.0]    # departure
    )
    stamps = pd.date_range("2026-11-09 09:30", periods=len(closes), freq="5min", tz="America/New_York").tz_convert("UTC")
    highs = [c + 0.5 for c in closes]
    highs[2] = 105.0
    lows = [c - 0.5 for c in closes]

    df = pd.DataFrame({"timestamp": stamps, "open": closes, "high": highs, "low": lows, "close": closes, "volume": [1000]*len(closes)})
    atr = 1.0
    result = _level_zones(df, [{"index": 2, "type": "HIGH", "price": 105.0, "timestamp": stamps[2].isoformat(), "timeframe": "5M"}], atr, 97.5, "RESISTANCE")

    assert len(result) >= 1
    zone = result[0]
    assert zone["retest_count"] == 2, f"Expected 2 independent retests, got {zone['retest_count']}"
    assert zone["touches"] == 3, f"Expected 3 total touches (1 formation + 2 retests), got {zone['touches']}"


def test_supply_zones_capped_at_three_most_significant_and_demand_at_five():
    """Verify display rules: exactly at most 3 strongest supply zones, up to 5 demand zones."""
    from src.data.ingestion import latest_processed_path
    from backend.app.services.analysis.technical_analysis import calculate_indicators
    import os

    path = latest_processed_path()
    if not os.path.exists(path):
        pytest.skip("Market data snapshot not found")

    df = pd.read_csv(path).tail(1170).reset_index(drop=True)
    indicators = calculate_indicators(df)
    atr = float(indicators.atr_14.iloc[-1])
    result = analyze_structure(indicators, atr)

    supplies = result["supply_zones"]
    demands = result["demand_zones"]

    assert len(supplies) <= 3, f"Supply zones must be capped at 3, got {len(supplies)}"
    assert len(demands) <= 5, f"Demand zones must allow up to 5, got {len(demands)}"

    # Check non-overlap / separation of supply zones
    for i in range(len(supplies) - 1):
        z1 = supplies[i]
        z2 = supplies[i + 1]
        assert z1["upper"] <= z2["lower"] or z2["upper"] <= z1["lower"], f"Supply zones overlap: {z1} vs {z2}"
        assert abs(z1["midpoint"] - z2["midpoint"]) >= 0.60, f"Supply zones too close: {z1['midpoint']} vs {z2['midpoint']}"

    # Verify all selected supply zones are valid and prioritized
    for z in supplies:
        assert z["status"] in ("ACTIVE", "TESTED"), f"Expected valid supply zone, got {z['status']}"
        assert z.get("timeframe") in ("1H", "15M", "5M")


