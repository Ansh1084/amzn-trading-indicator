"""Unit tests for the 60-Minute Trading Decision System."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from backend.app.services.analysis.decision_engine_60m import (
    DECISION_THRESHOLD,
    HORIZON_BARS,
    HORIZON_MINUTES,
    MIN_TARGET_RISK_REWARD,
    evaluate_60m_decision,
)
from backend.app.services.analysis.persistence_60m import (
    ensure_60m_tables,
    get_60m_performance_summary,
    get_latest_60m_decision,
    list_60m_signals,
    resolve_60m_signals,
    save_60m_decision,
)
from src.forecasts.store import connect

ET = ZoneInfo("America/New_York")


def make_mock_data(
    candle_time_et: str = "2026-11-16 11:00:00",
    price: float = 250.0,
    trend_type: str = "BULLISH",
    rsi: float = 58.0,
    macd_hist: float = 0.15,
    relative_volume: float = 1.3,
    vwap: float = 249.0,
    atr: float = 1.0,
    near_res_dist: float = 2.0,
    near_sup_dist: float = 2.0,
    breakout: str = "NONE",
    breakdown: str = "NONE",
):
    """Construct mock candles, indicators, structure, regime, and market intelligence."""
    ts_local = pd.Timestamp(candle_time_et, tz=ET)
    ts_utc = ts_local.tz_convert("UTC")

    candle = pd.DataFrame([{
        "timestamp": ts_utc.isoformat(),
        "open": price - 0.2,
        "high": price + 0.3,
        "low": price - 0.3,
        "close": price,
        "volume": 50000.0,
    }])

    # Indicator row
    is_bull = trend_type == "BULLISH"
    is_bear = trend_type == "BEARISH"

    indicators = pd.DataFrame([{
        "timestamp": ts_utc.isoformat(),
        "close": price,
        "ema_9": price + 0.5 if is_bull else (price - 0.5 if is_bear else price),
        "ema_20": price if is_bull else (price if is_bear else price),
        "ema_50": price - 0.5 if is_bull else (price + 0.5 if is_bear else price),
        "ema_20_slope_5": 0.001 if is_bull else (-0.001 if is_bear else 0.0),
        "ema_50_slope_10": 0.0005 if is_bull else (-0.0005 if is_bear else 0.0),
        "rsi_14": rsi,
        "rsi_7": rsi + 2,
        "macd_histogram": macd_hist,
        "macd_histogram_change": 0.02 if macd_hist > 0 else -0.02,
        "roc_5_pct": 0.25 if is_bull else (-0.25 if is_bear else 0.0),
        "roc_20_pct": 0.50 if is_bull else (-0.50 if is_bear else 0.0),
        "relative_volume": relative_volume,
        "vwap": vwap,
        "atr_14": atr,
        "bullish_divergence": False,
        "bearish_divergence": False,
    }])

    structure = {
        "sequence": {
            "higher_high": is_bull,
            "higher_low": is_bull,
            "lower_high": is_bear,
            "lower_low": is_bear,
        },
        "breakout": {"status": breakout, "volume_confirmed": breakout == "CONFIRMED_BREAKOUT"},
        "breakdown": {"status": breakdown, "volume_confirmed": breakdown == "CONFIRMED_BREAKDOWN"},
        "nearest_support": {"upper": price - (near_sup_dist * atr), "lower": price - (near_sup_dist * atr) - 0.5},
        "nearest_resistance": {"lower": price + (near_res_dist * atr), "upper": price + (near_res_dist * atr) + 0.5},
    }

    regime = {"classification": "NORMAL", "status": "READY"}

    intel = {
        "trend_alignment": {
            "overall": trend_type,
            "timeframes": [
                {"timeframe": "1H", "trend": trend_type},
                {"timeframe": "15M", "trend": trend_type},
                {"timeframe": "5M", "trend": trend_type},
            ],
        },
        "opening_range": {"classification": "INSIDE_OPENING_RANGE", "high": price + 2.0, "low": price - 2.0},
        "rvol": {"rvol": relative_volume, "classification": "ELEVATED" if relative_volume >= 1.2 else "NORMAL"},
        "previous_day_levels": {"position": {"high": "BELOW", "low": "ABOVE"}},
    }

    return candle, indicators, structure, regime, intel


def test_strong_bullish_conditions_generates_buy():
    candle, indicators, structure, regime, intel = make_mock_data(
        candle_time_et="2026-11-16 10:30:00",
        price=250.0,
        trend_type="BULLISH",
        rsi=62.0,
        macd_hist=0.20,
        relative_volume=1.4,
        vwap=248.5,
        near_res_dist=2.5,
    )

    res = evaluate_60m_decision(candle, indicators, structure, regime, intel)

    assert res["intended_horizon_minutes"] == 60
    assert res["intended_horizon_bars"] == 12
    assert res["decision"] == "BUY"
    assert res["directional_bias"] == "BULLISH"
    assert res["directional_score"] >= DECISION_THRESHOLD
    assert res["risk_plan"]["valid"] is True
    assert res["risk_plan"]["direction"] == "BUY"
    assert res["risk_plan"]["risk_reward_ratio"] >= MIN_TARGET_RISK_REWARD


def test_strong_bearish_conditions_generates_sell():
    candle, indicators, structure, regime, intel = make_mock_data(
        candle_time_et="2026-11-16 10:30:00",
        price=250.0,
        trend_type="BEARISH",
        rsi=38.0,
        macd_hist=-0.20,
        relative_volume=1.4,
        vwap=251.5,
        near_sup_dist=2.5,
    )

    res = evaluate_60m_decision(candle, indicators, structure, regime, intel)

    assert res["decision"] == "SELL"
    assert res["directional_bias"] == "BEARISH"
    assert res["directional_score"] <= -DECISION_THRESHOLD
    assert res["risk_plan"]["valid"] is True
    assert res["risk_plan"]["direction"] == "SELL"
    assert res["risk_plan"]["risk_reward_ratio"] >= MIN_TARGET_RISK_REWARD


def test_mixed_conditions_defaults_to_hold():
    candle, indicators, structure, regime, intel = make_mock_data(
        candle_time_et="2026-11-16 11:30:00",
        price=250.0,
        trend_type="NEUTRAL",
        rsi=50.0,
        macd_hist=0.01,
        relative_volume=0.9,
        vwap=250.0,
    )

    res = evaluate_60m_decision(candle, indicators, structure, regime, intel)

    assert res["decision"] == "HOLD"
    assert abs(res["directional_score"]) < DECISION_THRESHOLD
    assert res["gating_status"] in ("SUB_THRESHOLD", "NEUTRAL")


def test_opening_discovery_curfew_forces_hold():
    # 09:35 ET is inside the 09:30-09:45 opening discovery window
    candle, indicators, structure, regime, intel = make_mock_data(
        candle_time_et="2026-11-16 09:35:00",
        trend_type="BULLISH",
        rsi=65.0,
        relative_volume=2.0,
    )

    res = evaluate_60m_decision(candle, indicators, structure, regime, intel)

    assert res["decision"] == "HOLD"
    assert res["gating_status"] == "OPENING_DISCOVERY"
    assert "09:30–09:45" in res["reasons"][0]


def test_closing_curfew_forces_hold():
    # 15:20 ET is within 45 minutes of the 16:00 ET market close
    candle, indicators, structure, regime, intel = make_mock_data(
        candle_time_et="2026-11-16 15:20:00",
        trend_type="BULLISH",
        rsi=65.0,
    )

    res = evaluate_60m_decision(candle, indicators, structure, regime, intel)

    assert res["decision"] == "HOLD"
    assert res["gating_status"] == "SESSION_CLOSE_PROXIMITY"
    assert "Within 45 minutes" in res["reasons"][0]


def test_price_near_resistance_waits_for_confirmation():
    # Strong bullish score, but resistance is only 0.20 ATR away (< 0.35) without confirmed breakout
    candle, indicators, structure, regime, intel = make_mock_data(
        candle_time_et="2026-11-16 11:00:00",
        trend_type="BULLISH",
        rsi=62.0,
        near_res_dist=0.20,
        breakout="NONE",
    )

    res = evaluate_60m_decision(candle, indicators, structure, regime, intel)

    assert res["decision"] == "WAIT FOR CONFIRMATION"
    assert res["gating_status"] == "RESISTANCE_PROXIMITY"
    assert "overhead resistance" in res["reasons"][-1]


def test_price_near_support_waits_for_confirmation():
    # Strong bearish score, but support is only 0.20 ATR away without confirmed breakdown
    candle, indicators, structure, regime, intel = make_mock_data(
        candle_time_et="2026-11-16 11:00:00",
        trend_type="BEARISH",
        rsi=38.0,
        macd_hist=-0.20,
        relative_volume=1.4,
        vwap=251.5,
        near_sup_dist=0.20,
        breakdown="NONE",
    )

    res = evaluate_60m_decision(candle, indicators, structure, regime, intel)

    assert res["decision"] == "WAIT FOR CONFIRMATION"
    assert res["gating_status"] == "SUPPORT_PROXIMITY"
    assert "major support" in res["reasons"][-1]


def test_missing_volume_handled_without_failure():
    candle, indicators, structure, regime, intel = make_mock_data(
        candle_time_et="2026-11-16 11:00:00",
        relative_volume=np.nan,
    )
    indicators.loc[0, "relative_volume"] = np.nan
    intel["rvol"] = {"rvol": None, "classification": "UNAVAILABLE"}

    res = evaluate_60m_decision(candle, indicators, structure, regime, intel)

    assert res["decision"] in ("BUY", "SELL", "HOLD", "WAIT FOR CONFIRMATION")
    assert "volume" in res["components"]


def test_missing_vwap_handled_without_failure():
    candle, indicators, structure, regime, intel = make_mock_data(
        candle_time_et="2026-11-16 11:00:00",
    )
    indicators.loc[0, "vwap"] = np.nan

    res = evaluate_60m_decision(candle, indicators, structure, regime, intel)

    assert res["decision"] in ("BUY", "SELL", "HOLD", "WAIT FOR CONFIRMATION")
    assert "vwap_price_action" in res["components"]


def test_missing_atr_uses_fallback():
    candle, indicators, structure, regime, intel = make_mock_data(
        candle_time_et="2026-11-16 11:00:00",
        atr=-1.0,
    )
    indicators.loc[0, "atr_14"] = np.nan

    res = evaluate_60m_decision(candle, indicators, structure, regime, intel)

    assert res["decision"] in ("BUY", "SELL", "HOLD", "WAIT FOR CONFIRMATION")
    assert res["risk_plan"] is not None


def test_separation_between_10m_ml_and_60m_decision():
    candle, indicators, structure, regime, intel = make_mock_data(
        candle_time_et="2026-11-16 11:00:00",
    )
    mock_ml_10m = {
        "available": True,
        "direction": "UP",
        "raw_probability_up": 0.54,
        "raw_directional_score": 0.54,
        "model_name": "LGBMClassifier",
        "model_version": "legacy-lightgbm-10m-e1b5cf0",
    }

    res = evaluate_60m_decision(candle, indicators, structure, regime, intel, ml_evidence=mock_ml_10m)

    # 10m model is explicitly segregated as supporting context
    assert res["ml_10m_supporting"]["prediction_horizon_minutes"] == 10
    assert res["ml_10m_supporting"]["role"] == "supporting_micro_momentum_context_only"
    assert res["ml_10m_supporting"]["score_type"] == "uncalibrated_raw_model_score_not_confidence"
    assert res["intended_horizon_minutes"] == 60
    assert res["dedicated_60m_ml_model"]["status"] == "NOT_AVAILABLE"


def test_signal_persistence_and_idempotency(tmp_path: Path):
    db_file = tmp_path / "test_60m.sqlite3"
    candle, indicators, structure, regime, intel = make_mock_data(
        candle_time_et="2026-11-16 10:30:00",
        trend_type="BULLISH",
        rsi=62.0,
    )
    decision = evaluate_60m_decision(candle, indicators, structure, regime, intel)

    stored1 = save_60m_decision(decision, db_path=db_file)
    stored2 = save_60m_decision(decision, db_path=db_file)

    assert stored1["signal_id"] == stored2["signal_id"]

    with connect(db_file) as db:
        count = db.execute("SELECT count(*) FROM analysis_60m_signals").fetchone()[0]
        assert count == 1

    latest = get_latest_60m_decision(db_path=db_file)
    assert latest is not None
    assert latest["decision"] == decision["decision"]
    assert latest["horizon_minutes"] == 60


def test_60m_outcome_resolution_lifecycle(tmp_path: Path):
    db_file = tmp_path / "test_60m.sqlite3"
    t0_et = "2026-11-16 10:30:00"
    candle, indicators, structure, regime, intel = make_mock_data(
        candle_time_et=t0_et,
        price=100.0,
        trend_type="BULLISH",
        rsi=65.0,
        near_res_dist=3.0,
    )
    decision = evaluate_60m_decision(candle, indicators, structure, regime, intel)
    assert decision["decision"] == "BUY"

    save_60m_decision(decision, db_path=db_file)

    # Simulate subsequent 5m candles
    # Stop loss is below 99.0, Target 1 is at ~101.0, Target 2 is at ~102.0
    t0_utc = pd.Timestamp(t0_et, tz=ET).tz_convert("UTC")
    subsequent_records = []
    for i in range(1, 15):
        ts = t0_utc + pd.Timedelta(minutes=5 * i)
        # Bar 1 enters at 100.0, bar 4 hits 102.5 (Target 2)
        close_p = 100.0 + (0.8 * i)
        subsequent_records.append({
            "timestamp": ts.isoformat(),
            "open": 100.0 if i == 1 else close_p - 0.2,
            "high": close_p + 0.5,
            "low": close_p - 0.3,
            "close": close_p,
            "volume": 20000.0,
        })
    subsequent_candles = pd.DataFrame(subsequent_records)

    res = resolve_60m_signals(subsequent_candles, db_path=db_file)
    assert res["resolved_count"] == 1

    with connect(db_file) as db:
        row = dict(db.execute("SELECT * FROM analysis_60m_signals").fetchone())
        assert row["status"] in ("TARGET_2_HIT", "EXPIRED", "CLOSED")
        assert row["net_return_pct"] is not None
        assert row["resolved_at"] is not None

    summary = get_60m_performance_summary(db_path=db_file)
    assert summary["has_data"] is True
    assert summary["resolved_count"] == 1
    assert summary["win_rate"] == 1.0


# ── Explicit 10-Case Verification Suite ───────────────────────────────────────

from backend.app.services.analysis.decision_engine_60m import (
    compute_prediction_schedule,
    validate_trade_plan,
)
from backend.app.services.analysis.persistence_60m import get_active_60m_signal


def test_case_1_hold_with_no_active_signal():
    """Case 1: HOLD with no active signal -> no approved plan, active_signal is None."""
    candle, indicators, structure, regime, intel = make_mock_data(
        candle_time_et="2026-11-16 11:30:00",
        trend_type="NEUTRAL",
        rsi=50.0,
    )
    res = evaluate_60m_decision(candle, indicators, structure, regime, intel, active_60m_signal=None)

    assert res["decision"] == "HOLD"
    assert res["active_signal"] is None
    assert res["risk_plan"]["valid"] is False
    assert res["risk_plan"]["approved"] is False
    assert "No new trade setup approved" in res["risk_plan"]["reason"]


def test_case_2_hold_with_existing_active_sell():
    """Case 2: HOLD with an existing active SELL -> preserves active SELL, no new approved plan."""
    candle, indicators, structure, regime, intel = make_mock_data(
        candle_time_et="2026-11-16 11:30:00",
        price=248.0,
        trend_type="BEARISH",
        rsi=35.0,
    )
    active_sell = {
        "signal_id": "sig-sell-123",
        "decision": "SELL",
        "status": "ACTIVE",
        "trigger_price": 250.0,
        "entry_zone_lower": 249.9,
        "entry_zone_upper": 250.1,
        "stop_loss": 251.25,
        "target_1": 248.75,
        "target_2": 247.50,
        "risk_reward_ratio": 2.0,
        "candle_timestamp": "2026-11-16T11:15:00Z",
    }
    res = evaluate_60m_decision(candle, indicators, structure, regime, intel, active_60m_signal=active_sell)

    assert res["decision"] == "HOLD"
    assert res["gating_status"] == "ALREADY_ACTIVE_SELL"
    assert res["risk_plan"]["valid"] is False
    assert res["risk_plan"]["approved"] is False

    # Existing active position must be present, validated, and clearly separated
    assert res["active_signal"] is not None
    assert res["active_signal"]["direction"] == "SELL"
    assert res["active_signal"]["status"] == "ACTIVE"
    assert res["active_signal"]["is_existing_position"] is True
    assert res["active_signal"]["trigger_price"] == 250.0
    assert res["active_signal"]["stop_loss"] == 251.25
    assert res["active_signal"]["target_2"] == 247.50
    assert res["active_signal"]["valid"] is True
    assert res["active_signal"]["unrealized_pnl_pct"] > 0  # In profit at 248 vs 250 short entry


def test_case_3_hold_with_existing_active_buy():
    """Case 3: HOLD with an existing active BUY -> preserves active BUY, no duplicate new buy."""
    candle, indicators, structure, regime, intel = make_mock_data(
        candle_time_et="2026-11-16 11:30:00",
        price=252.0,
        trend_type="BULLISH",
        rsi=62.0,
    )
    active_buy = {
        "signal_id": "sig-buy-456",
        "decision": "BUY",
        "status": "ACTIVE",
        "trigger_price": 250.0,
        "entry_zone_lower": 249.9,
        "entry_zone_upper": 250.1,
        "stop_loss": 248.75,
        "target_1": 251.25,
        "target_2": 252.50,
        "risk_reward_ratio": 2.0,
        "candle_timestamp": "2026-11-16T11:15:00Z",
    }
    res = evaluate_60m_decision(candle, indicators, structure, regime, intel, active_60m_signal=active_buy)

    assert res["decision"] == "HOLD"
    assert res["gating_status"] == "ALREADY_ACTIVE_BUY"
    assert res["risk_plan"]["valid"] is False
    assert res["risk_plan"]["approved"] is False

    assert res["active_signal"] is not None
    assert res["active_signal"]["direction"] == "BUY"
    assert res["active_signal"]["status"] == "ACTIVE"
    assert res["active_signal"]["is_existing_position"] is True
    assert res["active_signal"]["valid"] is True
    assert res["active_signal"]["unrealized_pnl_pct"] > 0  # In profit at 252 vs 250 long entry


def test_case_4_new_buy_with_approved_execution_plan():
    """Case 4: New BUY with approved execution plan -> validates direction and R:R."""
    candle, indicators, structure, regime, intel = make_mock_data(
        candle_time_et="2026-11-16 10:30:00",
        price=250.0,
        trend_type="BULLISH",
        rsi=62.0,
        macd_hist=0.20,
        relative_volume=1.4,
        vwap=248.5,
        near_res_dist=3.0,
    )
    res = evaluate_60m_decision(candle, indicators, structure, regime, intel)

    assert res["decision"] == "BUY"
    plan = res["risk_plan"]
    assert plan["valid"] is True
    assert plan["approved"] is True
    assert plan["direction"] == "BUY"

    # Trade-plan direction & risk/reward validation
    entry = plan["trigger_price"]
    stop = plan["stop_loss"]
    t1 = plan["target_1"]
    t2 = plan["target_2"]
    rr = plan["risk_reward_ratio"]

    assert stop < entry < t1 < t2
    assert rr >= 1.5

    is_valid, err = validate_trade_plan(plan, "BUY", entry)
    assert is_valid is True
    assert err is None


def test_case_5_new_sell_with_approved_execution_plan():
    """Case 5: New SELL with approved execution plan -> validates direction and R:R."""
    candle, indicators, structure, regime, intel = make_mock_data(
        candle_time_et="2026-11-16 10:30:00",
        price=250.0,
        trend_type="BEARISH",
        rsi=38.0,
        macd_hist=-0.20,
        relative_volume=1.4,
        vwap=251.5,
        near_sup_dist=3.0,
    )
    res = evaluate_60m_decision(candle, indicators, structure, regime, intel)

    assert res["decision"] == "SELL"
    plan = res["risk_plan"]
    assert plan["valid"] is True
    assert plan["approved"] is True
    assert plan["direction"] == "SELL"

    # Invert check for SELL: target_2 < target_1 < entry < stop_loss
    entry = plan["trigger_price"]
    stop = plan["stop_loss"]
    t1 = plan["target_1"]
    t2 = plan["target_2"]
    rr = plan["risk_reward_ratio"]

    assert t2 < t1 < entry < stop
    assert rr >= 1.5

    is_valid, err = validate_trade_plan(plan, "SELL", entry)
    assert is_valid is True
    assert err is None


def test_case_6_wait_with_no_execution_plan():
    """Case 6: WAIT with no execution plan -> risk_plan marked invalid and not approved."""
    candle, indicators, structure, regime, intel = make_mock_data(
        candle_time_et="2026-11-16 11:00:00",
        trend_type="BULLISH",
        rsi=62.0,
        near_res_dist=0.20,
    )
    res = evaluate_60m_decision(candle, indicators, structure, regime, intel)

    assert res["decision"] == "WAIT FOR CONFIRMATION"
    assert res["risk_plan"]["valid"] is False
    assert res["risk_plan"]["approved"] is False


def test_case_7_active_sell_invalidated(tmp_path: Path):
    """Case 7: Active SELL invalidated -> subsequent bar skips entry zone, becomes EXPIRED."""
    db_file = tmp_path / "test_case_7.sqlite3"
    t0_et = "2026-11-16 10:30:00"
    candle, indicators, structure, regime, intel = make_mock_data(
        candle_time_et=t0_et,
        price=250.0,
        trend_type="BEARISH",
        rsi=35.0,
        macd_hist=-0.20,
        relative_volume=1.4,
        vwap=251.5,
        near_sup_dist=3.0,
    )
    decision = evaluate_60m_decision(candle, indicators, structure, regime, intel)
    assert decision["decision"] == "SELL"

    save_60m_decision(decision, db_path=db_file)

    # Next bar gaps up way above planned entry zone (250 +- 0.1) to 252.0
    t0_utc = pd.Timestamp(t0_et, tz=ET).tz_convert("UTC")
    gap_bar = pd.DataFrame([{
        "timestamp": (t0_utc + pd.Timedelta(minutes=5)).isoformat(),
        "open": 252.5,
        "high": 253.0,
        "low": 252.0,
        "close": 252.8,
        "volume": 30000.0,
    }])

    resolve_60m_signals(gap_bar, db_path=db_file)

    # Signal should be EXPIRED (missed entry fill)
    active = get_active_60m_signal(db_path=db_file)
    assert active is None

    latest = get_latest_60m_decision(db_path=db_file)
    assert latest["status"] == "EXPIRED"


def test_case_8_active_signal_reaches_stop_loss(tmp_path: Path):
    """Case 8: Active signal reaches stop-loss -> resolved as STOPPED_OUT with negative return."""
    db_file = tmp_path / "test_case_8.sqlite3"
    t0_et = "2026-11-16 10:30:00"
    candle, indicators, structure, regime, intel = make_mock_data(
        candle_time_et=t0_et,
        price=100.0,
        trend_type="BULLISH",
        rsi=62.0,
        near_res_dist=3.0,
    )
    decision = evaluate_60m_decision(candle, indicators, structure, regime, intel)
    assert decision["decision"] == "BUY"
    stop_p = decision["risk_plan"]["stop_loss"]

    save_60m_decision(decision, db_path=db_file)

    # Subsequent bar 1 enters at 100.0; bar 2 drops to pierce stop_loss
    t0_utc = pd.Timestamp(t0_et, tz=ET).tz_convert("UTC")
    subsequent = pd.DataFrame([
        {
            "timestamp": (t0_utc + pd.Timedelta(minutes=5)).isoformat(),
            "open": 100.0,
            "high": 100.2,
            "low": 99.8,
            "close": 99.9,
            "volume": 20000.0,
        },
        {
            "timestamp": (t0_utc + pd.Timedelta(minutes=10)).isoformat(),
            "open": 99.9,
            "high": 100.0,
            "low": stop_p - 0.5,
            "close": stop_p - 0.3,
            "volume": 25000.0,
        },
    ])

    resolve_60m_signals(subsequent, db_path=db_file)

    active = get_active_60m_signal(db_path=db_file)
    assert active is None

    latest = get_latest_60m_decision(db_path=db_file)
    assert latest["status"] == "STOPPED_OUT"
    assert latest["net_return_pct"] < 0


def test_case_9_stale_market_data():
    """Case 9: Stale market data -> headline displays WAITING FOR MARKET DATA."""
    candle_ts = datetime(2026, 11, 16, 11, 0, tzinfo=ET)
    # as_of is 11:20 ET (20 minutes later during market open, > 420 seconds since candle close at 11:05)
    as_of = datetime(2026, 11, 16, 11, 20, tzinfo=ET)

    sched = compute_prediction_schedule(candle_ts, as_of=as_of)

    assert sched["is_stale"] is True
    assert sched["headline_display"] == "WAITING FOR MARKET DATA"

    # Explicit flag is_stale=True also sets headline
    sched_explicit = compute_prediction_schedule(candle_ts, as_of=datetime(2026, 11, 16, 11, 5, 20, tzinfo=ET), is_stale=True)
    assert sched_explicit["is_stale"] is True
    assert sched_explicit["headline_display"] == "WAITING FOR MARKET DATA"


def test_case_10_market_closed():
    """Case 10: Market closed -> handles session end, weekend, and next session prediction timing."""
    # Sunday evening
    sunday_ts = datetime(2026, 11, 15, 18, 0, tzinfo=ET)
    sched_weekend = compute_prediction_schedule(sunday_ts, as_of=sunday_ts)

    assert sched_weekend["is_market_open"] is False
    assert sched_weekend["market_state"] == "CLOSED"
    assert "MARKET CLOSED" in sched_weekend["headline_display"]
    assert "9:35 AM ET" in sched_weekend["headline_display"]

    # Weekday after-market (16:30 ET)
    weekday_close_ts = datetime(2026, 11, 16, 16, 30, tzinfo=ET)
    sched_closed = compute_prediction_schedule(weekday_close_ts, as_of=weekday_close_ts)

    assert sched_closed["is_market_open"] is False
    assert sched_closed["market_state"] == "CLOSED"
    assert "MARKET CLOSED" in sched_closed["headline_display"]
