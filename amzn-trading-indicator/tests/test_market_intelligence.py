from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from backend.app.services.analysis.market_intelligence import (
    _aggregate,
    analyze_market_intelligence,
)
from backend.app.services.analysis.technical_analysis import calculate_indicators
from backend.app.services.analysis.decision_engine import _evidence
from src.forecasts.calendar import ET, is_trading_day


def rth_frame(days=4, last_day_breakout=False):
    sessions = []
    day = datetime(2026, 11, 2).date()
    collected = 0
    while collected < days:
        if is_trading_day(day):
            rows = []
            price = 200 + collected * 2
            for slot in range(78):
                stamp = datetime.combine(day, time(9, 30), tzinfo=ET) + timedelta(minutes=5 * slot)
                close = price + slot * (0.05 if collected % 2 == 0 else -0.02)
                if last_day_breakout and collected == days - 1 and slot >= 3:
                    close = 207 + slot * .1
                rows.append({"timestamp": stamp.astimezone(ZoneInfo("UTC")), "open": close - .03,
                             "high": close + .08, "low": close - .08, "close": close,
                             "volume": 1000 + slot * 10 + collected * 100})
            sessions.extend(rows)
            collected += 1
        day += timedelta(days=1)
    return pd.DataFrame(sessions)


def test_15m_and_1h_aggregation_only_keeps_full_base_bar_groups():
    frame = rth_frame(1)
    aggregated = _aggregate(frame, 15)
    assert len(aggregated) == 26
    assert _aggregate(frame, 60).shape[0] == 6
    incomplete = frame.loc[frame.timestamp != frame.timestamp.iloc[2]]
    assert len(_aggregate(incomplete, 15)) == 25


def test_as_of_cutoff_excludes_forming_or_future_candles():
    frame = rth_frame(2)
    last = frame.iloc[-1].timestamp
    future = {**frame.iloc[-1].to_dict(), "timestamp": last + pd.Timedelta(minutes=5), "close": 999}
    extended = pd.concat([frame, pd.DataFrame([future])], ignore_index=True)
    result = analyze_market_intelligence(extended, as_of=last + pd.Timedelta(minutes=5))
    assert result["latest_completed_candle"] == last.isoformat()
    assert result["trend_alignment"]["timeframes"][2]["close"] == float(frame.iloc[-1].close)


def test_orb_levels_and_breakout_use_completed_opening_range_and_closes():
    frame = rth_frame(2, last_day_breakout=True)
    result = analyze_market_intelligence(frame, as_of=frame.timestamp.iloc[-1] + pd.Timedelta(minutes=5))
    orb = result["opening_range"]
    assert orb["status"] == "READY"
    assert orb["high"] > orb["low"]
    assert orb["classification"] in {"BULLISH_BREAKOUT", "BULLISH_BREAKOUT_RETEST"}
    assert orb["confirmation"] == "completed 5-minute candle close"


def test_orb_retest_and_failed_breakout_classifications():
    frame = rth_frame(2)
    last_day = frame.timestamp.dt.tz_convert(ET).dt.date.max()
    day_mask = frame.timestamp.dt.tz_convert(ET).dt.date == last_day
    current = frame.index[day_mask]
    opening_high = frame.loc[current[:3], "high"].max()
    previous_index, latest_index = current[-2], current[-1]
    frame.loc[previous_index, ["open", "close"]] = opening_high + .25
    frame.loc[previous_index, "high"] = opening_high + .4
    frame.loc[previous_index, "low"] = opening_high + .1
    frame.loc[latest_index, ["open", "close"]] = opening_high - .1
    frame.loc[latest_index, "high"] = opening_high + .05
    frame.loc[latest_index, "low"] = opening_high - .2
    failed = analyze_market_intelligence(frame, as_of=frame.timestamp.iloc[-1] + pd.Timedelta(minutes=5))
    assert failed["opening_range"]["classification"] == "FAILED_BULLISH_BREAKOUT"

    frame.loc[latest_index, ["open", "close"]] = opening_high + .2
    frame.loc[latest_index, "high"] = opening_high + .3
    frame.loc[latest_index, "low"] = opening_high - .1
    retest = analyze_market_intelligence(frame, as_of=frame.timestamp.iloc[-1] + pd.Timedelta(minutes=5))
    assert retest["opening_range"]["classification"] == "BULLISH_BREAKOUT_RETEST"


def test_rvol_uses_same_time_slot_and_reports_sample_count():
    frame = rth_frame(4)
    result = analyze_market_intelligence(frame, as_of=frame.timestamp.iloc[-1] + pd.Timedelta(minutes=5))
    rvol = result["rvol"]
    assert rvol["status"] == "READY"
    assert rvol["sample_count"] == 3
    assert rvol["slot_et"] == "15:55"
    assert rvol["rvol"] > 0


def test_market_intelligence_and_decision_volume_metrics_are_distinct_and_auditable():
    frame = rth_frame(4)
    frame["volume"] = frame.volume.astype(float)
    # Hold the final day's previous same-session bars near 829.47 shares, while
    # previous sessions have 10,600 shares at this exact ET slot.
    latest_day = frame.timestamp.dt.tz_convert(ET).dt.date.max()
    latest_indices = frame.index[frame.timestamp.dt.tz_convert(ET).dt.date == latest_day]
    frame.loc[latest_indices[-20:-1], "volume"] = 829.4736842105264
    frame.loc[latest_indices[-1], "volume"] = 4240
    prior_days = frame.index[frame.timestamp.dt.tz_convert(ET).dt.date < latest_day]
    for index in prior_days:
        if frame.loc[index, "timestamp"].tz_convert(ET).strftime("%H:%M") == "15:55":
            frame.loc[index, "volume"] = 10600

    timestamp = frame.timestamp.iloc[-1]
    intelligence = analyze_market_intelligence(frame, as_of=timestamp + pd.Timedelta(minutes=5))
    indicators = calculate_indicators(frame)
    current = indicators.iloc[-1]
    volume_reason = next(item for group in _evidence(indicators, {}, {}, {"available": False})[1:] for item in group
                         if "Session rolling volume ratio" in item)

    assert intelligence["latest_completed_candle"] == timestamp.isoformat()
    assert intelligence["rvol"]["candle_timestamp"] == timestamp.isoformat()
    assert intelligence["rvol"]["current_volume"] == 4240
    assert intelligence["rvol"]["average_same_slot_volume"] == 10600
    assert intelligence["rvol"]["sample_count"] == 3
    assert intelligence["rvol"]["rvol"] == pytest.approx(0.4)
    assert current.timestamp == timestamp
    assert current.volume == 4240
    assert current.volume_sma_20 == pytest.approx(1000)
    assert current.volume_observation_count_20 == 20
    assert current.relative_volume == pytest.approx(4.24)
    assert "4,240" in volume_reason and "1,000.00" in volume_reason and "20 observed bars" in volume_reason
    assert "same-session trailing 20-bar mean" in volume_reason


def test_mixed_timeframe_alignment_remains_mixed_with_explicit_conflicts(monkeypatch):
    import backend.app.services.analysis.market_intelligence as intelligence_module

    trends = {"1H": "BEARISH", "15M": "NEUTRAL", "5M": "BULLISH"}

    def fixed_trend(_frame, label):
        return {"timeframe": label, "trend": trends[label], "status": "READY", "bars": 30}

    monkeypatch.setattr(intelligence_module, "_trend", fixed_trend)
    frame = rth_frame(4, last_day_breakout=True)
    result = analyze_market_intelligence(frame, as_of=frame.timestamp.iloc[-1] + pd.Timedelta(minutes=5))
    alignment = result["trend_alignment"]
    assert [item["trend"] for item in alignment["timeframes"]] == ["BEARISH", "NEUTRAL", "BULLISH"]
    assert alignment["overall"] == "MIXED"
    assert "1H BEARISH, 15M NEUTRAL, 5M BULLISH" in alignment["explanation"]
    assert alignment["conflict_warning"] is True
    indicators = calculate_indicators(frame)
    assert indicators.iloc[-1].trend_classification == "STRONG_BULLISH"
    evidence_text = " ".join(item for group in _evidence(indicators, {}, {}, {"available": False})[1:] for item in group)
    assert alignment["overall"] == "MIXED"
    assert "Moving average structure is STRONG_BULLISH" in evidence_text


def test_decision_evidence_names_ema_structure_separately_from_mtf_alignment():
    frame = calculate_indicators(rth_frame(4))
    evidence = _evidence(frame, {}, {}, {"available": False})
    text = [item for group in evidence[1:] for item in group]
    trend_explanation = next(item for item in text if "Moving average structure" in item)
    assert "EMA 9/20/50" in trend_explanation
    assert "separate from 1H/15M/5M trend alignment" in trend_explanation
    assert "Trend alignment is" not in trend_explanation


def test_previous_day_levels_exclude_out_of_session_candles_and_find_exact_prior_session():
    frame = rth_frame(2)
    extra = frame.iloc[-1].copy()
    extra["timestamp"] = extra.timestamp + pd.Timedelta(hours=2)
    extra["high"] = 500
    frame = pd.concat([frame, pd.DataFrame([extra])], ignore_index=True)
    result = analyze_market_intelligence(frame, as_of=extra.timestamp + pd.Timedelta(minutes=5))
    levels = result["previous_day_levels"]
    assert levels["status"] == "READY"
    assert levels["session_date"] == frame.timestamp.iloc[0].tz_convert(ET).date().isoformat()
    assert levels["high"] < 500


def test_insufficient_history_is_explicit_for_trend_rvol_and_orb():
    frame = rth_frame(1)
    result = analyze_market_intelligence(frame.iloc[:2], as_of=frame.timestamp.iloc[1] + pd.Timedelta(minutes=5))
    assert result["trend_alignment"]["timeframes"][0]["status"] == "INSUFFICIENT_DATA"
    assert result["rvol"]["status"] == "INSUFFICIENT_DATA"
    assert result["opening_range"]["status"] == "INSUFFICIENT_DATA"
