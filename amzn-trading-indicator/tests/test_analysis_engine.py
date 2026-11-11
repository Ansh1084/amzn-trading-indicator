from datetime import datetime, timezone

import pandas as pd
import pytest

from backend.app.services.analysis.service import analyze_candles, validate_completed_candles, _expected_latest_start
from backend.app.services.analysis.persistence import save_signal, list_signals, active_signals
from backend.app.services.analysis.outcomes import resolve_signals
from backend.app.services.analysis.market_structure import _crossing_events
from backend.app.services.analysis.market_structure import confirmed_pivots
from backend.app.services.analysis.risk_management import build_risk_plan


def candles(n=90, day="2026-11-09"):
    stamps = pd.date_range(f"{day} 09:30", periods=n, freq="5min", tz="America/New_York").tz_convert("UTC")
    close = [250 + i * .025 + (i % 7) * .03 for i in range(n)]
    return pd.DataFrame({"timestamp": stamps, "open": [x-.02 for x in close],
                         "high": [x+.08 for x in close], "low": [x-.08 for x in close],
                         "close": close, "volume": [1000+i*2 for i in range(n)],
                         "vwap": close})


def test_validation_excludes_incomplete_and_rejects_duplicates():
    frame = candles(3)
    cutoff = frame.timestamp.iloc[-1] + pd.Timedelta(minutes=4)
    result = validate_completed_candles(frame, cutoff.to_pydatetime(), stale_check=False)
    assert len(result) == 2
    frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_completed_candles(frame, datetime(2026, 11, 10, tzinfo=timezone.utc), stale_check=False)


def test_stale_previous_session_snapshot_is_rejected():
    frame = candles(78, "2026-11-06")
    monday_open = datetime(2026, 11, 9, 9, 45, tzinfo=__import__("zoneinfo").ZoneInfo("America/New_York"))
    with pytest.raises(ValueError, match="stale"):
        validate_completed_candles(frame, monday_open)


def test_expected_candle_calendar_weekend_and_early_close():
    # Monday pre-open uses Friday's 15:55 ET candle.
    preopen = datetime(2026, 11, 9, 9, 0, tzinfo=__import__("zoneinfo").ZoneInfo("America/New_York"))
    assert _expected_latest_start(preopen).isoformat() == "2026-11-06T15:55:00-05:00"
    # Saturday also resolves to Friday, not Saturday.
    saturday = datetime(2026, 11, 7, 12, 0, tzinfo=__import__("zoneinfo").ZoneInfo("America/New_York"))
    assert _expected_latest_start(saturday).isoformat() == "2026-11-06T15:55:00-05:00"
    # Black Friday early close has its final 5-minute bar at 12:55 ET.
    friday_close = datetime(2026, 11, 27, 14, 0, tzinfo=__import__("zoneinfo").ZoneInfo("America/New_York"))
    assert _expected_latest_start(friday_close).isoformat() == "2026-11-27T12:55:00-05:00"


def test_breakout_and_breakdown_use_crossed_levels_even_when_now_behind_price():
    resistance = {"lower": 100.0, "upper": 100.2, "midpoint": 100.1}
    support = {"lower": 99.8, "upper": 100.0, "midpoint": 99.9}
    breakout, breakdown = _crossing_events([resistance], [support], 100.1, 100.5, 1.0, 1.5)
    assert breakout["status"] == "CONFIRMED_BREAKOUT"
    breakout, breakdown = _crossing_events([resistance], [support], 99.9, 99.4, 1.0, 1.5)
    assert breakdown["status"] == "CONFIRMED_BREAKDOWN"
    old_breakout, _ = _crossing_events([resistance], [support], 101.0, 99.0, 1.0, 1.5, [102, 101, 99])
    assert old_breakout["status"] == "NONE"
    recent_failure, _ = _crossing_events([resistance], [support], 101.0, 99.0, 1.0, 1.5, [99, 100.3, 101, 99])
    assert recent_failure["status"] == "FAILED_BREAKOUT"


def test_pivot_is_not_visible_until_right_side_bars_complete():
    frame = candles(6)
    frame["high"] = [10, 11, 12, 15, 12, 11]
    frame["low"] = [8, 9, 10, 11, 9, 8]
    assert not any(p["index"] == 3 for p in confirmed_pivots(frame.iloc[:5]))
    pivot = next(p for p in confirmed_pivots(frame) if p["index"] == 3 and p["type"] == "HIGH")
    assert pivot["confirmed_at"] == frame.timestamp.iloc[5].isoformat()


def test_risk_plan_levels_are_directionally_consistent():
    long = build_risk_plan("BUY", 100, 2, {"support_zones": [], "resistance_zones": []})
    short = build_risk_plan("SELL", 100, 2, {"support_zones": [], "resistance_zones": []})
    assert long["valid"] and long["stop_loss"] < 100 < long["target_1"] <= long["target_2"]
    assert short["valid"] and short["target_2"] <= short["target_1"] < 100 < short["stop_loss"]


def test_analysis_persists_once_and_keeps_ml_unavailable_honest(tmp_path):
    path = tmp_path / "analysis.sqlite"
    data = candles(78)
    result = analyze_candles(data, (data.timestamp.iloc[-1] + pd.Timedelta(minutes=5)).to_pydatetime(),
                             persist=True, db_path=path, ml_evidence={"available": False, "reason": "test missing artifact"})
    assert result["analysis"]["ml_evidence"]["available"] is False
    repeat = analyze_candles(data, (data.timestamp.iloc[-1] + pd.Timedelta(minutes=5)).to_pydatetime(),
                             persist=True, db_path=path, ml_evidence={"available": False, "reason": "test missing artifact"})
    assert result["signal_record"]["signal_id"] == repeat["signal_record"]["signal_id"]
    assert len(list_signals(db_path=path)) == 1
    assert list_signals(db_path=path)[0]["payload"]["hypothetical_only"] is True
    assert result["signal_record"]["payload"]["generated_at"] == result["signal_record"]["payload"]["candle_close_timestamp"]


def test_hold_is_not_a_new_active_position(tmp_path):
    path=tmp_path/"hold.sqlite"
    stamp=pd.Timestamp("2026-11-09 14:00",tz="America/New_York").tz_convert("UTC").isoformat()
    payload={"analysis_version":"amzn-analysis-v1","candle_timestamp":stamp,"generated_at":stamp,
      "classification":"HOLD","directional_bias":"BULLISH","risk_plan":{"valid":True,"direction":"BUY"}}
    record=save_signal(payload,path)
    assert record["status"]=="NOT_ACTIONABLE"
    assert active_signals(path)==[]


def test_completed_bar_can_resolve_hypothetical_stop_once(tmp_path):
    path = tmp_path / "outcome.sqlite"
    stamp = pd.Timestamp("2026-11-09 15:00", tz="America/New_York").tz_convert("UTC")
    payload = {"analysis_version":"amzn-analysis-v1", "candle_timestamp":stamp.isoformat(),
      "generated_at":datetime.now(timezone.utc).isoformat(), "classification":"BUY", "directional_bias":"BULLISH",
      "final_decision_score":.5, "risk_plan":{"valid":True,"direction":"BUY","trigger_price":100,
      "entry_zone":{"lower":99.9,"upper":100.1},"stop_loss":99,"target_1":101,"target_2":102}}
    row=save_signal(payload,path)
    future=pd.DataFrame({"timestamp":[stamp+pd.Timedelta(minutes=5)],"open":[100],"high":[100.5],
                         "low":[98.8],"close":[99.2],"volume":[100]})
    result=resolve_signals(future,(stamp+pd.Timedelta(minutes=10)).to_pydatetime(),path)
    assert result["resolved_signal_ids"]==[row["signal_id"]]
    stored=list_signals(db_path=path)[0]
    assert stored["status"]=="STOP_LOSS_HIT"
    assert stored["outcome"]["gross_return_pct"]<0
    assert resolve_signals(future,(stamp+pd.Timedelta(minutes=10)).to_pydatetime(),path)["resolved_signal_ids"]==[]


def test_target_one_event_is_idempotent_and_unresolved_has_no_pnl(tmp_path):
    path=tmp_path/"target-one.sqlite"
    stamp=pd.Timestamp("2026-11-09 14:00",tz="America/New_York").tz_convert("UTC")
    payload={"analysis_version":"amzn-analysis-v1","candle_timestamp":stamp.isoformat(),"generated_at":stamp.isoformat(),
      "classification":"BUY","directional_bias":"BULLISH","risk_plan":{"valid":True,"direction":"BUY","trigger_price":100,
      "entry_zone":{"lower":99.9,"upper":100.1},"stop_loss":99,"target_1":101,"target_2":102}}
    saved=save_signal(payload,path)
    future=pd.DataFrame({"timestamp":[stamp+pd.Timedelta(minutes=5)],"open":[100],"high":[101.2],"low":[99.95],"close":[101.1],"volume":[10]})
    resolve_signals(future,(stamp+pd.Timedelta(minutes=10)).to_pydatetime(),path)
    resolve_signals(future,(stamp+pd.Timedelta(minutes=10)).to_pydatetime(),path)
    stored=list_signals(db_path=path)[0]
    assert stored["status"]=="ACTIVE"
    assert stored["gross_return_pct"] is None and stored["net_return_pct"] is None
    from backend.app.services.analysis.persistence import signal_events
    assert [e["event_type"] for e in signal_events(saved["signal_id"],path)].count("TARGET_1_HIT")==1


def test_open_outside_entry_zone_expires_without_return(tmp_path):
    path=tmp_path/"expired.sqlite"
    stamp=pd.Timestamp("2026-11-09 14:00",tz="America/New_York").tz_convert("UTC")
    payload={"analysis_version":"amzn-analysis-v1","candle_timestamp":stamp.isoformat(),"generated_at":stamp.isoformat(),
      "classification":"SELL","directional_bias":"BEARISH","risk_plan":{"valid":True,"direction":"SELL","trigger_price":100,
      "entry_zone":{"lower":99.9,"upper":100.1},"stop_loss":101,"target_1":99,"target_2":98}}
    save_signal(payload,path)
    future=pd.DataFrame({"timestamp":[stamp+pd.Timedelta(minutes=5)],"open":[100.5],"high":[100.7],"low":[100.3],"close":[100.6],"volume":[10]})
    resolve_signals(future,(stamp+pd.Timedelta(minutes=10)).to_pydatetime(),path)
    stored=list_signals(db_path=path)[0]
    assert stored["status"]=="EXPIRED"
    assert stored["gross_return_pct"] is None and stored["net_return_pct"] is None


def test_stop_has_priority_over_target_two_and_does_not_count_target_one(tmp_path):
    path=tmp_path/"ambiguous.sqlite"
    stamp=pd.Timestamp("2026-11-09 14:00",tz="America/New_York").tz_convert("UTC")
    payload={"analysis_version":"amzn-analysis-v1","candle_timestamp":stamp.isoformat(),"generated_at":stamp.isoformat(),
      "classification":"BUY","directional_bias":"BULLISH","risk_plan":{"valid":True,"direction":"BUY","trigger_price":100,
      "entry_zone":{"lower":99.9,"upper":100.1},"stop_loss":99,"target_1":101,"target_2":102}}
    saved=save_signal(payload,path)
    future=pd.DataFrame({"timestamp":[stamp+pd.Timedelta(minutes=5)],"open":[100],"high":[102.2],"low":[98.8],"close":[100],"volume":[5]})
    resolve_signals(future,(stamp+pd.Timedelta(minutes=10)).to_pydatetime(),path)
    stored=list_signals(db_path=path)[0]
    assert stored["status"]=="STOP_LOSS_HIT"
    from backend.app.services.analysis.persistence import signal_events
    assert [e["event_type"] for e in signal_events(saved["signal_id"],path)].count("TARGET_1_HIT")==0


def test_short_stop_loss_return_uses_short_direction(tmp_path):
    path=tmp_path/"short.sqlite"
    stamp=pd.Timestamp("2026-11-09 14:00",tz="America/New_York").tz_convert("UTC")
    payload={"analysis_version":"amzn-analysis-v1","candle_timestamp":stamp.isoformat(),"generated_at":stamp.isoformat(),
      "classification":"SELL","directional_bias":"BEARISH","risk_plan":{"valid":True,"direction":"SELL","trigger_price":100,
      "entry_zone":{"lower":99.9,"upper":100.1},"stop_loss":101,"target_1":99,"target_2":98}}
    save_signal(payload,path)
    future=pd.DataFrame({"timestamp":[stamp+pd.Timedelta(minutes=5)],"open":[100],"high":[101.2],"low":[99.5],"close":[100.5],"volume":[5]})
    resolve_signals(future,(stamp+pd.Timedelta(minutes=10)).to_pydatetime(),path)
    stored=list_signals(db_path=path)[0]
    assert stored["status"]=="STOP_LOSS_HIT"
    assert stored["outcome"]["gross_return_pct"]<0


def test_open_intraday_position_does_not_exit_on_next_session_prices(tmp_path):
    path=tmp_path/"overnight.sqlite"
    stamp=pd.Timestamp("2026-11-09 15:45",tz="America/New_York").tz_convert("UTC")
    payload={"analysis_version":"amzn-analysis-v1","candle_timestamp":stamp.isoformat(),"generated_at":stamp.isoformat(),
      "classification":"BUY","directional_bias":"BULLISH","risk_plan":{"valid":True,"direction":"BUY","trigger_price":100,
      "entry_zone":{"lower":99.9,"upper":100.1},"stop_loss":99,"target_1":101,"target_2":102}}
    saved=save_signal(payload,path)
    future=pd.DataFrame({"timestamp":[stamp+pd.Timedelta(minutes=5),pd.Timestamp("2026-11-10 09:30",tz="America/New_York").tz_convert("UTC")],
      "open":[100,101],"high":[100.2,103],"low":[99.95,100.5],"close":[100.1,102],"volume":[10,20]})
    result=resolve_signals(future,pd.Timestamp("2026-11-10 09:35",tz="America/New_York").to_pydatetime(),path)
    stored=list_signals(db_path=path)[0]
    assert saved["signal_id"] in result["pending_signal_ids"]
    assert stored["status"]=="ACTIVE" and stored["gross_return_pct"] is None
