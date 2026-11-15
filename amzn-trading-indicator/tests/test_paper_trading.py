from datetime import datetime, timezone

import pandas as pd
import pytest

from backend.app.services import paper_trading as paper
from backend.app.api import routes


def signal(signal_id="s1", side="BUY", ts="2026-11-09T14:30:00+00:00"):
    is_long = side == "BUY"
    return {"signal_id": signal_id, "candle_timestamp": ts, "generated_at": "2026-11-09T14:35:00+00:00",
            "classification": side, "status": "ACTIVE", "payload": {
                "classification": side, "candle_close_timestamp": "2026-11-09T14:35:00+00:00",
                "market_regime": {"classification": "TRENDING"},
                "risk_plan": {"valid": True, "trigger_price": 100.0,
                    "entry_zone": {"lower": 99.9, "upper": 100.1},
                    "stop_loss": 99.0 if is_long else 101.0,
                    "target_1": 101.0 if is_long else 99.0,
                    "target_2": 102.0 if is_long else 98.0}}}


def candle(ts, open_=100, high=100.5, low=99.5, close=100.2, volume=100):
    return {"timestamp": pd.Timestamp(ts), "open": open_, "high": high, "low": low,
            "close": close, "volume": volume}


def enabled(path, **settings):
    paper.update_settings(enabled=True, transaction_cost_bps=10, slippage_bps=20, path=path, **settings)


def enter_trade(path, side="BUY", signal_id="s1", entry_candle=None):
    enabled(path, sizing_mode="fixed_shares", sizing_value=2)
    queued = paper.on_actionable_signal(signal(signal_id, side), path)
    assert queued and queued["status"] == "PENDING_ENTRY"
    row = entry_candle or candle("2026-11-09T14:35:00+00:00")
    assert paper.process_completed_candles([row], path) == 1
    return paper.dashboard(path=path)["open_trades"][0]


def test_paper_trading_starts_disabled_and_only_queues_actionable_fresh_signals(tmp_path):
    path = tmp_path / "paper.sqlite3"
    assert paper.dashboard(path=path)["settings"]["enabled"] is False
    assert paper.on_actionable_signal(signal(), path) is None
    enabled(path)
    wait = signal(); wait["classification"] = "WAIT FOR CONFIRMATION"; wait["payload"]["classification"] = wait["classification"]
    assert paper.on_actionable_signal(wait, path) is None
    stale = signal("stale"); stale["generated_at"] = "2026-11-09T15:00:00+00:00"
    assert paper.on_actionable_signal(stale, path) is None


def test_duplicate_signal_is_idempotent_and_entry_uses_next_bar_open(tmp_path):
    path = tmp_path / "paper.sqlite3"
    enabled(path)
    record = signal()
    queued = paper.on_actionable_signal(record, path)
    assert paper.on_actionable_signal(record, path) is None
    before = paper.process_completed_candles([candle("2026-11-09T14:30:00+00:00")], path)
    assert before == 0
    next_bar = candle("2026-11-09T14:35:00+00:00", open_=100.05)
    assert paper.process_completed_candles([next_bar], path) == 1
    trade = paper.dashboard(path=path)["open_trades"][0]
    assert trade["trade_id"] == queued["trade_id"]
    assert trade["entry_timestamp"] == "2026-11-09T14:35:00+00:00"
    assert trade["entry_price"] == next_bar["open"]
    assert trade["entry_fill_price"] > trade["entry_price"]  # adverse long slippage


def test_next_open_outside_entry_zone_expires_without_fabricated_fill(tmp_path):
    path = tmp_path / "paper.sqlite3"
    enabled(path)
    paper.on_actionable_signal(signal(), path)
    paper.process_completed_candles([candle("2026-11-09T14:35:00+00:00", open_=100.5)], path)
    result = paper.dashboard(path=path)
    assert result["pending_trades"] == []
    assert result["metrics"]["open_count"] == 0
    assert result["closed_trades"] == []


def test_missing_immediate_next_candle_does_not_use_a_later_open(tmp_path):
    path = tmp_path / "paper.sqlite3"
    enabled(path)
    paper.on_actionable_signal(signal(), path)
    paper.process_completed_candles([candle("2026-11-09T14:40:00+00:00")], path)
    result = paper.dashboard(path=path)
    assert result["pending_trades"] == []
    assert result["open_trades"] == []
    assert result["journal"][0]["event_type"] == "EXPIRED_NO_FILL"


def test_long_and_short_net_pnl_include_transaction_costs_and_adverse_slippage(tmp_path):
    for side, exit_price, expected_direction in [("BUY", 102, "LONG"), ("SELL", 98, "SHORT")]:
        path = tmp_path / f"{side}.sqlite3"
        open_trade = enter_trade(path, side)
        assert open_trade["direction"] == expected_direction
        closed = paper.close_trade(open_trade["trade_id"], exit_price, "2026-11-09T14:40:00+00:00", path)
        assert closed["status"] == "CLOSED"
        assert closed["gross_pnl"] > 0
        assert closed["gross_pnl"] == pytest.approx((closed["exit_price"] - closed["entry_price"]) * closed["quantity"] * (1 if side == "BUY" else -1))
        assert closed["transaction_costs"] > 0 and closed["slippage_cost"] > 0
        assert closed["net_pnl"] == pytest.approx(closed["gross_pnl"] - closed["transaction_costs"] - closed["slippage_cost"])
        assert closed["r_multiple"] > 0


def test_stop_executes_first_when_stop_and_target_are_touched_in_same_bar(tmp_path):
    path = tmp_path / "paper.sqlite3"
    enter_trade(path)
    collision = candle("2026-11-09T14:40:00+00:00", open_=100, high=102.5, low=98.5, close=100.4)
    assert paper.process_completed_candles([collision], path) == 1
    closed = paper.dashboard(path=path)["closed_trades"][0]
    assert closed["exit_reason"] == "STOP_LOSS"
    assert closed["exit_price"] == 99
    assert closed["exit_fill_price"] < 99  # adverse slippage is tracked separately


def test_target_one_partial_is_recorded_once_and_target_two_closes_remainder(tmp_path):
    path = tmp_path / "paper.sqlite3"
    enter_trade(path)
    t1 = candle("2026-11-09T14:40:00+00:00", high=101.2)
    assert paper.process_completed_candles([t1], path) == 1
    assert paper.process_completed_candles([t1], path) == 0
    opened = paper.dashboard(path=path)["open_trades"][0]
    assert opened["remaining_quantity"] == pytest.approx(1)
    t2 = candle("2026-11-09T14:45:00+00:00", high=102.2)
    assert paper.process_completed_candles([t2], path) == 1
    closed = paper.dashboard(path=path)["closed_trades"][0]
    assert closed["exit_reason"] == "TARGET_2" and closed["remaining_quantity"] == 0


def test_account_persists_and_reports_distinct_realized_and_unrealized_pnl(tmp_path):
    path = tmp_path / "paper.sqlite3"
    opened = enter_trade(path)
    first = paper.dashboard(mark_price=101, mark_timestamp="2026-11-09T14:40:00+00:00", path=path)
    assert first["metrics"]["unrealized_pnl"] is not None
    assert first["metrics"]["realized_pnl"] == 0
    reopened = paper.dashboard(mark_price=101, mark_timestamp="2026-11-09T14:40:00+00:00", path=path)
    assert reopened["open_trades"][0]["trade_id"] == opened["trade_id"]
    assert reopened["disclaimer"].startswith("PAPER TRADING")


def test_pause_cancels_queued_entries_but_keeps_open_trade_exit_management(tmp_path):
    path = tmp_path / "paper.sqlite3"
    trade = enter_trade(path)
    paper.update_settings(enabled=False, path=path)
    stop_bar = candle("2026-11-09T14:40:00+00:00", low=98.5, high=100.5)
    assert paper.process_completed_candles([stop_bar], path) == 1
    dashboard = paper.dashboard(path=path)
    assert dashboard["closed_trades"][0]["exit_reason"] == "STOP_LOSS"
    assert dashboard["settings"]["enabled"] is False


def test_incomplete_or_non_session_candles_are_not_processed(tmp_path):
    path = tmp_path / "paper.sqlite3"
    enabled(path)
    paper.on_actionable_signal(signal(), path)
    incomplete = candle("2026-11-09T14:35:00+00:00")
    assert paper.process_completed_candles([incomplete], path, as_of="2026-11-09T14:39:59+00:00") == 0
    weekend = candle("2026-11-14T14:35:00+00:00")
    assert paper.process_completed_candles([weekend], path, as_of="2026-11-14T14:40:00+00:00") == 0
    assert paper.dashboard(path=path)["pending_trades"][0]["status"] == "PENDING_ENTRY"


def test_invalid_directional_stop_plan_is_not_queued(tmp_path):
    path = tmp_path / "paper.sqlite3"
    enabled(path)
    record = signal()
    record["payload"]["risk_plan"]["stop_loss"] = 101
    assert paper.on_actionable_signal(record, path) is None


def test_historical_replay_uses_isolated_research_ledger():
    bars = pd.DataFrame([
        candle("2026-11-09T14:30:00+00:00"),
        candle("2026-11-09T14:35:00+00:00", open_=100, high=102.5, low=98.5, close=100.1),
    ])
    replay = paper.replay_historical(bars, [signal()], starting_capital=5_000,
                                     transaction_cost_bps=10, slippage_bps=20)
    assert replay["mode"] == "HISTORICAL_RESEARCH_ONLY"
    assert replay["signal_count"] == replay["actionable_signals"] == replay["trade_count"] == 1
    assert replay["metrics"]["realized_pnl"] < 0
    assert replay["metrics"]["total_trades"] == 1
    assert replay["limitations"]


def test_paper_dashboard_and_settings_routes_serialize_real_empty_account(monkeypatch, tmp_path):
    monkeypatch.setattr(paper, "DB_PATH", tmp_path / "route-paper.sqlite3")
    monkeypatch.setattr(routes.market_store, "get_candles", lambda **_kwargs: [])
    monkeypatch.setattr(routes, "_market_stream_status", lambda: {"freshness": "UNAVAILABLE", "market_state": "CLOSED"})
    result = routes.paper_dashboard()
    assert result["settings"]["enabled"] is False
    assert result["mark"]["price"] is None
    assert result["mark"]["freshness"] == "UNAVAILABLE"
    import json
    json.dumps(result, allow_nan=False)
    settings = routes.paper_settings(routes.PaperSettingsRequest(enabled=True))
    assert settings["enabled"] is True


def test_market_intelligence_route_uses_persisted_candles(monkeypatch):
    frame = pd.DataFrame([candle((datetime(2026, 11, 9, 14, 30, tzinfo=timezone.utc) + pd.Timedelta(minutes=5 * i)).isoformat()) for i in range(30)])
    monkeypatch.setattr(routes.market_store, "get_candles", lambda **_kwargs: frame.to_dict("records"))
    result = routes.market_intelligence()
    assert result["latest_completed_candle"] is not None
    assert result["decision_weight_changed"] is False
