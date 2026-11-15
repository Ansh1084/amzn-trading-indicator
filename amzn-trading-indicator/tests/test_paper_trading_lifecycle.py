from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest

from backend.app.services import paper_trading as paper


def _candle(timestamp: str, *, open_: float, high: float, low: float, close: float, volume: int):
    return {"timestamp": pd.Timestamp(timestamp), "open": open_, "high": high,
            "low": low, "close": close, "volume": volume}


def test_one_isolated_full_paper_trade_lifecycle_survives_restart_and_duplicate_replay(tmp_path, monkeypatch):
    db_path = (tmp_path / "paper-trading-lifecycle.sqlite3").resolve()
    production_path = Path(paper.DB_PATH).resolve()
    assert db_path != production_path, "Lifecycle test must never use the production paper ledger"

    real_connect = paper._connect

    @contextmanager
    def isolated_connect(path=None):
        assert path is not None, "Every paper database operation must provide an explicit database path"
        assert Path(path).resolve() == db_path, f"Unexpected paper database path: {path}"
        with real_connect(path) as connection:
            yield connection

    monkeypatch.setattr(paper, "_connect", isolated_connect)

    settings = paper.update_settings(enabled=True, starting_capital=100_000,
        sizing_mode="fixed_shares", sizing_value=1,
        transaction_cost_bps=1, slippage_bps=2, path=db_path)
    assert settings["enabled"] is True
    assert settings["starting_capital"] == 100_000
    assert settings["sizing_value"] == 1
    assert settings["transaction_cost_bps"] == 1
    assert settings["slippage_bps"] == 2

    signal_id = "paper-lifecycle-amzn-buy-20261109-0930"
    signal = {
        "signal_id": signal_id,
        "analysis_version": "amzn-analysis-lifecycle-test-v1",
        "candle_timestamp": "2026-11-09T14:30:00+00:00",
        "generated_at": "2026-11-09T14:35:00+00:00",
        "classification": "BUY",
        "directional_bias": "BULLISH",
        "status": "ACTIVE",
        "payload": {
            "classification": "BUY",
            "candle_close_timestamp": "2026-11-09T14:35:00+00:00",
            "setup_type": "LIFECYCLE_TEST",
            "risk_plan": {
                "valid": True, "direction": "BUY", "trigger_price": 100.00,
                "entry_zone": {"lower": 100.00, "upper": 101.00},
                "stop_loss": 98.00, "target_1": 103.00, "target_2": 104.00,
            },
        },
    }
    queued = paper.on_actionable_signal(signal, path=db_path)
    assert queued is not None, "Valid fresh BUY signal should be queued when paper trading is enabled"
    assert queued["status"] == "PENDING_ENTRY"
    assert queued["signal_id"] == signal_id
    assert queued["quantity"] == 1

    signal_candle = _candle("2026-11-09T14:30:00+00:00", open_=100.0, high=100.8,
                            low=99.8, close=100.4, volume=1000)
    entry_candle = _candle("2026-11-09T14:35:00+00:00", open_=100.5, high=101.0,
                           low=100.25, close=100.75, volume=900)
    # Gap through the $98 stop: the engine must use the worse candle open ($97.50).
    stop_candle = _candle("2026-11-09T14:40:00+00:00", open_=97.5, high=99.0,
                          low=97.25, close=98.5, volume=1200)
    candles = [signal_candle, entry_candle, stop_candle]

    # At 09:39:59 ET the 09:35 candle is still forming; it cannot fill the trade.
    assert paper.process_completed_candles(candles[:2], path=db_path,
        as_of="2026-11-09T14:39:59+00:00") == 0
    pending = paper.dashboard(path=db_path)
    assert len(pending["pending_trades"]) == 1
    assert pending["open_trades"] == []
    assert pending["pending_trades"][0]["entry_timestamp"] is None

    # At the 09:40 ET boundary, only the next completed candle can create entry.
    assert paper.process_completed_candles(candles[:2], path=db_path,
        as_of="2026-11-09T14:40:00+00:00") == 1
    opened = paper.dashboard(path=db_path)
    assert len(opened["open_trades"]) == 1
    assert opened["pending_trades"] == []
    trade = opened["open_trades"][0]
    assert trade["entry_timestamp"] == "2026-11-09T14:35:00+00:00"
    assert trade["entry_price"] == pytest.approx(100.50)
    assert trade["entry_fill_price"] == pytest.approx(100.50 * 1.0002)
    assert trade["quantity"] == pytest.approx(1)
    assert trade["exit_timestamp"] is None

    # The 09:40 stop candle is also excluded until its 09:45 close boundary.
    assert paper.process_completed_candles(candles, path=db_path,
        as_of="2026-11-09T14:44:59+00:00") == 0
    before_stop_close = paper.dashboard(path=db_path)
    assert before_stop_close["open_trades"][0]["trade_id"] == trade["trade_id"]
    assert before_stop_close["closed_trades"] == []

    assert paper.process_completed_candles(candles, path=db_path,
        as_of="2026-11-09T14:45:00+00:00") == 1
    closed_dashboard = paper.dashboard(path=db_path)
    assert closed_dashboard["open_trades"] == []
    assert closed_dashboard["pending_trades"] == []
    assert len(closed_dashboard["closed_trades"]) == 1
    closed = closed_dashboard["closed_trades"][0]
    assert closed["status"] == "CLOSED"
    assert closed["exit_reason"] == "STOP_LOSS"
    assert closed["exit_timestamp"] == "2026-11-09T14:40:00+00:00"
    assert closed["exit_price"] == pytest.approx(97.50), "Gap-through-stop fill must use worse candle open"
    assert closed["exit_fill_price"] == pytest.approx(97.50 * (1 - 0.0002))

    # Independently calculate from persisted raw references and configured bps.
    expected_gross = (97.50 - 100.50) * 1
    expected_entry_fee = (100.50 * 1.0002) * 1 * (1 / 10_000)
    expected_exit_fee = (97.50 * (1 - 0.0002)) * 1 * (1 / 10_000)
    expected_entry_slippage = abs(100.50 * 1.0002 - 100.50) * 1
    expected_exit_slippage = abs(97.50 - 97.50 * (1 - 0.0002)) * 1
    expected_fees = expected_entry_fee + expected_exit_fee
    expected_slippage = expected_entry_slippage + expected_exit_slippage
    expected_net = expected_gross - expected_fees - expected_slippage

    assert closed["gross_pnl"] == pytest.approx(expected_gross)
    assert closed["transaction_costs"] == pytest.approx(expected_fees)
    assert closed["slippage_cost"] == pytest.approx(expected_slippage)
    assert closed["net_pnl"] == pytest.approx(expected_net)
    assert closed["r_multiple"] == pytest.approx(expected_net / 2.0)
    assert closed_dashboard["metrics"]["realized_pnl"] == pytest.approx(expected_net)
    assert closed_dashboard["metrics"]["unrealized_pnl"] == pytest.approx(0)
    assert closed_dashboard["metrics"]["equity"] == pytest.approx(100_000 + expected_net)

    # The service has no long-lived connection: each operation's context closes it.
    # Reopening the same isolated file simulates backend/service restart.
    restarted_settings = paper.settings(path=db_path)
    restarted = paper.dashboard(path=db_path)
    assert restarted_settings["enabled"] is True
    assert restarted_settings["starting_capital"] == pytest.approx(100_000)
    assert restarted_settings["transaction_cost_bps"] == pytest.approx(1)
    assert restarted_settings["slippage_bps"] == pytest.approx(2)
    assert len(restarted["closed_trades"]) == 1
    assert restarted["closed_trades"][0]["trade_id"] == trade["trade_id"]
    assert restarted["metrics"]["realized_pnl"] == pytest.approx(expected_net)
    assert restarted["metrics"]["equity"] == pytest.approx(100_000 + expected_net)
    assert restarted["open_trades"] == []
    assert restarted["pending_trades"] == []

    duplicate = paper.on_actionable_signal(signal, path=db_path)
    assert duplicate is None, "The unique persisted signal ID must not queue another entry"
    changes = paper.process_completed_candles(candles, path=db_path,
        as_of="2026-11-09T14:45:00+00:00")
    after_duplicate = paper.dashboard(path=db_path)
    assert changes == 0
    assert len(after_duplicate["closed_trades"]) == 1
    assert after_duplicate["metrics"]["realized_pnl"] == pytest.approx(expected_net)
    assert after_duplicate["metrics"]["equity"] == pytest.approx(100_000 + expected_net)
    assert after_duplicate["metrics"]["total_trades"] == 1

    assert db_path.exists(), "All lifecycle state should be persisted to the temporary SQLite file"
    assert not (tmp_path / "FORBIDDEN_DEFAULT.sqlite3").exists()
    print(f"isolated_db={db_path}")
    print(f"signal_id={signal_id}")
    print(f"entry=2026-11-09T14:35:00+00:00 raw=100.50 fill={closed['entry_fill_price']:.8f}")
    print(f"exit=2026-11-09T14:40:00+00:00 reason=STOP_LOSS raw=97.50 fill={closed['exit_fill_price']:.8f}")
    print(f"gross={closed['gross_pnl']:.8f} fees={closed['transaction_costs']:.8f} slippage={closed['slippage_cost']:.8f} net={closed['net_pnl']:.8f}")
    print(f"final_equity={after_duplicate['metrics']['equity']:.8f}; restart=persisted; duplicate_replay=no_change")
