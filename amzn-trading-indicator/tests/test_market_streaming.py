import pandas as pd
import pytest
from fastapi.responses import JSONResponse
import json
import sqlite3

from backend.app.services import market_store
from backend.app.services import market_stream
from backend.app.api import routes


def candle(ts="2026-11-09T14:30:00+00:00", close=250.0):
    return {"timestamp": ts, "open": close - .1, "high": close + .2, "low": close - .2,
            "close": close, "volume": 100, "vwap": close - .05, "data_source": "test_iex"}


def test_seed_and_provider_upsert_are_idempotent(tmp_path):
    db = tmp_path / "market.sqlite3"
    frame = pd.DataFrame([candle()])
    assert market_store.seed_completed_candles(frame, db) == 1
    assert market_store.seed_completed_candles(frame, db) == 0
    inserted = market_store.upsert_completed_candles([candle()], db)
    assert inserted == []
    assert len(market_store.get_candles(path=db)) == 1
    events = market_store.list_events(path=db)
    assert events == []


def test_late_provider_correction_updates_same_timestamp(tmp_path):
    db = tmp_path / "market.sqlite3"
    market_store.upsert_completed_candles([candle()], db)
    updated = market_store.upsert_completed_candles([candle(close=251)], db)
    assert len(updated) == 1
    assert updated[0]["event_type"] == "candle_corrected"
    rows = market_store.get_candles(path=db)
    assert len(rows) == 1 and rows[0]["close"] == 251 and rows[0]["vwap"] == pytest.approx(250.95)


def test_snapshot_restores_real_vwap_without_replacing_existing_ohlcv(tmp_path):
    db = tmp_path / "market.sqlite3"
    existing = candle(close=250)
    existing.pop("vwap")
    market_store.upsert_completed_candles([existing], db)
    snapshot = pd.DataFrame([{**candle(close=251), "symbol": "AMZN"}])
    market_store.seed_completed_candles(snapshot, db)
    row = market_store.get_candles(path=db)[0]
    assert row["close"] == 250
    assert row["vwap"] == pytest.approx(250.95)


def test_provider_reconciliation_backfills_vwap_once_without_changing_ohlcv(tmp_path):
    db = tmp_path / "market.sqlite3"
    original = candle(close=250)
    original.pop("vwap")
    market_store.upsert_completed_candles([original], db)

    restored = market_store.upsert_completed_candles([candle(close=250)], db)
    row = market_store.get_candles(path=db)[0]
    assert len(restored) == 1 and restored[0]["event_type"] == "candle_corrected"
    assert {key: row[key] for key in ("open", "high", "low", "close", "volume")} == {
        key: original[key] for key in ("open", "high", "low", "close", "volume")
    }
    assert row["vwap"] == pytest.approx(249.95)
    assert market_store.upsert_completed_candles([candle(close=250)], db) == []


def test_provider_without_vwap_leaves_missing_vwap_null(tmp_path):
    db = tmp_path / "market.sqlite3"
    bar = candle(close=250)
    bar.pop("vwap")
    market_store.upsert_completed_candles([bar], db)
    assert market_store.upsert_completed_candles([bar], db) == []
    assert market_store.get_candles(path=db)[0]["vwap"] is None


def test_provider_does_not_replace_valid_vwap_for_vwap_only_difference(tmp_path):
    db = tmp_path / "market.sqlite3"
    market_store.upsert_completed_candles([candle(close=250)], db)
    revised_vwap = {**candle(close=250), "vwap": 249.97}
    assert market_store.upsert_completed_candles([revised_vwap], db) == []
    assert market_store.get_candles(path=db)[0]["vwap"] == pytest.approx(249.95)


def test_oldest_missing_vwap_timestamp_excludes_forming_candles(tmp_path):
    db = tmp_path / "market.sqlite3"
    without = candle("2026-11-09T14:30:00+00:00", 250)
    without.pop("vwap")
    market_store.upsert_completed_candles([without, candle("2026-11-09T14:35:00+00:00", 251)], db)
    market_store.upsert_minute_bar(candle("2026-11-09T14:40:00+00:00", 252), db)
    assert market_store.oldest_missing_vwap_timestamp(db) == "2026-11-09T14:30:00+00:00"


def test_missing_interval_is_reported_without_filling(tmp_path):
    db = tmp_path / "market.sqlite3"
    market_store.upsert_completed_candles([candle(), candle("2026-11-09T14:40:00+00:00", 252)], db)
    rows = market_store.get_candles(path=db)
    assert len(rows) == 2
    gap = [event for event in market_store.list_events(path=db) if event["event_type"] == "data_gap_detected"]
    assert len(gap) == 1
    assert gap[0]["payload"]["missing_interval_count"] == 1


def test_completed_bar_upsert_is_idempotent_and_replay_cursor_is_stable(tmp_path):
    db = tmp_path / "market.sqlite3"
    bars = [candle("2026-11-09T14:35:00+00:00", 251), candle("2026-11-09T14:30:00+00:00", 250)]
    first = market_store.upsert_completed_candles(bars, db)
    second = market_store.upsert_completed_candles(list(reversed(bars)), db)
    assert [event["event_type"] for event in first] == ["candle_completed", "candle_completed"]
    assert second == []
    events = market_store.list_events(path=db)
    assert [event["event_id"] for event in events] == sorted(event["event_id"] for event in events)
    assert len(events) == 2
    assert market_store.list_events(after_id=events[0]["event_id"], path=db) == events[1:]
    rows = market_store.get_candles(path=db)
    assert len(rows) == 2 and [row["timestamp"] for row in rows] == sorted(row["timestamp"] for row in rows)


def test_real_minute_updates_only_create_a_forming_bar(tmp_path):
    db = tmp_path / "market.sqlite3"
    result = market_store.upsert_minute_bar(candle("2026-11-09T14:31:00+00:00", 250.1), db)
    assert result and result["event_type"] == "candle_update"
    assert market_store.get_candles(path=db) == []
    forming = market_store.latest_forming(db)
    assert forming is not None
    assert market_store.list_events(path=db)[0]["payload"]["completion_status"] == "forming"


def test_minute_bars_aggregate_chronologically_and_volume_weight_vwap(tmp_path):
    db = tmp_path / "market.sqlite3"
    bars = [
        candle("2026-11-09T14:32:00+00:00", 250.2),
        candle("2026-11-09T14:30:00+00:00", 250.0),
        candle("2026-11-09T14:34:00+00:00", 250.4),
        candle("2026-11-09T14:31:00+00:00", 250.1),
        candle("2026-11-09T14:33:00+00:00", 250.3),
    ]
    for bar in bars:
        market_store.upsert_minute_bar(bar, db)
    forming = market_store.latest_forming(db)
    assert forming["timestamp"] == "2026-11-09T14:30:00+00:00"
    assert forming["open"] == pytest.approx(249.9)
    assert forming["close"] == pytest.approx(250.4)
    assert forming["high"] == pytest.approx(250.6)
    assert forming["low"] == pytest.approx(249.8)
    assert forming["volume"] == pytest.approx(500)
    assert forming["vwap"] == pytest.approx(250.15)
    assert market_store.get_candles(path=db) == []
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM market_minute_bars").fetchone()[0] == 5


def test_out_of_order_minute_correction_updates_forming_bar_without_duplicate_minutes(tmp_path):
    db = tmp_path / "market.sqlite3"
    earlier = candle("2026-11-09T14:32:00+00:00", 250.2)
    later = candle("2026-11-09T14:33:00+00:00", 250.3)
    market_store.upsert_minute_bar(later, db)
    market_store.upsert_minute_bar(earlier, db)
    corrected = candle("2026-11-09T14:32:00+00:00", 251.0)
    market_store.upsert_minute_bar(corrected, db)
    forming = market_store.latest_forming(db)
    assert forming["open"] == pytest.approx(250.9)
    assert forming["close"] == pytest.approx(250.3)
    assert forming["high"] == pytest.approx(251.2)
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM market_minute_bars").fetchone()[0] == 2


def test_analysis_latest_is_persistently_deduplicated_per_candle_and_version(monkeypatch):
    service = market_stream.MarketStreamService()
    rows = [{**candle(), "symbol": "AMZN"}]
    analyzed = []
    published = []
    exists = {"value": False}

    monkeypatch.setattr(market_store, "get_candles", lambda **_kwargs: rows)
    monkeypatch.setattr(market_stream, "get_signal_for_candle",
                        lambda timestamp, version: {"signal_id": "already-saved"} if exists["value"] else None)

    def analyze(frame, **_kwargs):
        analyzed.append(frame.iloc[-1]["timestamp"])
        signal = {"signal_id": "saved-once", "classification": "BUY"}
        exists["value"] = True
        return {"analysis": {"candle_timestamp": rows[-1]["timestamp"]}, "signal_record": signal}

    monkeypatch.setattr("backend.app.services.analysis.service.analyze_candles", analyze)
    monkeypatch.setattr(market_store, "publish_event", lambda kind, payload: published.append((kind, payload)))
    service._analyze_latest()
    service._analyze_latest()
    assert len(analyzed) == 1
    assert [kind for kind, _ in published] == ["analysis_updated", "signal_created"]


def test_forming_only_state_does_not_trigger_analysis(monkeypatch):
    service = market_stream.MarketStreamService()
    monkeypatch.setattr(market_store, "get_candles", lambda **_kwargs: [])
    monkeypatch.setattr(market_stream, "get_signal_for_candle",
                        lambda *_args: (_ for _ in ()).throw(AssertionError("no completed candle means no signal lookup")))
    service._analyze_latest()


def test_analysis_failure_is_published_and_not_marked_successful(monkeypatch):
    service = market_stream.MarketStreamService()
    rows = [{**candle(), "symbol": "AMZN"}]
    events = []
    monkeypatch.setattr(market_store, "get_candles", lambda **_kwargs: rows)
    monkeypatch.setattr(market_stream, "get_signal_for_candle", lambda *_args: None)
    monkeypatch.setattr("backend.app.services.analysis.service.analyze_candles",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("test analysis error")))
    monkeypatch.setattr(market_store, "publish_event", lambda kind, payload: events.append((kind, payload)))
    with pytest.raises(ValueError, match="test analysis error"):
        service._analyze_latest()
    assert len(events) == 1 and events[0][0] == "analysis_failed"
    assert "test analysis error" in events[0][1]["error"]


def test_stream_start_is_idempotent_for_one_service_instance(monkeypatch):
    service = market_stream.MarketStreamService()
    service._seed_snapshot = lambda: None
    monkeypatch.setattr(market_stream, "load_dotenv", lambda *_args: None)
    monkeypatch.setenv("ALPACA_API_KEY", "configured-test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "configured-test-secret")
    created = []

    class FakeThread:
        def __init__(self, **_kwargs): self.alive = False
        def start(self): self.alive = True; created.append(self)
        def is_alive(self): return self.alive

    monkeypatch.setattr(market_stream.threading, "Thread", FakeThread)
    assert service.start() is True
    assert service.start() is True
    assert len(created) == 1


def test_timestamps_must_be_timezone_aware(tmp_path):
    bad = candle("2026-11-09T14:30:00", 250)
    try:
        market_store.upsert_completed_candles([bad], tmp_path / "market.sqlite3")
    except ValueError as exc:
        assert "timezone-aware" in str(exc)
    else:
        raise AssertionError("naive market timestamps were accepted")


def test_old_sqlite_candle_schema_is_additively_repaired_with_amzn_symbol(tmp_path):
    db = tmp_path / "legacy-market.sqlite3"
    connection = sqlite3.connect(db)
    connection.execute("""CREATE TABLE market_candles (
        timestamp TEXT PRIMARY KEY, end_timestamp TEXT NOT NULL, open REAL NOT NULL,
        high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL, volume REAL NOT NULL,
        data_source TEXT NOT NULL, is_complete INTEGER NOT NULL, provider_timestamp TEXT,
        updated_at TEXT NOT NULL)""")
    connection.execute("INSERT INTO market_candles VALUES(?,?,?,?,?,?,?,?,?,?,?)", (
        "2026-11-09T14:30:00+00:00", "2026-11-09T14:35:00+00:00", 249.9, 250.2,
        249.8, 250, 100, "alpaca_iex_historical_snapshot", 1,
        "2026-11-09T14:30:00+00:00", "2026-11-09T14:35:00+00:00"))
    connection.commit()
    connection.close()

    rows = market_store.get_candles(path=db)
    with sqlite3.connect(db) as migrated:
        columns = {row[1] for row in migrated.execute("PRAGMA table_info(market_candles)")}
        count = migrated.execute("SELECT COUNT(*) FROM market_candles").fetchone()[0]
    assert {"symbol", "vwap"}.issubset(columns) and count == 1
    assert rows[0]["symbol"] == "AMZN"
    assert rows[0]["vwap"] is None


@pytest.mark.parametrize("symbol", [None, float("nan"), ""])
def test_missing_or_nan_symbol_is_rejected(symbol):
    record = candle()
    record["symbol"] = symbol
    with pytest.raises(ValueError, match="symbol"):
        market_store.validate_candle_record(record)


@pytest.mark.parametrize("field,value", [("open", float("nan")), ("high", float("inf")),
                                          ("low", float("-inf")), ("close", float("nan")),
                                          ("volume", float("inf"))])
def test_nonfinite_ohlcv_is_rejected(field, value):
    record = {**candle(), "symbol": "AMZN"}
    record[field] = value
    with pytest.raises(ValueError, match="non-finite"):
        market_store.validate_candle_record(record)


def _patch_market_route(monkeypatch, frame):
    monkeypatch.setattr(routes, "_candles", lambda source: frame.copy())
    monkeypatch.setattr(routes.market_store, "get_candles", lambda **kwargs: [])
    monkeypatch.setattr(routes.market_store, "latest_event_id", lambda: 7)
    monkeypatch.setattr(routes, "_current_forming_candle", lambda: None)
    monkeypatch.setattr(routes, "_market_stream_status", lambda: {
        "provider": "Alpaca", "symbol": "AMZN", "feed": "IEX", "connection_state": "DISCONNECTED",
        "market_state": "CLOSED", "freshness": "UNAVAILABLE"})
    monkeypatch.setattr(routes, "dataset_status", lambda limit: {"rows": len(frame), "source": "LOCAL SNAPSHOT"})


def _serialize_market_route(monkeypatch, frame):
    _patch_market_route(monkeypatch, frame)
    body = routes.market_candles(limit=1000, source="local")
    response = JSONResponse(content=body)
    return response, json.loads(response.body)


def test_market_candles_response_has_valid_amzn_symbol_and_json(monkeypatch):
    frame = pd.DataFrame([{**candle(), "symbol": "AMZN", "trade_count": float("nan"), "vwap": float("inf")}])
    response, body = _serialize_market_route(monkeypatch, frame)
    assert response.status_code == 200
    assert body["candles"][0]["symbol"] == "AMZN"
    assert body["candles"][0]["trade_count"] is None
    assert body["candles"][0]["vwap"] is None


def test_market_candles_empty_result_is_json_serializable(monkeypatch):
    columns = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
    response, body = _serialize_market_route(monkeypatch, pd.DataFrame(columns=columns))
    assert response.status_code == 200
    assert body["candles"] == [] and body["latest_timestamp"] is None


def test_market_candles_excludes_bad_symbols_and_nonfinite_prices(monkeypatch, caplog):
    valid = {**candle("2026-11-09T14:30:00+00:00", 250), "symbol": "AMZN"}
    missing_symbol = {**candle("2026-11-09T14:35:00+00:00", 251), "symbol": float("nan")}
    invalid_price = {**candle("2026-11-09T14:40:00+00:00", 252), "symbol": "AMZN", "close": float("inf")}
    response, body = _serialize_market_route(monkeypatch, pd.DataFrame([valid, missing_symbol, invalid_price]))
    assert response.status_code == 200
    assert len(body["candles"]) == 1 and body["candles"][0]["symbol"] == "AMZN"
    assert "Excluded 2 invalid AMZN candle record(s)" in caplog.text


def test_market_candles_endpoint_handler_returns_http_200_for_valid_stored_records(tmp_path, monkeypatch):
    db = tmp_path / "endpoint.sqlite3"
    market_store.upsert_completed_candles([candle()], db)
    original_get_candles = market_store.get_candles
    monkeypatch.setattr(routes, "_candles", lambda source: pd.DataFrame(
        columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"]))
    monkeypatch.setattr(market_store, "get_candles", lambda limit=10000, after=None: original_get_candles(limit, after, db))
    monkeypatch.setattr(market_store, "latest_event_id", lambda: 1)
    monkeypatch.setattr(routes, "_current_forming_candle", lambda: None)
    monkeypatch.setattr(routes, "_market_stream_status", lambda: {"provider": "Alpaca", "symbol": "AMZN",
        "feed": "IEX", "connection_state": "DISCONNECTED", "market_state": "CLOSED", "freshness": "FRESH"})
    monkeypatch.setattr(routes, "dataset_status", lambda limit: {"rows": 1, "source": "LOCAL SNAPSHOT"})
    response = JSONResponse(content=routes.market_candles(limit=1000, source="local"))
    body = json.loads(response.body)
    route = next(item for item in routes.router.routes if getattr(item, "path", None) == "/market/candles")
    assert route.endpoint is routes.market_candles
    assert response.status_code == 200 and body["candles"][0]["symbol"] == "AMZN"

