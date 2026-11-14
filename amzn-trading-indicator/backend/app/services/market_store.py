"""Durable completed/forming market candles and replayable market events."""
from __future__ import annotations

import json
import logging
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[3] / "artifacts" / "market_stream.sqlite3"
MARKET_SYMBOL = "AMZN"
logger = logging.getLogger(__name__)


@contextmanager
def connect(path: str | Path | None = None):
    target = Path(path or DB_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(target, timeout=30)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript("""
        CREATE TABLE IF NOT EXISTS market_candles (
          timestamp TEXT PRIMARY KEY, end_timestamp TEXT NOT NULL, symbol TEXT NOT NULL DEFAULT 'AMZN',
          open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL,
          close REAL NOT NULL, volume REAL NOT NULL, vwap REAL, data_source TEXT NOT NULL,
          is_complete INTEGER NOT NULL, provider_timestamp TEXT, updated_at TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS ix_market_candles_complete ON market_candles(is_complete,timestamp);
        CREATE TABLE IF NOT EXISTS market_minute_bars (
          timestamp TEXT PRIMARY KEY, open REAL NOT NULL, high REAL NOT NULL,
          low REAL NOT NULL, close REAL NOT NULL, volume REAL NOT NULL, vwap REAL,
          updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS market_events (
          event_id INTEGER PRIMARY KEY AUTOINCREMENT, event_type TEXT NOT NULL,
          event_timestamp TEXT NOT NULL, candle_timestamp TEXT, completion_status TEXT,
          data_source TEXT NOT NULL, payload TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS ix_market_events_timestamp ON market_events(event_id);
        CREATE TABLE IF NOT EXISTS market_stream_status (
          singleton INTEGER PRIMARY KEY CHECK(singleton=1), provider TEXT NOT NULL,
          symbol TEXT NOT NULL, feed TEXT NOT NULL, connection_state TEXT NOT NULL,
          last_received_event_at TEXT, last_completed_candle TEXT, last_persisted_candle TEXT,
          last_reconciliation_at TEXT, reconciliation_error TEXT, last_error TEXT,
          retry_count INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL);
        """)
        columns = {row[1] for row in db.execute("PRAGMA table_info(market_candles)")}
        if "symbol" not in columns:
            # The original AMZN-only store omitted this field. Its existing rows
            # all came from the AMZN-specific snapshot/provider path, so the
            # additive default is a safe, deterministic repair.
            old_count = db.execute("SELECT COUNT(*) FROM market_candles").fetchone()[0]
            db.execute("ALTER TABLE market_candles ADD COLUMN symbol TEXT NOT NULL DEFAULT 'AMZN'")
            logger.warning("Added missing symbol column to market_candles; %d existing AMZN records defaulted to %s", old_count, MARKET_SYMBOL)
            columns.add("symbol")
        if "vwap" not in columns:
            db.execute("ALTER TABLE market_candles ADD COLUMN vwap REAL")
            logger.info("Added nullable market_candles.vwap; legacy values will be restored only from source market data")
        minute_columns = {row[1] for row in db.execute("PRAGMA table_info(market_minute_bars)")}
        if "vwap" not in minute_columns:
            db.execute("ALTER TABLE market_minute_bars ADD COLUMN vwap REAL")
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _utc(value) -> str:
    stamp = datetime.fromisoformat(value) if isinstance(value, str) else value
    if stamp.tzinfo is None:
        raise ValueError("Market timestamps must be timezone-aware")
    return stamp.astimezone(timezone.utc).isoformat()


def _event(db, event_type: str, payload: dict, candle_timestamp: str | None = None,
           complete: bool | None = None, source: str = "alpaca_iex") -> int:
    now = datetime.now(timezone.utc).isoformat()
    cursor = db.execute("""INSERT INTO market_events
      (event_type,event_timestamp,candle_timestamp,completion_status,data_source,payload)
      VALUES(?,?,?,?,?,?)""", (event_type, now, candle_timestamp,
      "completed" if complete is True else "forming" if complete is False else None,
      source, json.dumps(payload, separators=(",", ":"), default=str)))
    return int(cursor.lastrowid)


def _bar_values(bar: dict) -> dict:
    normalized = validate_candle_record(bar, allow_missing_symbol=True)
    stamp = normalized["timestamp"]
    return {"timestamp": stamp, "end_timestamp": _utc(datetime.fromisoformat(stamp) + timedelta(minutes=5)),
            "symbol": normalized["symbol"], "open": normalized["open"], "high": normalized["high"],
            "low": normalized["low"], "close": normalized["close"], "volume": normalized["volume"],
            "vwap": normalized["vwap"],
            "data_source": str(bar.get("data_source", "alpaca_iex_5m")),
            "provider_timestamp": str(bar.get("provider_timestamp") or stamp)}


def validate_candle_record(bar: dict, *, allow_missing_symbol: bool = False) -> dict:
    """Validate core market fields; never turn an invalid price into a valid candle."""
    if "symbol" not in bar:
        if not allow_missing_symbol:
            raise ValueError("Market candle has no symbol")
        symbol = MARKET_SYMBOL
    else:
        raw_symbol = bar["symbol"]
        if not isinstance(raw_symbol, str) or not raw_symbol.strip():
            raise ValueError("Market candle has a missing or non-text symbol")
        symbol = raw_symbol.strip().upper()
        if symbol != MARKET_SYMBOL:
            raise ValueError(f"Unexpected market symbol {symbol!r}; expected {MARKET_SYMBOL}")
    _check_ohlcv(bar)
    normalized = {"symbol": symbol, "timestamp": _utc(bar["timestamp"]),
                  **{key: float(bar[key]) for key in ("open", "high", "low", "close", "volume")}}
    raw_vwap = bar.get("vwap")
    try:
        vwap = float(raw_vwap) if raw_vwap is not None else None
    except (TypeError, ValueError, OverflowError):
        vwap = None
    normalized["vwap"] = vwap if vwap is not None and math.isfinite(vwap) and vwap > 0 else None
    return normalized


def _check_ohlcv(bar: dict) -> None:
    values = [float(bar[key]) for key in ("open", "high", "low", "close", "volume")]
    if not all(map(lambda x: x == x and abs(x) != float("inf"), values)) or values[4] < 0:
        raise ValueError("Market candle contains non-finite OHLCV or negative volume")
    o, h, low, c, _ = values
    if low <= 0 or low > min(o, c) or h < max(o, c):
        raise ValueError("Market candle violates OHLC price bounds")


def seed_completed_candles(frame, path: str | Path | None = None) -> int:
    """Load the existing historical snapshot idempotently without replacing live corrections."""
    rows = []
    now = datetime.now(timezone.utc).isoformat()
    if "symbol" not in frame.columns:
        logger.info("AMZN single-symbol snapshot has no symbol column; assigning configured symbol %s", MARKET_SYMBOL)
    columns = ["timestamp", "open", "high", "low", "close", "volume"] + (["symbol"] if "symbol" in frame.columns else []) + (["vwap"] if "vwap" in frame.columns else [])
    for row in frame[columns].itertuples(index=False, name=None):
        values = dict(zip(columns, row))
        bar = {**values, "data_source": "alpaca_iex_historical_snapshot"}
        try:
            v = _bar_values(bar)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            logger.warning("Skipping invalid AMZN snapshot candle at %r: %s", values.get("timestamp"), exc)
            continue
        rows.append((v["timestamp"], v["end_timestamp"], v["symbol"], v["open"], v["high"], v["low"], v["close"],
                     v["volume"], v["vwap"], v["data_source"], 1, v["provider_timestamp"], now))
    with connect(path) as db:
        before = db.total_changes
        db.executemany("""INSERT OR IGNORE INTO market_candles
          (timestamp,end_timestamp,symbol,open,high,low,close,volume,vwap,data_source,is_complete,provider_timestamp,updated_at)
          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(timestamp) DO UPDATE SET
          vwap=excluded.vwap WHERE market_candles.vwap IS NULL AND excluded.vwap IS NOT NULL""", rows)
        inserted = db.total_changes - before
        latest = db.execute("SELECT timestamp FROM market_candles WHERE is_complete=1 ORDER BY timestamp DESC LIMIT 1").fetchone()
    if latest:
        write_status({"last_completed_candle": latest["timestamp"], "last_persisted_candle": latest["timestamp"]}, emit=False, path=path)
    return inserted


def upsert_completed_candles(bars: list[dict], path: str | Path | None = None) -> list[dict]:
    """Insert provider-confirmed bars or apply genuine late corrections by timestamp."""
    emitted = []
    with connect(path) as db:
        for raw in sorted(bars, key=lambda item: _utc(item["timestamp"])):
            _check_ohlcv(raw)
            bar = _bar_values(raw)
            old = db.execute("SELECT * FROM market_candles WHERE timestamp=?", (bar["timestamp"],)).fetchone()
            ohlcv_same = bool(old and all(
                abs(float(old[key]) - bar[key]) < 1e-10 for key in ("open", "high", "low", "close", "volume")))
            # Keep an existing valid VWAP when only a differing VWAP is returned.
            # A genuine OHLCV correction can still replace it alongside the bar.
            if old and old["vwap"] is not None and ohlcv_same:
                bar["vwap"] = float(old["vwap"])
            vwap_same = bar["vwap"] is None or (old and old["vwap"] is not None and abs(float(old["vwap"]) - bar["vwap"]) < 1e-10)
            same = bool(old and old["is_complete"] and old["symbol"] == bar["symbol"] and ohlcv_same and vwap_same)
            if same:
                continue
            now = datetime.now(timezone.utc).isoformat()
            db.execute("""INSERT INTO market_candles
              (timestamp,end_timestamp,symbol,open,high,low,close,volume,vwap,data_source,is_complete,provider_timestamp,updated_at)
              VALUES(?,?,?,?,?,?,?,?,?,?,1,?,?) ON CONFLICT(timestamp) DO UPDATE SET
              end_timestamp=excluded.end_timestamp,open=excluded.open,high=excluded.high,low=excluded.low,
              symbol=excluded.symbol,
              close=excluded.close,volume=excluded.volume,vwap=COALESCE(excluded.vwap,market_candles.vwap),data_source=excluded.data_source,is_complete=1,
              provider_timestamp=excluded.provider_timestamp,updated_at=excluded.updated_at""",
              (bar["timestamp"], bar["end_timestamp"], bar["symbol"], bar["open"], bar["high"], bar["low"], bar["close"],
               bar["volume"], bar["vwap"], bar["data_source"], bar["provider_timestamp"], now))
            candle = {k: bar[k] for k in ("timestamp", "symbol", "open", "high", "low", "close", "volume", "vwap")}
            is_correction = bool(old and old["is_complete"])
            event_type = "candle_corrected" if is_correction else "candle_completed"
            event_id = _event(db, event_type, {"symbol": "AMZN", "candle": candle,
              "event_timestamp": now, "completion_status": "completed", "data_source": bar["data_source"]},
              bar["timestamp"], True, bar["data_source"])
            event = {"event_id": event_id, "event_type": event_type, "event_timestamp": now,
                     "candle_timestamp": bar["timestamp"], "candle": candle,
                     "completion_status": "completed", "data_source": bar["data_source"]}
            emitted.append(event)
            db.execute("DELETE FROM market_candles WHERE timestamp=? AND is_complete=0", (bar["timestamp"],))
            db.execute("UPDATE market_stream_status SET last_completed_candle=?,last_persisted_candle=?,updated_at=? WHERE singleton=1",
                       (bar["timestamp"], bar["timestamp"], now))
        # Surface missing regular-session intervals; do not synthesize replacements.
        completed_events = list(emitted)
        for event in completed_events:
            current = datetime.fromisoformat(event["candle_timestamp"])
            prior = db.execute("SELECT timestamp FROM market_candles WHERE is_complete=1 AND timestamp<? ORDER BY timestamp DESC LIMIT 1",
                               (event["candle_timestamp"],)).fetchone()
            if prior:
                previous = datetime.fromisoformat(prior["timestamp"])
                if previous.date() == current.date() and (current - previous).total_seconds() > 300:
                    missing = int((current - previous).total_seconds() // 300) - 1
                    if missing > 0:
                        gap = {"symbol": "AMZN", "from": prior["timestamp"], "to": current.isoformat(),
                               "missing_interval_count": missing, "event_timestamp": datetime.now(timezone.utc).isoformat(),
                               "data_source": "alpaca_iex"}
                        eid = _event(db, "data_gap_detected", gap, current.isoformat(), None)
                        emitted.append({"event_id": eid, "event_type": "data_gap_detected", **gap})
    return emitted


def upsert_minute_bar(bar: dict, path: str | Path | None = None) -> dict | None:
    """Persist a genuine Alpaca 1-minute bar and publish its partial 5-minute aggregate."""
    minute = _utc(bar["timestamp"])
    _check_ohlcv(bar)
    import pandas as pd
    bucket = pd.Timestamp(minute).floor("5min").isoformat()
    now = datetime.now(timezone.utc).isoformat()
    with connect(path) as db:
        minute_vwap = validate_candle_record({**bar, "symbol": MARKET_SYMBOL}).get("vwap")
        db.execute("""INSERT INTO market_minute_bars(timestamp,open,high,low,close,volume,vwap,updated_at)
          VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(timestamp) DO UPDATE SET open=excluded.open,high=excluded.high,
          low=excluded.low,close=excluded.close,volume=excluded.volume,vwap=excluded.vwap,updated_at=excluded.updated_at""",
          (minute, float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"]), float(bar["volume"]), minute_vwap, now))
        complete = db.execute("SELECT is_complete FROM market_candles WHERE timestamp=?", (bucket,)).fetchone()
        if complete and complete["is_complete"]:
            return None
        parts = db.execute("SELECT * FROM market_minute_bars WHERE timestamp>=? AND timestamp<? ORDER BY timestamp",
                           (bucket, (pd.Timestamp(bucket) + pd.Timedelta(minutes=5)).isoformat())).fetchall()
        if not parts:
            return None
        aggregate = {"timestamp": bucket, "symbol": MARKET_SYMBOL, "open": float(parts[0]["open"]), "high": max(float(r["high"]) for r in parts),
                     "low": min(float(r["low"]) for r in parts), "close": float(parts[-1]["close"]),
                     "volume": sum(float(r["volume"]) for r in parts)}
        observed_vwaps = [float(row["vwap"]) for row in parts if row["vwap"] is not None and math.isfinite(float(row["vwap"]))]
        if len(observed_vwaps) == len(parts) and aggregate["volume"] > 0:
            aggregate["vwap"] = sum(float(row["vwap"]) * float(row["volume"]) for row in parts) / aggregate["volume"]
        else:
            aggregate["vwap"] = None
        # This row is explicitly incomplete and is never returned as an analysis candle.
        db.execute("""INSERT INTO market_candles(timestamp,end_timestamp,symbol,open,high,low,close,volume,vwap,data_source,
          is_complete,provider_timestamp,updated_at) VALUES(?,?,?,?,?,?,?,?,?,'alpaca_iex_minute_aggregation',0,?,?)
          ON CONFLICT(timestamp) DO UPDATE SET open=excluded.open,high=excluded.high,low=excluded.low,
          symbol=excluded.symbol,
          close=excluded.close,volume=excluded.volume,vwap=excluded.vwap,data_source=excluded.data_source,provider_timestamp=excluded.provider_timestamp,
          updated_at=excluded.updated_at WHERE market_candles.is_complete=0""",
          (bucket, (pd.Timestamp(bucket) + pd.Timedelta(minutes=5)).isoformat(), MARKET_SYMBOL, aggregate["open"], aggregate["high"],
           aggregate["low"], aggregate["close"], aggregate["volume"], aggregate["vwap"], minute, now))
        event_payload = {"symbol": "AMZN", "candle": aggregate, "event_timestamp": now,
                         "completion_status": "forming", "data_source": "alpaca_iex_minute_aggregation",
                         "observed_minute_count": len(parts)}
        event_id = _event(db, "candle_update", event_payload, bucket, False, "alpaca_iex_minute_aggregation")
        db.execute("UPDATE market_stream_status SET last_received_event_at=?,updated_at=? WHERE singleton=1", (now, now))
        return {"event_id": event_id, "event_type": "candle_update", **event_payload}


def write_status(values: dict, path: str | Path | None = None, emit: bool = True) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    defaults = {"provider": "Alpaca", "symbol": "AMZN", "feed": "IEX", "connection_state": "CONNECTING",
                "last_received_event_at": None, "last_completed_candle": None, "last_persisted_candle": None,
                "last_reconciliation_at": None, "reconciliation_error": None, "last_error": None, "retry_count": 0}
    with connect(path) as db:
        old = db.execute("SELECT * FROM market_stream_status WHERE singleton=1").fetchone()
        state = {**defaults, **(dict(old) if old else {}), **values, "updated_at": now}
        db.execute("""INSERT INTO market_stream_status(singleton,provider,symbol,feed,connection_state,last_received_event_at,
          last_completed_candle,last_persisted_candle,last_reconciliation_at,reconciliation_error,last_error,retry_count,updated_at)
          VALUES(1,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(singleton) DO UPDATE SET provider=excluded.provider,
          symbol=excluded.symbol,feed=excluded.feed,connection_state=excluded.connection_state,
          last_received_event_at=excluded.last_received_event_at,last_completed_candle=excluded.last_completed_candle,
          last_persisted_candle=excluded.last_persisted_candle,last_reconciliation_at=excluded.last_reconciliation_at,
          reconciliation_error=excluded.reconciliation_error,last_error=excluded.last_error,
          retry_count=excluded.retry_count,updated_at=excluded.updated_at""",
          tuple(state[k] for k in ("provider","symbol","feed","connection_state","last_received_event_at",
          "last_completed_candle","last_persisted_candle","last_reconciliation_at","reconciliation_error","last_error","retry_count","updated_at")))
        changed = old is None or old["connection_state"] != state["connection_state"] or old["last_error"] != state["last_error"]
        if emit and changed:
            _event(db, "market_status", {**state, "event_timestamp": now, "data_source": "alpaca_iex"})
        return state


def get_status(path: str | Path | None = None) -> dict:
    with connect(path) as db:
        row = db.execute("SELECT * FROM market_stream_status WHERE singleton=1").fetchone()
        return dict(row) if row else {"provider": "Alpaca", "symbol": "AMZN", "feed": "IEX",
                                     "connection_state": "DISCONNECTED", "updated_at": None}


def get_candles(limit: int = 10000, after: str | None = None, path: str | Path | None = None) -> list[dict]:
    with connect(path) as db:
        if after:
            rows = db.execute("SELECT timestamp,symbol,open,high,low,close,volume,vwap FROM market_candles WHERE is_complete=1 AND timestamp>? ORDER BY timestamp LIMIT ?",
                              (_utc(after), limit)).fetchall()
        else:
            rows = db.execute("SELECT timestamp,symbol,open,high,low,close,volume,vwap FROM market_candles WHERE is_complete=1 ORDER BY timestamp DESC LIMIT ?",
                              (limit,)).fetchall()
            rows = list(reversed(rows))
        valid = []
        for row in rows:
            record = dict(row)
            try:
                valid.append(validate_candle_record(record))
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                logger.warning("Excluding invalid stored AMZN candle at %r: %s", record.get("timestamp"), exc)
        return valid


def oldest_missing_vwap_timestamp(path: str | Path | None = None) -> str | None:
    """Find the earliest completed candle whose source VWAP can be reconciled."""
    with connect(path) as db:
        row = db.execute("SELECT MIN(timestamp) FROM market_candles WHERE is_complete=1 AND vwap IS NULL").fetchone()
        return row[0] if row else None


def latest_forming(path: str | Path | None = None) -> dict | None:
    with connect(path) as db:
        row = db.execute("SELECT timestamp,symbol,open,high,low,close,volume,vwap,data_source,updated_at FROM market_candles WHERE is_complete=0 ORDER BY timestamp DESC LIMIT 1").fetchone()
        if not row:
            return None
        result = dict(row)
        try:
            result.update(validate_candle_record(result))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            logger.warning("Excluding invalid forming AMZN candle at %r: %s", result.get("timestamp"), exc)
            return None
        start = datetime.fromisoformat(result["timestamp"])
        end = (start + timedelta(minutes=5)).isoformat()
        result["observed_minute_count"] = int(db.execute(
            "SELECT COUNT(*) FROM market_minute_bars WHERE timestamp>=? AND timestamp<?",
            (result["timestamp"], end)).fetchone()[0])
        return result


def list_events(after_id: int = 0, limit: int = 250, path: str | Path | None = None) -> list[dict]:
    with connect(path) as db:
        rows = db.execute("SELECT * FROM market_events WHERE event_id>? ORDER BY event_id LIMIT ?", (after_id, limit)).fetchall()
        events = []
        for row in rows:
            event = dict(row)
            event["payload"] = json.loads(event["payload"])
            events.append(event)
        return events


def latest_event_id(path: str | Path | None = None) -> int:
    with connect(path) as db:
        return int(db.execute("SELECT COALESCE(MAX(event_id),0) FROM market_events").fetchone()[0])


def publish_event(event_type: str, payload: dict, path: str | Path | None = None) -> int:
    with connect(path) as db:
        return _event(db, event_type, payload, payload.get("candle_timestamp"), None,
                      str(payload.get("data_source", "alpaca_iex")))
