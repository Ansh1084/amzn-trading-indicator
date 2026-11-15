"""Persistent simulated AMZN trade ledger; this module never submits broker orders."""
from __future__ import annotations

import json
import logging
import math
import sqlite3
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.forecasts.calendar import is_trading_day, session_close

logger = logging.getLogger(__name__)
DB_PATH = Path(__file__).resolve().parents[3] / "artifacts" / "paper_trading.sqlite3"
DEFAULT_CAPITAL = 100_000.0
MAX_HOLD_MINUTES = 180


@contextmanager
def _connect(path=None):
    target = Path(path or DB_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(target, timeout=30)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript("""
        CREATE TABLE IF NOT EXISTS paper_settings (
          singleton INTEGER PRIMARY KEY CHECK(singleton=1), enabled INTEGER NOT NULL DEFAULT 0,
          starting_capital REAL NOT NULL, sizing_mode TEXT NOT NULL, sizing_value REAL NOT NULL,
          transaction_cost_bps REAL NOT NULL, slippage_bps REAL NOT NULL,
          enabled_at TEXT, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS paper_trades (
          trade_id TEXT PRIMARY KEY, signal_id TEXT NOT NULL UNIQUE, symbol TEXT NOT NULL,
          setup_type TEXT NOT NULL, direction TEXT NOT NULL, signal_timestamp TEXT NOT NULL,
          status TEXT NOT NULL, entry_timestamp TEXT, entry_price REAL, entry_fill_price REAL, quantity REAL NOT NULL,
          remaining_quantity REAL NOT NULL, capital_allocated REAL NOT NULL,
          entry_lower REAL NOT NULL, entry_upper REAL NOT NULL,
          entry_reference_price REAL,
          stop_loss REAL NOT NULL, target_1 REAL NOT NULL, target_2 REAL NOT NULL,
          transaction_cost_bps REAL NOT NULL DEFAULT 1, slippage_bps REAL NOT NULL DEFAULT 2,
          exit_price REAL, exit_fill_price REAL, exit_reference_price REAL, exit_timestamp TEXT, exit_reason TEXT,
          gross_pnl REAL NOT NULL DEFAULT 0, transaction_costs REAL NOT NULL DEFAULT 0,
          slippage_cost REAL NOT NULL DEFAULT 0, net_pnl REAL, realized_net_pnl REAL NOT NULL DEFAULT 0,
          risk_per_share REAL NOT NULL, r_multiple REAL,
          mark_price REAL, mark_timestamp TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS ix_paper_trades_status ON paper_trades(status, signal_timestamp);
        CREATE TABLE IF NOT EXISTS paper_journal (
          journal_id INTEGER PRIMARY KEY AUTOINCREMENT, trade_id TEXT NOT NULL,
          timestamp TEXT NOT NULL, event_type TEXT NOT NULL, details TEXT NOT NULL,
          FOREIGN KEY(trade_id) REFERENCES paper_trades(trade_id));
        CREATE INDEX IF NOT EXISTS ix_paper_journal_trade ON paper_journal(trade_id,journal_id);
        """)
        columns = {row[1] for row in db.execute("PRAGMA table_info(paper_trades)")}
        for name, declaration in (("entry_lower", "REAL NOT NULL DEFAULT 0"), ("entry_upper", "REAL NOT NULL DEFAULT 0"),
                                  ("entry_reference_price", "REAL"), ("exit_reference_price", "REAL"),
                                  ("entry_fill_price", "REAL"), ("exit_fill_price", "REAL"),
                                  ("realized_net_pnl", "REAL NOT NULL DEFAULT 0"),
                                  ("transaction_cost_bps", "REAL NOT NULL DEFAULT 1"),
                                  ("slippage_bps", "REAL NOT NULL DEFAULT 2")):
            if name not in columns:
                db.execute(f"ALTER TABLE paper_trades ADD COLUMN {name} {declaration}")
        db.execute("""INSERT OR IGNORE INTO paper_settings
          (singleton,enabled,starting_capital,sizing_mode,sizing_value,transaction_cost_bps,slippage_bps,updated_at)
          VALUES(1,0,?,'fixed_shares',1,1,2,?)""", (DEFAULT_CAPITAL, _now()))
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _decode(row):
    return dict(row) if row else None


def _record_event(db, trade_id, event_type, details, timestamp=None):
    db.execute("INSERT INTO paper_journal(trade_id,timestamp,event_type,details) VALUES(?,?,?,?)",
               (trade_id, timestamp or _now(), event_type, json.dumps(details, default=str)))


def settings(path=None):
    with _connect(path) as db:
        row = db.execute("SELECT * FROM paper_settings WHERE singleton=1").fetchone()
        result = _decode(row)
        result["enabled"] = bool(result["enabled"])
        return result


def update_settings(*, enabled=None, starting_capital=None, sizing_mode=None, sizing_value=None,
                    transaction_cost_bps=None, slippage_bps=None, path=None):
    allowed_modes = {"fixed_shares", "fixed_capital", "equity_percent"}
    with _connect(path) as db:
        current = dict(db.execute("SELECT * FROM paper_settings WHERE singleton=1").fetchone())
        if starting_capital is not None:
            if not math.isfinite(starting_capital) or starting_capital <= 0:
                raise ValueError("Starting capital must be finite and greater than zero")
            if db.execute("SELECT 1 FROM paper_trades LIMIT 1").fetchone():
                raise ValueError("Starting capital cannot change after the first paper trade is recorded")
            current["starting_capital"] = float(starting_capital)
        if sizing_mode is not None:
            if sizing_mode not in allowed_modes:
                raise ValueError("Sizing mode must be fixed_shares, fixed_capital, or equity_percent")
            current["sizing_mode"] = sizing_mode
        if sizing_value is not None:
            if not math.isfinite(sizing_value) or sizing_value <= 0:
                raise ValueError("Sizing value must be finite and greater than zero")
            if current["sizing_mode"] == "equity_percent" and sizing_value > 100:
                raise ValueError("Equity sizing percentage must be at most 100")
            current["sizing_value"] = float(sizing_value)
        for name, value in (("transaction_cost_bps", transaction_cost_bps), ("slippage_bps", slippage_bps)):
            if value is not None:
                if not math.isfinite(value) or value < 0 or value > 500:
                    raise ValueError(f"{name} must be between zero and 500 basis points")
                current[name] = float(value)
        if enabled is not None:
            current["enabled"] = int(bool(enabled))
            current["enabled_at"] = _now() if enabled else None
            if not enabled:
                pending = db.execute("SELECT trade_id FROM paper_trades WHERE status='PENDING_ENTRY'").fetchall()
                for row in pending:
                    db.execute("UPDATE paper_trades SET status='CANCELLED_PAUSED',exit_reason='PAUSED_BEFORE_ENTRY',updated_at=? WHERE trade_id=?",
                               (_now(), row["trade_id"]))
                    _record_event(db, row["trade_id"], "CANCELLED_PAUSED", {"reason": "Paper trading was paused before entry fill"})
        current["updated_at"] = _now()
        db.execute("""UPDATE paper_settings SET enabled=?,starting_capital=?,sizing_mode=?,sizing_value=?,
          transaction_cost_bps=?,slippage_bps=?,enabled_at=?,updated_at=? WHERE singleton=1""",
          tuple(current[key] for key in ("enabled", "starting_capital", "sizing_mode", "sizing_value",
                                         "transaction_cost_bps", "slippage_bps", "enabled_at", "updated_at")))
    return settings(path)


def _open_allocated(db):
    row = db.execute("SELECT COALESCE(SUM(capital_allocated),0) AS allocated FROM paper_trades WHERE status IN ('PENDING_ENTRY','OPEN')").fetchone()
    return float(row["allocated"])


def _realized(db):
    row = db.execute("SELECT COALESCE(SUM(net_pnl),0) AS pnl FROM paper_trades WHERE status='CLOSED'").fetchone()
    return float(row["pnl"])


def on_actionable_signal(record: dict, path=None) -> dict | None:
    """Queue, but do not fill, a valid fresh BUY/SELL for its next 5m candle."""
    if not record:
        return None
    payload = record.get("payload") or {}
    classification = record.get("classification") or payload.get("classification")
    plan = payload.get("risk_plan") or {}
    if classification not in {"BUY", "SELL"} or record.get("status") not in {"ACTIVE", "TARGET_1_HIT"} or not plan.get("valid"):
        return None
    candle_stamp = pd.Timestamp(record["candle_timestamp"])
    close_stamp = pd.Timestamp(payload.get("candle_close_timestamp", candle_stamp + pd.Timedelta(minutes=5)))
    generated = pd.Timestamp(record.get("generated_at") or payload.get("generated_at"))
    if close_stamp.tzinfo is None or generated.tzinfo is None or generated < close_stamp:
        logger.info("Paper trade not queued for %s: analysis timestamp is invalid or precedes candle completion", record.get("signal_id"))
        return None
    age_minutes = (generated - close_stamp).total_seconds() / 60
    if age_minutes > 15:
        logger.info("Paper trade not queued for %s: analysis is stale by %.1f minutes", record.get("signal_id"), age_minutes)
        return None
    direction = "LONG" if classification == "BUY" else "SHORT"
    trigger = float(plan.get("trigger_price"))
    stop, target1, target2 = (float(plan[key]) for key in ("stop_loss", "target_1", "target_2"))
    levels_finite = all(math.isfinite(value) and value > 0 for value in (trigger, stop, target1, target2))
    correct_levels = stop < trigger < target1 <= target2 if direction == "LONG" else stop > trigger > target1 >= target2
    if not levels_finite or not correct_levels:
        logger.warning("Paper trade not queued for %s: risk-plan levels are inconsistent", record.get("signal_id"))
        return None
    timestamp = candle_stamp.isoformat()
    with _connect(path) as db:
        config = dict(db.execute("SELECT * FROM paper_settings WHERE singleton=1").fetchone())
        if not config["enabled"]:
            return None
        if db.execute("SELECT 1 FROM paper_trades WHERE signal_id=?", (record["signal_id"],)).fetchone():
            return None
        equity = float(config["starting_capital"]) + _realized(db)
        available = max(0.0, equity - _open_allocated(db))
        mode, value = config["sizing_mode"], float(config["sizing_value"])
        if mode == "fixed_shares":
            quantity = value
        else:
            dollars = value if mode == "fixed_capital" else equity * value / 100.0
            quantity = math.floor(min(dollars, available) / trigger * 1_000_000) / 1_000_000
        if quantity <= 0 or quantity * trigger > available:
            logger.info("Paper trade not queued for %s: insufficient available paper equity", record["signal_id"])
            return None
        trade_id = str(uuid4())
        entry_zone = plan.get("entry_zone") or {}
        if "lower" not in entry_zone or "upper" not in entry_zone:
            logger.warning("Paper trade not queued for %s: planned entry zone is missing", record.get("signal_id"))
            return None
        entry_lower, entry_upper = float(entry_zone["lower"]), float(entry_zone["upper"])
        if not math.isfinite(entry_lower) or not math.isfinite(entry_upper) or entry_lower <= 0 or entry_lower > trigger or entry_upper < trigger:
            logger.warning("Paper trade not queued for %s: entry zone is invalid", record.get("signal_id"))
            return None
        now = _now()
        db.execute("""INSERT INTO paper_trades
          (trade_id,signal_id,symbol,setup_type,direction,signal_timestamp,status,quantity,remaining_quantity,
           capital_allocated,entry_lower,entry_upper,stop_loss,target_1,target_2,risk_per_share,
           transaction_cost_bps,slippage_bps,created_at,updated_at)
          VALUES(?,?,?,?,?,?,'PENDING_ENTRY',?,?,?,?,?,?,?,?,?,?,?,?,?)""",
          (trade_id, record["signal_id"], "AMZN", payload.get("setup_type") or payload.get("market_regime", {}).get("classification", "SIGNAL"),
           direction, timestamp, quantity, quantity, quantity * trigger, entry_lower, entry_upper, stop, target1, target2,
           abs(trigger - stop), float(config["transaction_cost_bps"]), float(config["slippage_bps"]), now, now))
        _record_event(db, trade_id, "ENTRY_QUEUED", {"quantity": quantity, "reference_price": trigger,
                      "assumption": "next same-session completed candle open; only if open is inside planned entry zone"}, now)
        return _decode(db.execute("SELECT * FROM paper_trades WHERE trade_id=?", (trade_id,)).fetchone())


def process_completed_candles(candles, path=None, as_of=None) -> int:
    """Process genuine completed 5m OHLC rows chronologically; idempotent per trade/candle."""
    if candles is None or len(candles) == 0:
        return 0
    frame = candles.copy() if isinstance(candles, pd.DataFrame) else pd.DataFrame(candles)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if required - set(frame.columns):
        raise ValueError(f"Paper execution requires completed OHLCV columns: {sorted(required)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    cutoff = pd.Timestamp(as_of or datetime.now(timezone.utc))
    if cutoff.tzinfo is None:
        raise ValueError("Paper execution as_of must be timezone-aware")
    cutoff = cutoff.tz_convert("UTC")
    numeric = frame[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not all(math.isfinite(float(value)) for value in numeric.to_numpy().ravel()):
        raise ValueError("Paper execution candles must contain finite OHLCV values")
    if ((numeric.volume < 0) | (numeric.low <= 0) | (numeric.high < numeric[["open", "close"]].max(axis=1)) |
        (numeric.low > numeric[["open", "close"]].min(axis=1))).any():
        raise ValueError("Paper execution candle has invalid OHLCV relationships")
    frame[list(numeric.columns)] = numeric
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    changes = 0
    with _connect(path) as db:
        config = dict(db.execute("SELECT * FROM paper_settings WHERE singleton=1").fetchone())
        if not config["enabled"]:
            pending = db.execute("SELECT trade_id FROM paper_trades WHERE status='PENDING_ENTRY'").fetchall()
            for pending_trade in pending:
                db.execute("UPDATE paper_trades SET status='CANCELLED_PAUSED',exit_reason='PAUSED_BEFORE_ENTRY',updated_at=? WHERE trade_id=?",
                           (_now(), pending_trade["trade_id"]))
                _record_event(db, pending_trade["trade_id"], "CANCELLED_PAUSED", {"reason": "Paper trading is paused"})
        for row in frame.to_dict("records"):
            ts = pd.Timestamp(row["timestamp"])
            end = ts + pd.Timedelta(minutes=5)
            local = ts.tz_convert("America/New_York")
            close_at = session_close(local.date())
            local_end = end.tz_convert("America/New_York")
            if (end > cutoff or not is_trading_day(local.date()) or (local.hour, local.minute) < (9, 30) or
                (local_end.hour, local_end.minute) > (close_at.hour, close_at.minute)):
                continue
            stamp = ts.isoformat()
            trades = db.execute("SELECT * FROM paper_trades WHERE status IN ('PENDING_ENTRY','OPEN') ORDER BY signal_timestamp").fetchall()
            for trade in trades:
                if ts <= pd.Timestamp(trade["signal_timestamp"]):
                    continue
                trade_date = pd.Timestamp(trade["signal_timestamp"]).tz_convert("America/New_York").date()
                if local.date() != trade_date:
                    if trade["status"] == "PENDING_ENTRY":
                        db.execute("UPDATE paper_trades SET status='EXPIRED_NO_FILL',exit_reason='NO_NEXT_SESSION_ENTRY',updated_at=? WHERE trade_id=?", (_now(), trade["trade_id"]))
                        _record_event(db, trade["trade_id"], "EXPIRED_NO_FILL", {"reason": "No eligible next-candle entry in the signal session"}, stamp)
                        changes += 1
                    continue
                if trade["status"] == "PENDING_ENTRY":
                    signal_stamp = pd.Timestamp(trade["signal_timestamp"])
                    if ts != signal_stamp + pd.Timedelta(minutes=5):
                        db.execute("UPDATE paper_trades SET status='EXPIRED_NO_FILL',exit_reason='NEXT_CANDLE_MISSING',updated_at=? WHERE trade_id=?",
                                   (_now(), trade["trade_id"]))
                        _record_event(db, trade["trade_id"], "EXPIRED_NO_FILL", {"reason": "Immediate next 5-minute candle is missing; later entry is not substituted"}, stamp)
                        changes += 1
                        continue
                    lower, upper = float(trade["entry_lower"]), float(trade["entry_upper"])
                    zone = {"lower": lower, "upper": upper}
                    raw_entry = float(row["open"])
                    if not lower <= raw_entry <= upper:
                        db.execute("UPDATE paper_trades SET status='EXPIRED_NO_FILL',exit_reason='NEXT_OPEN_OUTSIDE_ENTRY_ZONE',updated_at=? WHERE trade_id=?", (_now(), trade["trade_id"]))
                        _record_event(db, trade["trade_id"], "EXPIRED_NO_FILL", {"next_open": raw_entry, "entry_zone": zone}, stamp)
                        changes += 1
                        continue
                    side = 1 if trade["direction"] == "LONG" else -1
                    entry_slip = raw_entry * float(trade["slippage_bps"]) / 10_000
                    fill = raw_entry + side * entry_slip
                    entry_cost = fill * float(trade["quantity"]) * float(trade["transaction_cost_bps"]) / 10_000
                    entry_slip_cost = abs(fill - raw_entry) * float(trade["quantity"])
                    db.execute("""UPDATE paper_trades SET status='OPEN',entry_timestamp=?,entry_price=?,entry_fill_price=?,entry_reference_price=?,
                      transaction_costs=transaction_costs+?,slippage_cost=slippage_cost+?,mark_price=?,mark_timestamp=?,updated_at=?
                      WHERE trade_id=? AND status='PENDING_ENTRY'""",
                      (stamp, raw_entry, fill, raw_entry, entry_cost, entry_slip_cost, float(row["close"]), stamp, _now(), trade["trade_id"]))
                    _record_event(db, trade["trade_id"], "ENTRY_FILLED", {"raw_open": raw_entry, "fill_price": fill,
                                  "quantity": float(trade["quantity"]), "entry_cost": entry_cost, "entry_slippage": entry_slip_cost}, stamp)
                    changes += 1
                    trade = db.execute("SELECT * FROM paper_trades WHERE trade_id=?", (trade["trade_id"],)).fetchone()
                if trade["status"] != "OPEN":
                    continue
                side = 1 if trade["direction"] == "LONG" else -1
                high, low, open_price, close_price = (float(row[key]) for key in ("high", "low", "open", "close"))
                stop = float(trade["stop_loss"]); target1 = float(trade["target_1"]); target2 = float(trade["target_2"])
                stop_hit = low <= stop if side == 1 else high >= stop
                target2_hit = high >= target2 if side == 1 else low <= target2
                if stop_hit:
                    raw_exit = min(open_price, stop) if side == 1 else max(open_price, stop)
                    _close_position(db, trade, raw_exit, stamp, "STOP_LOSS", config)
                    changes += 1
                    continue
                if target2_hit:
                    _close_position(db, trade, target2, stamp, "TARGET_2", config)
                    changes += 1
                    continue
                target1_hit = high >= target1 if side == 1 else low <= target1
                if target1_hit and float(trade["remaining_quantity"]) >= 0.000002:
                    current_events = db.execute("SELECT 1 FROM paper_journal WHERE trade_id=? AND event_type='TARGET_1_PARTIAL'", (trade["trade_id"],)).fetchone()
                    if not current_events:
                        partial = math.floor(float(trade["quantity"]) * .5 * 1_000_000) / 1_000_000
                        if partial > 0 and partial < float(trade["remaining_quantity"]):
                            _book_exit(db, trade, partial, target1, stamp, "TARGET_1_PARTIAL", config, close_trade=False)
                            changes += 1
                            trade = db.execute("SELECT * FROM paper_trades WHERE trade_id=?", (trade["trade_id"],)).fetchone()
                entry_time = pd.Timestamp(trade["entry_timestamp"])
                elapsed = (end - entry_time).total_seconds() / 60
                if elapsed >= MAX_HOLD_MINUTES:
                    _close_position(db, trade, close_price, stamp, "TIME_LIMIT", config)
                    changes += 1
                    continue
                if (local_end.hour, local_end.minute) >= (close_at.hour, close_at.minute):
                    _close_position(db, trade, close_price, stamp, "SESSION_END", config)
                    changes += 1
                    continue
                reference_entry = float(trade["entry_reference_price"] or trade["entry_price"])
                gross_unreal = (close_price - reference_entry) * float(trade["remaining_quantity"]) * side
                db.execute("UPDATE paper_trades SET mark_price=?,mark_timestamp=?,updated_at=? WHERE trade_id=? AND status='OPEN'",
                           (close_price, stamp, _now(), trade["trade_id"]))
                if gross_unreal:
                    _record_event(db, trade["trade_id"], "MARK_UPDATED", {"mark_price": close_price, "unrealized_gross_pnl": gross_unreal}, stamp)
        return changes


def _book_exit(db, trade, quantity, raw_exit, timestamp, reason, config, close_trade):
    side = 1 if trade["direction"] == "LONG" else -1
    raw_exit = float(raw_exit)
    cost_bps = float(trade["transaction_cost_bps"])
    slippage_bps = float(trade["slippage_bps"])
    slipped = raw_exit - side * raw_exit * slippage_bps / 10_000
    entry_reference = float(trade["entry_price"])
    gross = (raw_exit - entry_reference) * quantity * side
    fee = slipped * quantity * cost_bps / 10_000
    slip_cost = abs(slipped - raw_exit) * quantity
    gross_total = float(trade["gross_pnl"]) + gross
    costs_total = float(trade["transaction_costs"]) + fee
    slip_total = float(trade["slippage_cost"]) + slip_cost
    remaining = max(0.0, float(trade["remaining_quantity"]) - quantity)
    status = "CLOSED" if close_trade or remaining < 0.000001 else "OPEN"
    net = gross_total - costs_total - slip_total if status == "CLOSED" else None
    r_multiple = net / (float(trade["risk_per_share"]) * float(trade["quantity"])) if net is not None else None
    prior_realized = float(trade["realized_net_pnl"] or 0)
    if status == "CLOSED":
        realized_net = net
    else:
        # Entry costs/slippage are allocated by quantity across partial exits.
        entry_fill = float(trade["entry_fill_price"] or trade["entry_price"])
        entry_cost_total = entry_fill * float(trade["quantity"]) * cost_bps / 10_000
        entry_slippage_total = abs(entry_fill - entry_reference) * float(trade["quantity"])
        realized_net = prior_realized + gross - fee - slip_cost - (entry_cost_total + entry_slippage_total) * (quantity / float(trade["quantity"]))
    db.execute("""UPDATE paper_trades SET status=?,remaining_quantity=?,exit_price=?,exit_fill_price=?,exit_reference_price=?,exit_timestamp=?,exit_reason=?,
      gross_pnl=?,transaction_costs=?,slippage_cost=?,net_pnl=?,realized_net_pnl=?,r_multiple=?,mark_price=?,mark_timestamp=?,updated_at=? WHERE trade_id=?""",
      (status, remaining, raw_exit, slipped, raw_exit, timestamp, reason, gross_total, costs_total, slip_total, net, realized_net, r_multiple,
       slipped, timestamp, _now(), trade["trade_id"]))
    _record_event(db, trade["trade_id"], reason, {"raw_exit": raw_exit, "fill_price": slipped,
                  "quantity": quantity, "gross_pnl_increment": gross, "transaction_cost": fee,
                  "slippage_cost": slip_cost, "remaining_quantity": remaining, "status": status}, timestamp)


def _close_position(db, trade, raw_exit, timestamp, reason, config):
    _book_exit(db, trade, float(trade["remaining_quantity"]), raw_exit, timestamp, reason, config, close_trade=True)


def close_trade(trade_id: str, price: float, timestamp: str, path=None):
    if not math.isfinite(price) or price <= 0:
        raise ValueError("A finite positive completed-candle mark price is required")
    with _connect(path) as db:
        trade = db.execute("SELECT * FROM paper_trades WHERE trade_id=?", (trade_id,)).fetchone()
        if not trade:
            raise KeyError(trade_id)
        if trade["status"] != "OPEN":
            raise ValueError("Only open paper trades can be closed")
        if pd.Timestamp(timestamp) <= pd.Timestamp(trade["entry_timestamp"]):
            raise ValueError("Close mark must be later than the entry timestamp")
        config = dict(db.execute("SELECT * FROM paper_settings WHERE singleton=1").fetchone())
        _close_position(db, trade, price, timestamp, "MANUAL", config)
        return _decode(db.execute("SELECT * FROM paper_trades WHERE trade_id=?", (trade_id,)).fetchone())


def dashboard(mark_price=None, mark_timestamp=None, path=None):
    with _connect(path) as db:
        config = dict(db.execute("SELECT * FROM paper_settings WHERE singleton=1").fetchone())
        trades = [dict(row) for row in db.execute("SELECT * FROM paper_trades ORDER BY created_at DESC").fetchall()]
        for trade in trades:
            if trade["status"] == "OPEN" and mark_price is not None and mark_timestamp and pd.Timestamp(mark_timestamp) >= pd.Timestamp(trade["entry_timestamp"]):
                trade["mark_price"] = float(mark_price); trade["mark_timestamp"] = mark_timestamp
        closed = [t for t in trades if t["status"] == "CLOSED"]
        open_trades = [t for t in trades if t["status"] == "OPEN"]
        realized = sum(float(t["realized_net_pnl"] or 0) for t in trades)
        unrealized = 0.0
        for trade in open_trades:
            mark = trade.get("mark_price")
            if mark is None:
                continue
            side = 1 if trade["direction"] == "LONG" else -1
            reference_entry = float(trade["entry_price"])
            gross_mark = (float(mark) - reference_entry) * float(trade["remaining_quantity"]) * side
            exit_estimate = float(mark) * float(trade["remaining_quantity"]) * (
                float(trade["transaction_cost_bps"]) + float(trade["slippage_bps"])) / 10_000
            entry_fill = float(trade["entry_fill_price"] or trade["entry_price"])
            entry_cost_remaining = entry_fill * float(trade["quantity"]) * float(trade["transaction_cost_bps"]) / 10_000 * float(trade["remaining_quantity"]) / float(trade["quantity"])
            entry_slippage_remaining = abs(entry_fill - reference_entry) * float(trade["remaining_quantity"])
            unrealized += gross_mark - entry_cost_remaining - entry_slippage_remaining - exit_estimate
            trade["unrealized_net_pnl"] = gross_mark - entry_cost_remaining - entry_slippage_remaining - exit_estimate
        wins = [float(t["net_pnl"]) for t in closed if float(t["net_pnl"]) > 0]
        losses = [float(t["net_pnl"]) for t in closed if float(t["net_pnl"]) < 0]
        gross_profit = sum(wins); gross_loss = abs(sum(losses))
        equity = float(config["starting_capital"]) + realized + unrealized
        closed_chron = sorted(closed, key=lambda t: (t["exit_timestamp"] or "", t["trade_id"]))
        curve = [{"timestamp": "START", "equity": float(config["starting_capital"])}]
        running = float(config["starting_capital"]); peak = running; max_dd = 0.0
        for trade in closed_chron:
            running += float(trade["net_pnl"] or 0); peak = max(peak, running)
            max_dd = max(max_dd, (peak - running) / peak if peak else 0)
            curve.append({"timestamp": trade["exit_timestamp"], "equity": running})
        closed_n = len(closed)
        metrics = {"realized_pnl": realized, "unrealized_pnl": unrealized, "net_pnl": realized + unrealized,
                   "equity": equity, "gross_profit": sum(max(0, float(t["gross_pnl"])) for t in closed),
                   "gross_loss": abs(sum(min(0, float(t["gross_pnl"])) for t in closed)),
                   "win_rate": len(wins) / closed_n if closed_n else None,
                   "profit_factor": sum(wins) / abs(sum(losses)) if losses else None,
                   "average_win": sum(wins) / len(wins) if wins else None,
                   "average_loss": sum(losses) / len(losses) if losses else None,
                   "maximum_drawdown": max_dd, "average_r_multiple": (sum(float(t["r_multiple"]) for t in closed if t["r_multiple"] is not None) / sum(t["r_multiple"] is not None for t in closed)) if any(t["r_multiple"] is not None for t in closed) else None,
                   "total_trades": closed_n, "winning_trades": len(wins), "losing_trades": len(losses),
                   "open_count": len(open_trades), "pending_count": sum(t["status"] == "PENDING_ENTRY" for t in trades),
                   "mark_timestamp": mark_timestamp, "mark_is_last_known": bool(open_trades and mark_timestamp)}
        config["enabled"] = bool(config["enabled"])
        journal = [dict(row) for row in db.execute("SELECT * FROM paper_journal ORDER BY journal_id DESC LIMIT 200").fetchall()]
        for event in journal:
            event["details"] = json.loads(event["details"])
        return {"disclaimer": "PAPER TRADING — SIMULATED, NOT REAL BROKER EXECUTION", "settings": config,
                "metrics": metrics, "open_trades": open_trades, "closed_trades": closed,
                "pending_trades": [t for t in trades if t["status"] == "PENDING_ENTRY"],
                "equity_curve": curve, "journal": journal}


def replay_historical(candles, signals, *, starting_capital=100_000, sizing_mode="fixed_shares",
                      sizing_value=1, transaction_cost_bps=10, slippage_bps=2):
    """Replay supplied persisted signals over genuine completed candles in an isolated database."""
    frame = candles.copy() if isinstance(candles, pd.DataFrame) else pd.DataFrame(candles)
    if frame.empty:
        return {"mode": "HISTORICAL_RESEARCH_ONLY", "signal_count": len(signals), "actionable_signals": 0,
                "trade_count": 0, "metrics": None, "reason": "No historical candles were supplied."}
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    index_by_stamp = {stamp.isoformat(): index for index, stamp in enumerate(frame.timestamp)}
    grouped = {}
    actionable = 0
    for record in signals:
        if record.get("classification") not in {"BUY", "SELL"}:
            continue
        actionable += 1
        stamp = pd.Timestamp(record["candle_timestamp"]).tz_convert("UTC").isoformat()
        if stamp in index_by_stamp:
            grouped.setdefault(index_by_stamp[stamp], []).append({**record, "status": "ACTIVE"})
    if not grouped:
        return {"mode": "HISTORICAL_RESEARCH_ONLY", "period_start": frame.timestamp.iloc[0].isoformat(),
                "period_end": frame.timestamp.iloc[-1].isoformat(), "signal_count": len(signals),
                "actionable_signals": actionable, "trade_count": 0, "pending_count": 0, "open_count": 0,
                "metrics": {"starting_capital": float(starting_capital), "realized_pnl": 0.0,
                            "unrealized_pnl": 0.0, "net_pnl": 0.0, "gross_profit": 0.0,
                            "gross_loss": 0.0, "win_rate": None, "profit_factor": None,
                            "maximum_drawdown": 0.0, "total_trades": 0},
                "results_by_setup_or_regime": {},
                "limitations": ["No stored actionable signal exists in the supplied period, so no paper fills or performance metrics can be inferred.",
                                "No synthetic signals or trades were created."]}
    with tempfile.TemporaryDirectory(prefix="amzn-paper-replay-") as temp:
        path = Path(temp) / "research-only.sqlite3"
        update_settings(enabled=True, starting_capital=starting_capital, sizing_mode=sizing_mode,
                        sizing_value=sizing_value, transaction_cost_bps=transaction_cost_bps,
                        slippage_bps=slippage_bps, path=path)
        cursor = 0
        for index in sorted(grouped):
            if index >= cursor:
                segment = frame.iloc[cursor:index + 1]
                if not segment.empty:
                    process_completed_candles(segment, path, as_of=segment.timestamp.iloc[-1] + pd.Timedelta(minutes=5))
                cursor = index + 1
            for record in grouped[index]:
                on_actionable_signal(record, path)
        if cursor < len(frame):
            segment = frame.iloc[cursor:]
            process_completed_candles(segment, path, as_of=segment.timestamp.iloc[-1] + pd.Timedelta(minutes=5))
        final = dashboard(float(frame.close.iloc[-1]), frame.timestamp.iloc[-1].isoformat(), path)
    setup_results = {}
    for trade in final["closed_trades"]:
        key = trade["setup_type"] or "UNCLASSIFIED"
        item = setup_results.setdefault(key, {"trades": 0, "net_pnl": 0.0, "wins": 0})
        item["trades"] += 1
        item["net_pnl"] += float(trade["net_pnl"] or 0)
        item["wins"] += int(float(trade["net_pnl"] or 0) > 0)
    return {"mode": "HISTORICAL_RESEARCH_ONLY", "period_start": frame.timestamp.iloc[0].isoformat(),
            "period_end": frame.timestamp.iloc[-1].isoformat(), "signal_count": len(signals),
            "actionable_signals": actionable, "trade_count": len(final["closed_trades"]),
            "pending_count": final["metrics"]["pending_count"], "open_count": final["metrics"]["open_count"],
            "metrics": final["metrics"], "results_by_setup_or_regime": setup_results,
            "limitations": ["Uses only stored analysis records supplied to the replay; it does not synthesize new decisions.",
                            "No calibrated ML confidence or hypothetical forecast return is treated as a paper fill.",
                            "Research account data is isolated in a temporary database and never mixed into the paper ledger."]}
