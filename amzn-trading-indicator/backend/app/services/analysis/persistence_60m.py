"""SQLite persistence and lifecycle outcome resolution for 60-Minute Trading Decisions."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd

from backend.app.services.analysis.config import ANALYSIS_VERSION, BAR_MINUTES, ROUND_TRIP_COST_BPS, TIMEZONE
from src.forecasts.calendar import is_trading_day, session_close
from src.forecasts.store import connect

logger = logging.getLogger(__name__)
ET = ZoneInfo(TIMEZONE)


def ensure_60m_tables(db) -> None:
    """Ensure analysis_60m_signals table exists."""
    db.executescript("""
      CREATE TABLE IF NOT EXISTS analysis_60m_signals (
        signal_id TEXT PRIMARY KEY,
        candle_timestamp TEXT NOT NULL,
        generated_at TEXT NOT NULL,
        analysis_version TEXT NOT NULL,
        decision TEXT NOT NULL,
        directional_bias TEXT NOT NULL,
        directional_score REAL NOT NULL,
        horizon_minutes INTEGER NOT NULL DEFAULT 60,
        horizon_bars INTEGER NOT NULL DEFAULT 12,
        trigger_price REAL NOT NULL,
        entry_zone_lower REAL,
        entry_zone_upper REAL,
        stop_loss REAL,
        target_1 REAL,
        target_2 REAL,
        risk_reward_ratio REAL,
        payload TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'NEW',
        outcome_payload TEXT,
        gross_return_pct REAL,
        net_return_pct REAL,
        resolved_at TEXT,
        UNIQUE(candle_timestamp, analysis_version)
      );
      CREATE INDEX IF NOT EXISTS ix_analysis_60m_signals_time ON analysis_60m_signals(candle_timestamp DESC);
      CREATE INDEX IF NOT EXISTS ix_analysis_60m_signals_status ON analysis_60m_signals(status, candle_timestamp);
    """)


def _decode_60m_row(row) -> dict | None:
    if row is None:
        return None
    item = dict(row)
    item["payload"] = json.loads(item["payload"]) if item.get("payload") else {}
    item["outcome"] = json.loads(item["outcome_payload"]) if item.get("outcome_payload") else None
    return item


def save_60m_decision(payload: dict, db_path: str | Path | None = None) -> dict:
    """Persist a 60-minute trading decision idempotently."""
    candle_ts = payload["evaluation_candle_timestamp"]
    version = payload.get("analysis_version", ANALYSIS_VERSION)
    decision = payload["decision"]
    bias = payload.get("directional_bias", "NEUTRAL")
    score = float(payload.get("directional_score", 0.0))
    gen_at = payload.get("decision_timestamp", datetime.now(timezone.utc).isoformat())

    risk = payload.get("risk_plan", {})
    trigger = float(risk.get("trigger_price", 0.0))
    entry_zone = risk.get("entry_zone", {})
    lower = float(entry_zone.get("lower", trigger)) if entry_zone else trigger
    upper = float(entry_zone.get("upper", trigger)) if entry_zone else trigger
    stop = float(risk["stop_loss"]) if risk.get("stop_loss") is not None else None
    t1 = float(risk["target_1"]) if risk.get("target_1") is not None else None
    t2 = float(risk["target_2"]) if risk.get("target_2") is not None else None
    rr = float(risk["risk_reward_ratio"]) if risk.get("risk_reward_ratio") is not None else None

    # Status: ACTIVE for actionable entries, else HOLD or WAIT
    if decision in ("BUY", "SELL") and risk.get("valid"):
        initial_status = "ACTIVE"
    elif decision == "WAIT FOR CONFIRMATION":
        initial_status = "WAIT"
    else:
        initial_status = "HOLD"

    with connect(db_path) as db:
        ensure_60m_tables(db)
        existing = db.execute(
            "SELECT signal_id, status FROM analysis_60m_signals WHERE candle_timestamp = ? AND analysis_version = ?",
            (candle_ts, version),
        ).fetchone()

        sig_id = existing[0] if existing else str(uuid4())
        status = existing[1] if (existing and existing[1] not in ("NEW", "ACTIVE")) else initial_status

        db.execute(
            """INSERT INTO analysis_60m_signals (
                signal_id, candle_timestamp, generated_at, analysis_version, decision,
                directional_bias, directional_score, horizon_minutes, horizon_bars,
                trigger_price, entry_zone_lower, entry_zone_upper, stop_loss, target_1,
                target_2, risk_reward_ratio, payload, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 60, 12, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(candle_timestamp, analysis_version) DO UPDATE SET
                generated_at = excluded.generated_at,
                decision = excluded.decision,
                directional_bias = excluded.directional_bias,
                directional_score = excluded.directional_score,
                trigger_price = excluded.trigger_price,
                entry_zone_lower = excluded.entry_zone_lower,
                entry_zone_upper = excluded.entry_zone_upper,
                stop_loss = excluded.stop_loss,
                target_1 = excluded.target_1,
                target_2 = excluded.target_2,
                risk_reward_ratio = excluded.risk_reward_ratio,
                payload = excluded.payload""",
            (
                sig_id, candle_ts, gen_at, version, decision,
                bias, score, trigger, lower, upper, stop, t1,
                t2, rr, json.dumps(payload, default=str), status,
            ),
        )

        row = db.execute(
            "SELECT * FROM analysis_60m_signals WHERE signal_id = ?", (sig_id,)
        ).fetchone()
        return _decode_60m_row(row)


def get_latest_60m_decision(db_path: str | Path | None = None) -> dict | None:
    """Retrieve the most recent 60-minute decision."""
    with connect(db_path) as db:
        ensure_60m_tables(db)
        row = db.execute(
            "SELECT * FROM analysis_60m_signals ORDER BY candle_timestamp DESC LIMIT 1"
        ).fetchone()
        return _decode_60m_row(row)


def get_active_60m_signal(db_path: str | Path | None = None) -> dict | None:
    """Retrieve currently active actionable 60m trade signal."""
    with connect(db_path) as db:
        ensure_60m_tables(db)
        row = db.execute(
            "SELECT * FROM analysis_60m_signals WHERE status IN ('ACTIVE', 'TARGET_1_HIT') ORDER BY candle_timestamp DESC LIMIT 1"
        ).fetchone()
        return _decode_60m_row(row)


def list_60m_signals(limit: int = 50, offset: int = 0, status: str | None = None, db_path: str | Path | None = None) -> list[dict]:
    """List paginated 60-minute decisions."""
    with connect(db_path) as db:
        ensure_60m_tables(db)
        if status:
            rows = db.execute(
                "SELECT * FROM analysis_60m_signals WHERE status = ? ORDER BY candle_timestamp DESC LIMIT ? OFFSET ?",
                (status, limit, offset),
            ).fetchall()
        else:
            rows = db.execute(
                "SELECT * FROM analysis_60m_signals ORDER BY candle_timestamp DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [_decode_60m_row(r) for r in rows if r]


def resolve_60m_signals(candles: pd.DataFrame, db_path: str | Path | None = None) -> dict:
    """Resolve active 60-minute signals against subsequent completed 5-minute candles.

    Lifecycle:
      - ACTIVE -> TARGET_1_HIT -> TARGET_2_HIT -> STOPPED_OUT -> EXPIRED (12 bars / 60m)
    """
    if candles.empty:
        return {"resolved": [], "still_active": []}

    frame = candles.sort_values("timestamp").reset_index(drop=True)
    resolved_ids: list[str] = []
    active_ids: list[str] = []

    with connect(db_path) as db:
        ensure_60m_tables(db)
        active_records = db.execute(
            "SELECT * FROM analysis_60m_signals WHERE status IN ('ACTIVE', 'TARGET_1_HIT') ORDER BY candle_timestamp ASC"
        ).fetchall()

        for raw_record in active_records:
            rec = _decode_60m_row(raw_record)
            sig_id = rec["signal_id"]
            direction = rec["decision"]
            entry_zone = {
                "lower": rec.get("entry_zone_lower", rec["trigger_price"]),
                "upper": rec.get("entry_zone_upper", rec["trigger_price"]),
            }
            stop = float(rec["stop_loss"]) if rec.get("stop_loss") else None
            t1 = float(rec["target_1"]) if rec.get("target_1") else None
            t2 = float(rec["target_2"]) if rec.get("target_2") else None

            if not stop or not t2:
                # Invalid setup -> EXPIRED
                db.execute("UPDATE analysis_60m_signals SET status = 'EXPIRED' WHERE signal_id = ?", (sig_id,))
                resolved_ids.append(sig_id)
                continue

            sig_ts = pd.Timestamp(rec["candle_timestamp"]).tz_convert("UTC")
            future_bars = frame.loc[pd.to_datetime(frame["timestamp"], utc=True) > sig_ts]

            if future_bars.empty:
                active_ids.append(sig_id)
                continue

            entry_price: float | None = None
            bars_held = 0
            is_long = direction == "BUY"
            closed_event: str | None = None
            exit_price: float | None = None
            exit_ts: str | None = None

            for row in future_bars.itertuples(index=False):
                b_ts = pd.Timestamp(row.timestamp).tz_convert("UTC")
                b_local = b_ts.astimezone(ET)

                # Ensure same session
                if b_local.date() != sig_ts.astimezone(ET).date():
                    closed_event = "CLOSED"
                    exit_price = float(row.close)
                    exit_ts = b_ts.isoformat()
                    break

                # 1. Fill check on first subsequent candle
                if entry_price is None:
                    candidate_open = float(row.open)
                    if not (float(entry_zone["lower"]) <= candidate_open <= float(entry_zone["upper"])):
                        # Fill missed outside planned entry zone -> EXPIRED
                        closed_event = "EXPIRED"
                        exit_price = candidate_open
                        exit_ts = b_ts.isoformat()
                        break
                    entry_price = candidate_open

                bars_held += 1

                # 2. Check stop-loss and targets
                b_high = float(row.high)
                b_low = float(row.low)

                hit_stop = (b_low <= stop) if is_long else (b_high >= stop)
                hit_t2 = (b_high >= t2) if is_long else (b_low <= t2)
                hit_t1 = (b_high >= t1) if is_long else (b_low <= t1)

                if hit_stop:
                    closed_event = "STOPPED_OUT"
                    exit_price = stop
                    exit_ts = b_ts.isoformat()
                    break
                elif hit_t2:
                    closed_event = "TARGET_2_HIT"
                    exit_price = t2
                    exit_ts = b_ts.isoformat()
                    break

                if hit_t1 and rec["status"] == "ACTIVE":
                    db.execute("UPDATE analysis_60m_signals SET status = 'TARGET_1_HIT' WHERE signal_id = ?", (sig_id,))
                    rec["status"] = "TARGET_1_HIT"

                # 3. 60-Minute Horizon Limit (12 bars)
                if bars_held >= 12:
                    closed_event = "EXPIRED"
                    exit_price = float(row.close)
                    exit_ts = b_ts.isoformat()
                    break

                # 4. Session close check
                if b_local.time() >= session_close(b_local.date()):
                    closed_event = "CLOSED"
                    exit_price = float(row.close)
                    exit_ts = b_ts.isoformat()
                    break

            if closed_event:
                if entry_price and exit_price:
                    raw_ret = ((exit_price - entry_price) / entry_price * 100.0) * (1 if is_long else -1)
                    net_ret = raw_ret - (ROUND_TRIP_COST_BPS / 100.0)
                else:
                    raw_ret = 0.0
                    net_ret = 0.0
                outcome = {
                    "status": closed_event,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "exit_timestamp": exit_ts,
                    "bars_held": bars_held,
                    "gross_return_pct": round(raw_ret, 4),
                    "net_return_pct": round(net_ret, 4),
                    "assumed_cost_bps": ROUND_TRIP_COST_BPS if entry_price else 0.0,
                    "resolved_at": datetime.now(timezone.utc).isoformat(),
                }
                db.execute(
                    """UPDATE analysis_60m_signals
                       SET status = ?, outcome_payload = ?, gross_return_pct = ?, net_return_pct = ?, resolved_at = ?
                       WHERE signal_id = ?""",
                    (closed_event, json.dumps(outcome, default=str), round(raw_ret, 4), round(net_ret, 4), outcome["resolved_at"], sig_id),
                )
                resolved_ids.append(sig_id)
            else:
                active_ids.append(sig_id)

    return {"resolved_count": len(resolved_ids), "still_active_count": len(active_ids)}


def get_60m_performance_summary(db_path: str | Path | None = None) -> dict:
    """Calculate summary metrics for 60-minute decisions and hypothetical outcomes."""
    with connect(db_path) as db:
        ensure_60m_tables(db)
        total_decisions = db.execute("SELECT count(*) FROM analysis_60m_signals").fetchone()[0]
        buy_count = db.execute("SELECT count(*) FROM analysis_60m_signals WHERE decision = 'BUY'").fetchone()[0]
        sell_count = db.execute("SELECT count(*) FROM analysis_60m_signals WHERE decision = 'SELL'").fetchone()[0]
        hold_count = db.execute("SELECT count(*) FROM analysis_60m_signals WHERE decision = 'HOLD'").fetchone()[0]
        wait_count = db.execute("SELECT count(*) FROM analysis_60m_signals WHERE decision = 'WAIT FOR CONFIRMATION'").fetchone()[0]

        resolved_rows = db.execute(
            "SELECT net_return_pct, gross_return_pct, status FROM analysis_60m_signals WHERE net_return_pct IS NOT NULL"
        ).fetchall()

        resolved_count = len(resolved_rows)
        if resolved_count == 0:
            return {
                "has_data": False,
                "total_decisions": total_decisions,
                "counts": {"buy": buy_count, "sell": sell_count, "hold": hold_count, "wait": wait_count},
                "resolved_count": 0,
                "win_rate": None,
                "average_net_return_pct": None,
                "gross_pnl_pct": None,
                "net_pnl_pct": None,
                "profit_factor": None,
            }

        returns = [r[0] for r in resolved_rows]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]

        win_rate = len(wins) / resolved_count if resolved_count > 0 else 0.0
        avg_ret = sum(returns) / resolved_count
        net_pnl = sum(returns)
        gross_pnl = sum(r[1] for r in resolved_rows)

        sum_win = sum(wins)
        sum_loss = abs(sum(losses))
        profit_factor = round(sum_win / sum_loss, 2) if sum_loss > 0 else (99.0 if sum_win > 0 else 0.0)

        return {
            "has_data": True,
            "total_decisions": total_decisions,
            "counts": {"buy": buy_count, "sell": sell_count, "hold": hold_count, "wait": wait_count},
            "resolved_count": resolved_count,
            "win_rate": round(win_rate, 4),
            "average_net_return_pct": round(avg_ret, 4),
            "gross_pnl_pct": round(gross_pnl, 4),
            "net_pnl_pct": round(net_pnl, 4),
            "profit_factor": profit_factor,
            "cost_assumption_bps": ROUND_TRIP_COST_BPS,
        }
