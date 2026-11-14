"""SQLite persistence for hypothetical analysis signals and lifecycle events."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from src.forecasts.store import connect

def save_signal(payload: dict, db_path: str | Path | None = None) -> dict:
    stamp = datetime.now(timezone.utc).isoformat()
    with connect(db_path) as db:
        db.execute("INSERT OR IGNORE INTO analysis_signals(signal_id,candle_timestamp,generated_at,analysis_version,classification,directional_bias,final_score,payload,status) VALUES(?,?,?,?,?,?,?,?,?)",
                   (str(uuid4()), payload["candle_timestamp"], payload["generated_at"], payload["analysis_version"], payload["classification"], payload["directional_bias"], payload.get("final_decision_score"), json.dumps(payload, default=str), "ACTIVE" if payload["classification"] in {"BUY","SELL"} else "NOT_ACTIONABLE"))
        row = db.execute("SELECT * FROM analysis_signals WHERE candle_timestamp=? AND analysis_version=?", (payload["candle_timestamp"], payload["analysis_version"])).fetchone()
        return _decode(row)

def _decode(row):
    if row is None: return None
    item = dict(row); item["payload"] = json.loads(item["payload"])
    item["outcome"] = json.loads(item["outcome_payload"]) if item["outcome_payload"] else None
    return item

def get_signal(signal_id: str, db_path=None):
    with connect(db_path) as db: return _decode(db.execute("SELECT * FROM analysis_signals WHERE signal_id=?", (signal_id,)).fetchone())

def get_signal_for_candle(candle_timestamp: str, analysis_version: str, db_path=None):
    """Return the immutable analysis already persisted for one candle/version."""
    with connect(db_path) as db:
        row = db.execute("SELECT * FROM analysis_signals WHERE candle_timestamp=? AND analysis_version=?",
                         (candle_timestamp, analysis_version)).fetchone()
        return _decode(row)

def list_signals(start: str | None=None, end: str | None=None, status: str | None=None, limit=100, offset=0, db_path=None):
    clauses=[]; args=[]
    if start: clauses.append("candle_timestamp>=?"); args.append(start)
    if end: clauses.append("candle_timestamp<=?"); args.append(end)
    if status: clauses.append("status=?"); args.append(status)
    sql="SELECT * FROM analysis_signals" + (" WHERE " + " AND ".join(clauses) if clauses else "") + " ORDER BY candle_timestamp DESC LIMIT ? OFFSET ?"
    with connect(db_path) as db: return [_decode(r) for r in db.execute(sql, (*args,limit,offset)).fetchall()]

def active_signals(db_path=None):
    with connect(db_path) as db: return [_decode(r) for r in db.execute("SELECT * FROM analysis_signals WHERE status IN ('ACTIVE','TARGET_1_HIT') ORDER BY candle_timestamp").fetchall()]

def latest_signal(db_path=None):
    with connect(db_path) as db: return _decode(db.execute("SELECT * FROM analysis_signals ORDER BY generated_at DESC LIMIT 1").fetchone())

def append_event(signal_id: str, kind: str, details: dict, db_path=None):
    with connect(db_path) as db:
        stamp=datetime.now(timezone.utc).isoformat(); body=json.dumps(details,default=str)
        db.execute("""INSERT INTO analysis_signal_events(signal_id,timestamp,event_type,details)
          SELECT ?,?,?,? WHERE NOT EXISTS(SELECT 1 FROM analysis_signal_events WHERE signal_id=? AND event_type=?)""",
          (signal_id,stamp,kind,body,signal_id,kind))

def set_signal_status(signal_id: str, status: str, details: dict, gross=None, net=None, db_path=None):
    with connect(db_path) as db:
        changed=db.execute("UPDATE analysis_signals SET status=?,outcome_payload=?,gross_return_pct=?,net_return_pct=? WHERE signal_id=? AND status IN ('ACTIVE','TARGET_1_HIT')",(status,json.dumps(details,default=str),gross,net,signal_id)).rowcount
        if changed==0:
            if db.execute("SELECT 1 FROM analysis_signals WHERE signal_id=?",(signal_id,)).fetchone() is None: raise KeyError(signal_id)
            return False
        db.execute("INSERT INTO analysis_signal_events(signal_id,timestamp,event_type,details) VALUES(?,?,?,?)",(signal_id,datetime.now(timezone.utc).isoformat(),status,json.dumps(details,default=str)))
        return True

def signal_events(signal_id: str, db_path=None):
    with connect(db_path) as db:
        return [dict(r) for r in db.execute("SELECT * FROM analysis_signal_events WHERE signal_id=? ORDER BY event_id",(signal_id,)).fetchall()]
