"""Persistent status for long-running data and model operations."""
from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable
from uuid import uuid4

from src.forecasts.store import DB_PATH, append_log, connect

_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="amzn-operation")
_LOCKS = {"data_refresh": Lock(), "model_training": Lock()}
_INITIALIZED = False


def _connect():
    global _INITIALIZED
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Central migration/backup path runs before this legacy helper changes schema.
    with connect(DB_PATH):
        pass
    db = sqlite3.connect(DB_PATH, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("""CREATE TABLE IF NOT EXISTS operations (
        operation_id TEXT PRIMARY KEY, kind TEXT NOT NULL, evaluation_date TEXT,
        status TEXT NOT NULL, stage TEXT NOT NULL, progress REAL NOT NULL,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL, result TEXT, error TEXT)""")
    db.execute("""CREATE TABLE IF NOT EXISTS operation_events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT, operation_id TEXT NOT NULL,
        timestamp TEXT NOT NULL, stage TEXT NOT NULL, status TEXT NOT NULL,
        progress REAL NOT NULL, message TEXT, data TEXT)""")
    if not _INITIALIZED:
        db.execute("UPDATE operations SET status='FAILED',stage='interrupted',error='Backend restarted during operation',updated_at=? WHERE status='RUNNING'",
                   (datetime.now(timezone.utc).isoformat(),))
        _INITIALIZED = True
    return db


def _update(operation_id: str, *, status: str | None = None, stage: str | None = None,
            progress: float | None = None, result: Any = None, error: str | None = None) -> None:
    now = datetime.now(timezone.utc).isoformat()
    db = _connect()
    row = db.execute("SELECT * FROM operations WHERE operation_id=?", (operation_id,)).fetchone()
    if row is None:
        db.close()
        return
    new_status, new_stage = status or row["status"], stage or row["stage"]
    new_progress = progress if progress is not None else row["progress"]
    db.execute("UPDATE operations SET status=?,stage=?,progress=?,updated_at=?,result=?,error=? WHERE operation_id=?",
               (new_status, new_stage, new_progress,
                now, json.dumps(result, default=str) if result is not None else row["result"], error, operation_id))
    db.execute("INSERT INTO operation_events(operation_id,timestamp,stage,status,progress,message,data) VALUES(?,?,?,?,?,?,?)",
               (operation_id, now, new_stage, new_status, float(new_progress), error or new_stage,
                json.dumps(result, default=str) if result is not None else None))
    db.commit(); db.close()
    if isinstance(result, dict):
        keys = ("model_version", "evaluation_date", "dataset_rows", "feature_count", "added", "fetched",
                "rows", "latest_timestamp", "processed_path", "source")
        summary = {key: result[key] for key in keys if key in result}
    else:
        summary = {"stage": stage, "progress": progress}
    append_log(operation_id, row["evaluation_date"] or datetime.now().date().isoformat(), "operations",
               row["kind"], {"operation_id": operation_id}, summary,
               None, None, "background_operation", status or row["status"], error, path=DB_PATH)


def get_operation(operation_id: str) -> dict[str, Any] | None:
    db = _connect(); row = db.execute("SELECT * FROM operations WHERE operation_id=?", (operation_id,)).fetchone(); db.close()
    if row is None:
        return None
    item = dict(row)
    item["result"] = json.loads(item["result"]) if item["result"] else None
    return item


def list_operations(kind: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    db = _connect()
    rows = db.execute("SELECT * FROM operations" + (" WHERE kind=?" if kind else "") + " ORDER BY created_at DESC LIMIT ?",
                      ((kind, limit) if kind else (limit,))).fetchall(); db.close()
    out=[]
    for row in rows:
        item=dict(row); item["result"]=json.loads(item["result"]) if item["result"] else None; out.append(item)
    return out


def list_operation_events(operation_id: str, after_id: int = 0) -> list[dict[str, Any]]:
    db = _connect()
    rows = db.execute("SELECT * FROM operation_events WHERE operation_id=? AND event_id>? ORDER BY event_id",
                      (operation_id, int(after_id))).fetchall()
    db.close()
    return [dict(row) for row in rows]


def start_operation(kind: str, evaluation_date: str | None, work: Callable[[Callable[[str, float], None]], Any]) -> dict[str, Any]:
    lock = _LOCKS[kind]
    if not lock.acquire(blocking=False):
        raise RuntimeError(f"A {kind.replace('_', ' ')} operation is already running")
    operation_id = str(uuid4()); now = datetime.now(timezone.utc).isoformat()
    db = _connect()
    db.execute("INSERT INTO operations VALUES(?,?,?,?,?,?,?,?,?,?)",
               (operation_id, kind, evaluation_date, "RUNNING", "starting", 0.0, now, now, None, None))
    db.commit(); db.close()
    _update(operation_id, status="RUNNING", stage="starting", progress=0)

    def run():
        def progress(stage: str, value: float):
            _update(operation_id, status="RUNNING", stage=stage, progress=max(0.0, min(float(value), 0.99)))
        try:
            result = work(progress)
            _update(operation_id, status="COMPLETED", stage="completed", progress=1.0, result=result)
        except Exception as exc:
            detail = str(exc)
            for secret_name in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"):
                import os
                secret = os.getenv(secret_name)
                if secret: detail = detail.replace(secret, "[redacted]")
            _update(operation_id, status="FAILED", stage="failed", error=detail)
        finally:
            lock.release()
    _POOL.submit(run)
    return get_operation(operation_id) or {"operation_id": operation_id, "status": "RUNNING"}
