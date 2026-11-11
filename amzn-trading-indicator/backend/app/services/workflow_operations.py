"""Persistent, event-streamable operations shared by APIs and scheduler."""
from __future__ import annotations

import json
import os
import socket
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable
from uuid import uuid4

from src.forecasts.store import connect

_locks_guard = Lock()
_locks: dict[str, Lock] = {}


def _record_event(operation_id: str, stage: str, status: str, progress: float,
                  message: str | None = None, data: Any = None) -> None:
    with connect() as db:
        _insert_event(db, operation_id, stage, status, progress, message, data)


def _insert_event(db, operation_id: str, stage: str, status: str, progress: float,
                  message: str | None = None, data: Any = None) -> None:
    db.execute("INSERT INTO workflow_events(operation_id,timestamp,stage,status,progress,message,data) VALUES(?,?,?,?,?,?,?)",
               (operation_id, datetime.now(timezone.utc).isoformat(), stage, status,
                min(1.0, max(0.0, float(progress))), message,
                json.dumps(data, default=str) if data is not None else None))


def get_workflow(operation_id: str) -> dict[str, Any] | None:
    with connect() as db:
        row = db.execute("SELECT * FROM workflow_operations WHERE operation_id=?", (operation_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["result"] = json.loads(result["result"]) if result["result"] else None
        return result


def list_workflow_events(operation_id: str, after_id: int = 0) -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute("SELECT * FROM workflow_events WHERE operation_id=? AND event_id>? ORDER BY event_id",
                          (operation_id, int(after_id))).fetchall()
        return [{**dict(row), "data": json.loads(row["data"]) if row["data"] else None} for row in rows]


def list_target_events(evaluation_date: str, after_id: int = 0) -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute("""SELECT e.*,o.kind,o.evaluation_date FROM workflow_events e
            JOIN workflow_operations o ON o.operation_id=e.operation_id
            WHERE o.evaluation_date=? AND e.event_id>? ORDER BY e.event_id""",
            (evaluation_date, int(after_id))).fetchall()
        return [{**dict(row), "data": json.loads(row["data"]) if row["data"] else None} for row in rows]


def start_workflow(kind: str, evaluation_date: str, work: Callable[[Callable[..., None]], Any],
                   retry_failed: bool = False) -> dict[str, Any]:
    key = f"{kind}:{evaluation_date}"
    with _locks_guard:
        lock = _locks.setdefault(key, Lock())
    if not lock.acquire(blocking=False):
        raise RuntimeError(f"A {kind.replace('_', ' ')} workflow is already running for {evaluation_date}")
    with connect() as db:
        db.commit()
        db.execute("BEGIN IMMEDIATE")
        existing = db.execute("SELECT * FROM workflow_operations WHERE kind=? AND evaluation_date=? ORDER BY created_at DESC LIMIT 1",
                              (kind, evaluation_date)).fetchone()
        if existing and (existing["status"] in {"RUNNING", "COMPLETED"} or not retry_failed):
            lock.release()
            if existing["status"] == "FAILED" and not retry_failed:
                raise RuntimeError("A failed workflow exists; explicitly request a retry after reviewing its error")
            return {**dict(existing), "result": json.loads(existing["result"]) if existing["result"] else None}
        if existing and existing["status"] == "FAILED":
            key = f"{key}:retry:{uuid4()}"
        operation_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()
        db.execute("INSERT INTO workflow_operations(operation_id,kind,evaluation_date,status,stage,progress,created_at,updated_at,attempts,idempotency_key,owner_id) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                   (operation_id, kind, evaluation_date, "RUNNING", "starting", 0.0, now, now, 1, key,
                    f"{socket.gethostname()}:{os.getpid()}"))
    _record_event(operation_id, "started", "RUNNING", 0.0, "Workflow started")

    def progress(stage: str, value: float, message: str | None = None, data: Any = None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        value = min(.99, max(0.0, float(value)))
        with connect() as db:
            db.execute("UPDATE workflow_operations SET stage=?,progress=?,updated_at=? WHERE operation_id=?",
                       (stage, value, now, operation_id))
        _record_event(operation_id, stage, "RUNNING", value, message, data)

    def run() -> None:
        try:
            result = work(progress)
            now = datetime.now(timezone.utc).isoformat()
            with connect() as db:
                db.execute("UPDATE workflow_operations SET status='COMPLETED',stage='completed',progress=1,updated_at=?,result=?,error=NULL WHERE operation_id=?",
                           (now, json.dumps(result, default=str), operation_id))
                _insert_event(db, operation_id, "completed", "COMPLETED", 1.0, "Workflow completed", result)
        except Exception as exc:
            now = datetime.now(timezone.utc).isoformat()
            error = str(exc)
            for secret_name in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"):
                secret = os.getenv(secret_name)
                if secret:
                    error = error.replace(secret, "[redacted]")
            with connect() as db:
                db.execute("UPDATE workflow_operations SET status='FAILED',stage='failed',updated_at=?,error=? WHERE operation_id=?",
                           (now, error, operation_id))
                _insert_event(db, operation_id, "failed", "FAILED", 1.0, error)
        finally:
            lock.release()

    from threading import Thread
    Thread(target=run, name=f"workflow-{kind}-{operation_id[:8]}", daemon=True).start()
    return get_workflow(operation_id) or {"operation_id": operation_id, "status": "RUNNING"}


def recover_interrupted_workflows() -> int:
    """Mark interrupted process-local jobs failed so users can safely retry."""
    now = datetime.now(timezone.utc).isoformat()
    host = socket.gethostname()
    with connect() as db:
        rows = db.execute("SELECT operation_id,owner_id FROM workflow_operations WHERE status='RUNNING'").fetchall()
        interrupted = []
        for row in rows:
            try:
                owner_host, pid_text = (row["owner_id"] or "").rsplit(":", 1)
                if owner_host != host:
                    continue
                os.kill(int(pid_text), 0)
            except (ValueError, ProcessLookupError, PermissionError):
                interrupted.append(row["operation_id"])
            except OSError:
                interrupted.append(row["operation_id"])
        for operation_id in interrupted:
            db.execute("UPDATE workflow_operations SET status='FAILED',stage='interrupted',updated_at=?,error='Backend stopped during workflow; safe to retry after checking outputs' WHERE operation_id=?",
                       (now, operation_id))
    for operation_id in interrupted:
        _record_event(operation_id, "interrupted", "FAILED", 1.0,
                      "Backend stopped during workflow; safe to retry after checking outputs")
    return len(interrupted)
