"""SQLite-backed immutable forecasts, post-window outcomes, and pipeline logs."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "forecast_log.sqlite3"
SCHEMA_VERSION = 4


def _backup_before_migration(path: Path, source: sqlite3.Connection) -> Path | None:
    """Create a consistent online backup once before extending an existing DB."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    tables = {row[0] for row in source.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not tables:
        return None
    if "schema_migrations" in tables:
        version = source.execute("SELECT COALESCE(MAX(version),0) FROM schema_migrations").fetchone()[0]
        if int(version) >= SCHEMA_VERSION:
            return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = path.with_name(f"{path.name}.pre-v{SCHEMA_VERSION}-{stamp}.bak")
    dest = sqlite3.connect(backup)
    try:
        source.backup(dest)
    finally:
        dest.close()
    return backup


def _ensure_schema(db: sqlite3.Connection, path: Path, preexisting: bool) -> None:
    existing = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if preexisting and "schema_migrations" not in existing:
        _backup_before_migration(path, db)
    db.executescript("""
      CREATE TABLE IF NOT EXISTS schema_migrations (
        version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS forecast_versions (
        forecast_id TEXT PRIMARY KEY, evaluation_date TEXT NOT NULL, window_id TEXT NOT NULL,
        forecast_type TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 1,
        parent_forecast_id TEXT, generated_at TEXT NOT NULL, data_cutoff TEXT,
        training_cutoff TEXT, model_version TEXT, artifact_sha256 TEXT,
        status TEXT NOT NULL DEFAULT 'FROZEN', payload TEXT NOT NULL, actual_payload TEXT,
        UNIQUE(evaluation_date, window_id, forecast_type),
        FOREIGN KEY(parent_forecast_id) REFERENCES forecast_versions(forecast_id));
      CREATE INDEX IF NOT EXISTS ix_forecast_versions_date ON forecast_versions(evaluation_date, window_id, generated_at);
      CREATE TABLE IF NOT EXISTS workflow_operations (
        operation_id TEXT PRIMARY KEY, kind TEXT NOT NULL, evaluation_date TEXT NOT NULL,
        status TEXT NOT NULL, stage TEXT NOT NULL, progress REAL NOT NULL,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL, result TEXT, error TEXT,
        attempts INTEGER NOT NULL DEFAULT 0, idempotency_key TEXT UNIQUE, owner_id TEXT);
      CREATE TABLE IF NOT EXISTS workflow_events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT, operation_id TEXT NOT NULL,
        timestamp TEXT NOT NULL, stage TEXT NOT NULL, status TEXT NOT NULL,
        progress REAL NOT NULL, message TEXT, data TEXT,
        FOREIGN KEY(operation_id) REFERENCES workflow_operations(operation_id));
      CREATE INDEX IF NOT EXISTS ix_workflow_events_operation ON workflow_events(operation_id,event_id);
      CREATE TABLE IF NOT EXISTS operation_events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT, operation_id TEXT NOT NULL,
        timestamp TEXT NOT NULL, stage TEXT NOT NULL, status TEXT NOT NULL,
        progress REAL NOT NULL, message TEXT, data TEXT);
      CREATE INDEX IF NOT EXISTS ix_operation_events_operation ON operation_events(operation_id,event_id);
      CREATE TABLE IF NOT EXISTS scheduler_jobs (
        job_id TEXT PRIMARY KEY, evaluation_date TEXT NOT NULL, kind TEXT NOT NULL,
        run_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'PENDING', attempts INTEGER NOT NULL DEFAULT 0,
        operation_id TEXT, last_error TEXT, updated_at TEXT NOT NULL,
        UNIQUE(evaluation_date,kind));
      CREATE TABLE IF NOT EXISTS scheduler_lease (
        lease_name TEXT PRIMARY KEY, owner_id TEXT NOT NULL, expires_at TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS session_predictions (
        prediction_id TEXT PRIMARY KEY, evaluation_date TEXT NOT NULL, session_id TEXT NOT NULL,
        forecast_type TEXT NOT NULL, revision INTEGER NOT NULL, interval_start TEXT NOT NULL,
        interval_end TEXT NOT NULL, generated_at TEXT NOT NULL, feature_timestamp TEXT NOT NULL,
        model_name TEXT NOT NULL, model_version TEXT NOT NULL, payload TEXT NOT NULL, actual_payload TEXT,
        UNIQUE(evaluation_date,session_id,forecast_type,interval_start,revision));
      CREATE INDEX IF NOT EXISTS ix_session_predictions_date ON session_predictions(evaluation_date,session_id,forecast_type,interval_start);
      CREATE TABLE IF NOT EXISTS analysis_signals (
        signal_id TEXT PRIMARY KEY, candle_timestamp TEXT NOT NULL, generated_at TEXT NOT NULL,
        analysis_version TEXT NOT NULL, classification TEXT NOT NULL, directional_bias TEXT NOT NULL,
        final_score REAL, payload TEXT NOT NULL, status TEXT NOT NULL,
        outcome_payload TEXT, gross_return_pct REAL, net_return_pct REAL,
        UNIQUE(candle_timestamp,analysis_version));
      CREATE INDEX IF NOT EXISTS ix_analysis_signals_time ON analysis_signals(candle_timestamp DESC);
      CREATE INDEX IF NOT EXISTS ix_analysis_signals_status ON analysis_signals(status,candle_timestamp);
      CREATE TABLE IF NOT EXISTS analysis_signal_events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT, signal_id TEXT NOT NULL,
        timestamp TEXT NOT NULL, event_type TEXT NOT NULL, details TEXT NOT NULL,
        FOREIGN KEY(signal_id) REFERENCES analysis_signals(signal_id));
      CREATE INDEX IF NOT EXISTS ix_analysis_signal_events ON analysis_signal_events(signal_id,event_id);
    """)
    operation_columns = {row[1] for row in db.execute("PRAGMA table_info(workflow_operations)")}
    if "owner_id" not in operation_columns:
        db.execute("ALTER TABLE workflow_operations ADD COLUMN owner_id TEXT")
    # Preserve the v1 tables verbatim. Copy old forecast payloads into the new
    # queryable table with LEGACY type; logs remain in their original table.
    if "forecasts" in existing:
        for row in db.execute("SELECT forecast_id,evaluation_date,window_id,generated_at,payload,actual_payload FROM forecasts").fetchall():
            try:
                payload = json.loads(row["payload"])
            except (TypeError, ValueError):
                payload = {"legacy_payload": row["payload"]}
            payload.setdefault("forecast_type", "LEGACY")
            payload.setdefault("revision", 1)
            payload.setdefault("status", "FROZEN")
            actual = row["actual_payload"]
            db.execute("""INSERT OR IGNORE INTO forecast_versions
              (forecast_id,evaluation_date,window_id,forecast_type,revision,parent_forecast_id,generated_at,
               data_cutoff,training_cutoff,model_version,artifact_sha256,status,payload,actual_payload)
              VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
              (row["forecast_id"],row["evaluation_date"],row["window_id"],"LEGACY",1,None,
               row["generated_at"],payload.get("data_cutoff"),payload.get("training_cutoff"),
               payload.get("model_version"),payload.get("model_artifact_sha256"),"FROZEN",
               json.dumps(payload,default=str),actual))
    db.execute("INSERT OR IGNORE INTO schema_migrations(version,applied_at) VALUES(?,?)",
               (SCHEMA_VERSION, datetime.now(timezone.utc).isoformat()))


@contextmanager
def connect(path: str | Path | None = None):
    path = Path(path or DB_PATH)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    preexisting = path.exists() and path.stat().st_size > 0
    db = sqlite3.connect(path, timeout=30)
    db.row_factory = sqlite3.Row
    try:
        if preexisting:
            _backup_before_migration(path, db)
        db.executescript("""
      PRAGMA journal_mode=WAL;
      CREATE TABLE IF NOT EXISTS forecasts (
        forecast_id TEXT PRIMARY KEY, evaluation_date TEXT NOT NULL, window_id TEXT NOT NULL,
        generated_at TEXT NOT NULL, payload TEXT NOT NULL, actual_payload TEXT,
        UNIQUE(evaluation_date, window_id)
      );
      CREATE TABLE IF NOT EXISTS pipeline_logs (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT, forecast_id TEXT NOT NULL,
        evaluation_date TEXT NOT NULL, window_id TEXT NOT NULL, stage TEXT NOT NULL,
        timestamp TEXT NOT NULL, input_summary TEXT, output_summary TEXT,
        model_version TEXT, data_cutoff TEXT, rule TEXT, status TEXT NOT NULL,
        error_details TEXT,
        FOREIGN KEY(forecast_id) REFERENCES forecasts(forecast_id)
      );
      CREATE INDEX IF NOT EXISTS ix_logs_forecast ON pipeline_logs(forecast_id, log_id);
        """)
        _ensure_schema(db, path, False)
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def append_log(forecast_id: str, evaluation_date: str, window_id: str, stage: str,
               input_summary: Any, output_summary: Any, model_version: str | None,
               data_cutoff: str | None, rule: str | None, status: str = "ok",
               error_details: str | None = None, path: str | Path | None = None) -> None:
    with connect(path) as db:
        db.execute("""INSERT INTO pipeline_logs
          (forecast_id,evaluation_date,window_id,stage,timestamp,input_summary,output_summary,
           model_version,data_cutoff,rule,status,error_details)
          VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
          (forecast_id,evaluation_date,window_id,stage,datetime.now(timezone.utc).isoformat(),
           json.dumps(input_summary, default=str), json.dumps(output_summary, default=str),
           model_version,data_cutoff,rule,status,error_details))


def issue_forecast(payload: dict[str, Any], path: str | Path | None = None,
                   logs: list[tuple[str, Any, Any, str]] | None = None) -> None:
    with connect(path) as db:
        forecast_type = payload.get("forecast_type", "LEGACY")
        db.execute("""INSERT INTO forecast_versions
          (forecast_id,evaluation_date,window_id,forecast_type,revision,parent_forecast_id,generated_at,
           data_cutoff,training_cutoff,model_version,artifact_sha256,status,payload)
          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
          (payload["forecast_id"],payload["evaluation_date"],payload["window_id"],forecast_type,
           int(payload.get("revision",1)),payload.get("parent_forecast_id"),payload["generated_at"],
           payload.get("data_cutoff"),payload.get("training_cutoff"),payload.get("model_version"),
           payload.get("model_artifact_sha256"),payload.get("status","FROZEN"),json.dumps(payload,default=str)))
        for stage, inp, out, rule in logs or []:
            db.execute("""INSERT INTO pipeline_logs
              (forecast_id,evaluation_date,window_id,stage,timestamp,input_summary,output_summary,
               model_version,data_cutoff,rule,status,error_details)
              VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
              (payload["forecast_id"],payload["evaluation_date"],payload["window_id"],stage,
               payload["generated_at"],json.dumps(inp,default=str),json.dumps(out,default=str),
               payload["model_version"],payload["training_cutoff"],rule,"ok",None))


def record_actual(forecast_id: str, actual: dict[str, Any], path: str | Path | None = None,
                  log: tuple[str, Any, Any, str] | None = None) -> None:
    with connect(path) as db:
        row = db.execute("SELECT * FROM forecast_versions WHERE forecast_id=?", (forecast_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown forecast_id: {forecast_id}")
        if row["actual_payload"] is not None:
            raise ValueError("Actual result already recorded; comparison records are immutable.")
        db.execute("UPDATE forecast_versions SET actual_payload=? WHERE forecast_id=?", (json.dumps(actual,default=str),forecast_id))
        if log:
            stage, inp, out, rule = log
            payload = json.loads(row["payload"])
            db.execute("""INSERT INTO pipeline_logs
              (forecast_id,evaluation_date,window_id,stage,timestamp,input_summary,output_summary,
               model_version,data_cutoff,rule,status,error_details)
              VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
              (forecast_id,payload["evaluation_date"],payload["window_id"],stage,actual["compared_at"],
               json.dumps(inp,default=str),json.dumps(out,default=str),payload["model_version"],
               payload["training_cutoff"],rule,"ok",None))


def list_forecasts(evaluation_date: str | None = None, path: str | Path | None = None) -> list[dict[str, Any]]:
    with connect(path) as db:
        if evaluation_date:
            rows = db.execute("SELECT payload,actual_payload FROM forecast_versions WHERE evaluation_date=? ORDER BY window_id,forecast_type,revision",(evaluation_date,))
        else:
            rows = db.execute("SELECT payload,actual_payload FROM forecast_versions ORDER BY generated_at DESC")
        return [{**json.loads(r["payload"]), "actual": json.loads(r["actual_payload"]) if r["actual_payload"] else None} for r in rows]


def get_forecast(forecast_id: str, path: str | Path | None = None) -> dict[str, Any] | None:
    with connect(path) as db:
        row = db.execute("SELECT payload,actual_payload FROM forecast_versions WHERE forecast_id=?", (forecast_id,)).fetchone()
        if row is None:
            return None
        return {**json.loads(row["payload"]), "actual": json.loads(row["actual_payload"]) if row["actual_payload"] else None}


def list_logs(forecast_id: str | None = None, path: str | Path | None = None) -> list[dict[str, Any]]:
    with connect(path) as db:
        query = "SELECT * FROM pipeline_logs" + (" WHERE forecast_id=?" if forecast_id else "") + " ORDER BY log_id"
        rows = db.execute(query, (forecast_id,) if forecast_id else ())
        return [dict(row) for row in rows]
