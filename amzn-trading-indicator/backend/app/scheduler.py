"""Dedicated persistent scheduler process. Start separately from Uvicorn."""
from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import time
from datetime import date, datetime, timedelta, timezone
from uuid import uuid4
from zoneinfo import ZoneInfo

from backend.app.services.forecast_orchestrator import (start_afternoon_update,
                                                         start_next_day_forecasts,
                                                         start_post_window_comparison)
from backend.app.services.workflow_operations import get_workflow, recover_interrupted_workflows
from src.forecasts.calendar import is_trading_day, market_timestamp, previous_trading_day, session_close
from src.forecasts.store import connect
from src.forecasts.training import load_config

logger = logging.getLogger("amzn.scheduler")
ET = ZoneInfo("America/New_York")
OWNER = f"{socket.gethostname()}:{os.getpid()}:{uuid4()}"
LEASE_SECONDS = 60


def _scheduled_jobs(now: datetime, config: dict) -> list[dict]:
    local_now = now.astimezone(ET)
    settings = config.get("scheduler", {})
    jobs = []
    day = local_now.date()
    # Reconcile today and the next ten calendar days. The table's unique date/kind
    # key makes this safe to repeat after process restarts.
    for offset in range(0, 11):
        target = day + timedelta(days=offset)
        if not is_trading_day(target):
            continue
        previous = previous_trading_day(target)
        close = market_timestamp(previous, session_close(previous).strftime("%H:%M"))
        advance_at = close + timedelta(minutes=int(settings.get("advance_after_close_minutes", 5)))
        first_start = market_timestamp(target, config["windows"][0]["start"])
        if advance_at < first_start and local_now < first_start:
            jobs.append({"evaluation_date": target.isoformat(), "kind": "next_day_advance", "run_at": advance_at})
        session_close_dt = market_timestamp(target, session_close(target).strftime("%H:%M"))
        afternoon = next((w for w in config["windows"] if w["id"] == "afternoon"), None)
        if afternoon and market_timestamp(target, afternoon["end"]) <= session_close_dt:
            morning_end = market_timestamp(target, config["windows"][0]["end"])
            update_at = morning_end + timedelta(minutes=int(settings.get("morning_update_delay_minutes", 2)))
            compare_at = morning_end + timedelta(minutes=int(settings.get("comparison_delay_minutes", 2)))
            jobs.extend([
                {"evaluation_date": target.isoformat(), "kind": "afternoon_update", "run_at": update_at},
                {"evaluation_date": target.isoformat(), "kind": "post_window_comparison", "run_at": compare_at},
                {"evaluation_date": target.isoformat(), "kind": "post_window_comparison_afternoon", "run_at": market_timestamp(target, afternoon["end"]) + timedelta(minutes=int(settings.get("comparison_delay_minutes", 2)))},
            ])
        morning_end = market_timestamp(target, config["windows"][0]["end"])
        if morning_end <= session_close_dt:
            jobs.append({"evaluation_date": target.isoformat(), "kind": "post_window_comparison_morning", "run_at": morning_end + timedelta(minutes=int(settings.get("comparison_delay_minutes", 2)))})
    return jobs


def _persist_schedule(now: datetime, config: dict) -> None:
    stamp = now.astimezone(timezone.utc).isoformat()
    with connect() as db:
        for job in _scheduled_jobs(now, config):
            job_id = f"{job['evaluation_date']}:{job['kind']}"
            db.execute("INSERT OR IGNORE INTO scheduler_jobs(job_id,evaluation_date,kind,run_at,status,updated_at) VALUES(?,?,?,?, 'PENDING',?)",
                       (job_id, job["evaluation_date"], job["kind"], job["run_at"].astimezone(timezone.utc).isoformat(), stamp))
        # Recover job rows left RUNNING after a scheduler process crash.
        running = db.execute("""SELECT j.job_id,j.operation_id,o.status AS operation_status
            FROM scheduler_jobs j LEFT JOIN workflow_operations o ON o.operation_id=j.operation_id
            WHERE j.status='RUNNING'""").fetchall()
        for row in running:
            if row["operation_status"] == "RUNNING":
                continue
            db.execute("UPDATE scheduler_jobs SET status='RETRY',operation_id=NULL,last_error='Scheduler restarted; recovering interrupted workflow',updated_at=? WHERE job_id=?",
                       (stamp, row["job_id"]))


def _claim_lease(now: datetime) -> bool:
    expiry = (now + timedelta(seconds=LEASE_SECONDS)).astimezone(timezone.utc).isoformat()
    current = now.astimezone(timezone.utc).isoformat()
    with connect() as db:
        db.commit()
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT owner_id,expires_at FROM scheduler_lease WHERE lease_name='forecast_scheduler'").fetchone()
        if row and row["owner_id"] != OWNER and row["expires_at"] > current:
            db.rollback()
            return False
        db.execute("INSERT INTO scheduler_lease(lease_name,owner_id,expires_at) VALUES('forecast_scheduler',?,?) ON CONFLICT(lease_name) DO UPDATE SET owner_id=excluded.owner_id,expires_at=excluded.expires_at",
                   (OWNER, expiry))
        db.commit()
    return True


def _dispatch(kind: str, evaluation_date: str) -> dict:
    if kind == "next_day_advance":
        return start_next_day_forecasts(evaluation_date, retry_failed=True)
    if kind == "afternoon_update":
        return start_afternoon_update(evaluation_date, retry_failed=True)
    if kind.startswith("post_window_comparison"):
        window_id = "morning" if kind.endswith("_morning") else "afternoon" if kind.endswith("_afternoon") else None
        return start_post_window_comparison(evaluation_date, retry_failed=True, window_id=window_id)
    raise ValueError(f"Unknown scheduled job: {kind}")


def _tick(now: datetime | None = None) -> None:
    now = now or datetime.now(timezone.utc)
    config = load_config()
    if not config.get("scheduler", {}).get("enabled", True):
        return
    if not _claim_lease(now):
        logger.warning("Another scheduler owns the active lease")
        return
    _persist_schedule(now, config)
    current = now.astimezone(timezone.utc)
    with connect() as db:
        due = db.execute("SELECT * FROM scheduler_jobs WHERE status IN ('PENDING','RETRY') AND run_at<=? ORDER BY run_at LIMIT 20",
                         (current.isoformat(),)).fetchall()
    retry_delays = config.get("scheduler", {}).get("retry_delays_seconds", [60, 300, 900])
    max_attempts = int(config.get("scheduler", {}).get("maximum_attempts", 4))
    for row in due:
        # Skip any forecast task whose allowed generation window has already ended.
        target = date.fromisoformat(row["evaluation_date"])
        if row["kind"] == "next_day_advance":
            deadline = market_timestamp(target, config["windows"][0]["start"]).astimezone(timezone.utc)
            if current >= deadline:
                with connect() as db:
                    db.execute("UPDATE scheduler_jobs SET status='SKIPPED',last_error='Advance issuance deadline passed',updated_at=? WHERE job_id=?",
                               (current.isoformat(), row["job_id"]))
                continue
        if row["kind"] == "afternoon_update":
            deadline = market_timestamp(target, config["windows"][1]["end"]).astimezone(timezone.utc)
            if current >= deadline:
                with connect() as db:
                    db.execute("UPDATE scheduler_jobs SET status='SKIPPED',last_error='Afternoon update deadline passed',updated_at=? WHERE job_id=?",
                               (current.isoformat(), row["job_id"]))
                continue
        if row["attempts"] >= max_attempts:
            continue
        try:
            op = _dispatch(row["kind"], row["evaluation_date"])
            with connect() as db:
                db.execute("UPDATE scheduler_jobs SET status=?,attempts=attempts+1,operation_id=?,updated_at=? WHERE job_id=?",
                           ("RUNNING" if op.get("status") == "RUNNING" else op.get("status", "RUNNING"),
                            op.get("operation_id"), current.isoformat(), row["job_id"]))
        except Exception as exc:
            attempts = int(row["attempts"]) + 1
            if attempts >= max_attempts:
                status, next_run = "FAILED", current
            else:
                delay = retry_delays[min(attempts - 1, len(retry_delays) - 1)] if retry_delays else 60
                status, next_run = "RETRY", current + timedelta(seconds=int(delay))
            with connect() as db:
                db.execute("UPDATE scheduler_jobs SET status=?,attempts=?,run_at=?,last_error=?,updated_at=? WHERE job_id=?",
                           (status, attempts, next_run.isoformat(), str(exc), current.isoformat(), row["job_id"]))
            logger.exception("Scheduled job dispatch failed: %s", row["job_id"])
    # Reflect finished operation outcomes and apply bounded backoff.
    with connect() as db:
        running = db.execute("SELECT * FROM scheduler_jobs WHERE status='RUNNING' AND operation_id IS NOT NULL").fetchall()
    for row in running:
        operation = get_workflow(row["operation_id"])
        if not operation or operation["status"] == "RUNNING":
            continue
        if operation["status"] == "COMPLETED":
            state, error, next_run = "COMPLETED", None, current
        else:
            attempts = int(row["attempts"])
            if attempts >= max_attempts:
                state, error, next_run = "FAILED", operation.get("error"), current
            else:
                delay = retry_delays[min(attempts - 1, len(retry_delays) - 1)] if retry_delays else 60
                state, error, next_run = "RETRY", operation.get("error"), current + timedelta(seconds=int(delay))
        with connect() as db:
            db.execute("UPDATE scheduler_jobs SET status=?,run_at=?,last_error=?,updated_at=? WHERE job_id=?",
                       (state, next_run.isoformat(), error, current.isoformat(), row["job_id"]))
    # Session prediction outcomes are polled from the local processed snapshot.
    # The evaluator leaves missing/delayed intervals pending and SQL uniqueness
    # plus actual_payload IS NULL makes repeated ticks idempotent.
    try:
        from src.forecasts.legacy_sessions import auto_evaluate_available
        auto_evaluate_available()
    except (FileNotFoundError, ValueError, OSError) as exc:
        logger.info("Session evaluation deferred until local actual candles are available: %s", exc)
    # Resolve hypothetical analysis-engine setups from completed local bars.
    # Missing/stale snapshots are deferred; the API never places broker orders.
    try:
        from backend.app.api.routes import _analysis_local_candles
        from backend.app.services.analysis.outcomes import resolve_signals
        resolve_signals(_analysis_local_candles())
    except (FileNotFoundError, ValueError, OSError) as exc:
        logger.info("Analysis-signal resolution deferred until completed candles are available: %s", exc)


def scheduler_status() -> dict:
    now = datetime.now(timezone.utc)
    config = load_config()
    with connect() as db:
        lease = db.execute("SELECT * FROM scheduler_lease WHERE lease_name='forecast_scheduler'").fetchone()
        jobs = db.execute("SELECT * FROM scheduler_jobs ORDER BY run_at LIMIT 100").fetchall()
    active = bool(lease and lease["expires_at"] > now.isoformat())
    next_jobs = [dict(row) for row in jobs if row["status"] in {"PENDING", "RETRY", "RUNNING"}][:20]
    last_success = next((dict(row) for row in jobs if row["status"] == "COMPLETED"), None)
    failures = [dict(row) for row in jobs if row["status"] in {"FAILED", "SKIPPED"}][-20:]
    return {"enabled": bool(config.get("scheduler", {}).get("enabled", True)),
            "running": active, "owner": lease["owner_id"] if active else None,
            "lease_expires_at": lease["expires_at"] if active else None,
            "timezone": "America/New_York", "next_jobs": next_jobs,
            "last_successful_job": last_success, "failed_jobs": failures}


def run_scheduler(poll_seconds: int | None = None) -> None:
    recover_interrupted_workflows()
    settings = load_config().get("scheduler", {})
    interval = poll_seconds or int(settings.get("poll_seconds", 15))
    logger.info("Forecast scheduler started (%s, %ss poll)", OWNER, interval)
    while True:
        try:
            _tick()
        except Exception:
            logger.exception("Scheduler tick failed")
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dedicated AMZN forecast scheduler process")
    parser.add_argument("--poll-seconds", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        run_scheduler(args.poll_seconds)
    except KeyboardInterrupt:
        logger.info("Forecast scheduler stopped")
