from __future__ import annotations

import json
import sqlite3
import time
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

import backend.app.scheduler as scheduler
import backend.app.api.routes as api_routes
import backend.app.services.forecast_orchestrator as orchestrator
from backend.app.services.workflow_operations import get_workflow, list_target_events, list_workflow_events, start_workflow
from src.features.engineering import get_feature_columns
from src.forecasts.calendar import cutoff_for_evaluation, session_close
from src.forecasts import store as forecast_store
from src.forecasts.service import compare_completed_forecast, issue_window_forecast

ET = ZoneInfo("America/New_York")


class _Regressor:
    def __init__(self, value: float):
        self.value = value

    def predict(self, values):
        return np.asarray([self.value])


class _Classifier:
    def predict_proba(self, values):
        return np.asarray([[0.4, 0.6]])


def _candles() -> pd.DataFrame:
    prior = pd.date_range("2026-11-09 14:30:00+00:00", "2026-11-09 20:55:00+00:00", freq="5min")
    morning = pd.date_range("2026-11-10 14:30:00+00:00", "2026-11-10 17:25:00+00:00", freq="5min")
    timestamps = prior.append(morning)
    close = 250 + np.arange(len(timestamps)) * 0.01 + np.sin(np.arange(len(timestamps)) / 4)
    return pd.DataFrame({"timestamp": timestamps, "open": close - 0.03, "high": close + 0.12,
                         "low": close - 0.12, "close": close, "volume": np.full(len(close), 1000),
                         "vwap": close, "trade_count": 1, "symbol": "AMZN"})


def _stub_model(monkeypatch):
    eval_day = "2026-11-10"
    cutoff = cutoff_for_evaluation(date.fromisoformat(eval_day)).isoformat()
    metadata = {"training_cutoff": cutoff, "dataset_end": "2026-11-09T20:55:00+00:00",
                "model_version": "test-window-version", "model_artifact_sha256": {"morning": "abc", "afternoon": "def"},
                "walk_forward_evaluation": {}}
    bundle = {"feature_columns": get_feature_columns(), "medians": {},
              "models": {"direction": _Classifier(), "end_return": _Regressor(.002),
                         "low_return": _Regressor(-.01), "high_return": _Regressor(.01)},
              "direction_calibrator": None, "calibration": [], "calibration_minimum": 30}
    monkeypatch.setattr("src.forecasts.service.load_daily_model", lambda d, w: (bundle, metadata))


def test_advance_forecast_caps_inference_data_at_previous_close(tmp_path, monkeypatch):
    monkeypatch.setattr(forecast_store, "DB_PATH", tmp_path / "records.sqlite3")
    _stub_model(monkeypatch)
    generated = datetime(2026, 11, 9, 16, 5, tzinfo=ET)
    result = issue_window_forecast("2026-11-10", "morning", _candles(), generated,
                                  forecast_type="ADVANCE")
    expected = pd.Timestamp("2026-11-09T20:55:00Z")
    assert pd.Timestamp(result["data_cutoff"]) == expected
    assert pd.Timestamp(result["training_cutoff"]) == pd.Timestamp(cutoff_for_evaluation(date(2026, 11, 10)))
    assert result["forecast_type"] == "ADVANCE"
    assert result["window_start"].startswith("2026-11-10T09:30:00")


def test_updated_afternoon_forecast_uses_only_completed_morning_and_frozen_model(tmp_path, monkeypatch):
    monkeypatch.setattr(forecast_store, "DB_PATH", tmp_path / "records.sqlite3")
    _stub_model(monkeypatch)
    generated = datetime(2026, 11, 10, 12, 32, tzinfo=ET)
    result = issue_window_forecast("2026-11-10", "afternoon", _candles(), generated,
                                  forecast_type="UPDATED", parent_forecast_id=None)
    assert pd.Timestamp(result["data_cutoff"]) == pd.Timestamp("2026-11-10T17:25:00Z")
    assert pd.Timestamp(result["feature_generation_timestamp"]) == pd.Timestamp("2026-11-10T17:25:00Z")
    assert pd.Timestamp(result["training_cutoff"]) == pd.Timestamp(cutoff_for_evaluation(date(2026, 11, 10)))
    assert result["forecast_type"] == "UPDATED"
    assert result["model_version"] == "test-window-version"


def test_updated_afternoon_refuses_one_missing_morning_bar(tmp_path, monkeypatch):
    monkeypatch.setattr(forecast_store, "DB_PATH", tmp_path / "records.sqlite3")
    _stub_model(monkeypatch)
    candles = _candles().drop(index=78).reset_index(drop=True)
    with pytest.raises(ValueError, match="Morning window data is incomplete"):
        issue_window_forecast("2026-11-10", "afternoon", candles,
                              datetime(2026, 11, 10, 12, 32, tzinfo=ET), forecast_type="UPDATED")


def test_versioned_store_allows_advance_and_updated_but_prevents_duplicate_type(tmp_path):
    db = tmp_path / "versioned.sqlite3"
    base = {"evaluation_date": "2026-11-10", "window_id": "afternoon", "generated_at": "2026-11-09T21:05:00Z",
            "window_start": "2026-11-10T12:30:00-05:00", "window_end": "2026-11-10T15:30:00-05:00",
            "predicted_direction": "UP", "predicted_range": {"lower": 1, "upper": 2}}
    advance = {**base, "forecast_id": "advance-1", "forecast_type": "ADVANCE", "revision": 1}
    updated = {**base, "forecast_id": "updated-1", "forecast_type": "UPDATED", "revision": 1,
               "parent_forecast_id": "advance-1", "generated_at": "2026-11-10T17:32:00Z"}
    forecast_store.issue_forecast(advance, db)
    forecast_store.issue_forecast(updated, db)
    with pytest.raises(sqlite3.IntegrityError):
        forecast_store.issue_forecast({**advance, "forecast_id": "advance-2"}, db)
    found = forecast_store.list_forecasts("2026-11-10", db)
    assert {item["forecast_type"] for item in found} == {"ADVANCE", "UPDATED"}
    assert next(item for item in found if item["forecast_type"] == "UPDATED")["parent_forecast_id"] == "advance-1"


def test_additive_migration_backs_up_and_preserves_legacy_rows_and_logs(tmp_path):
    db_path = tmp_path / "legacy.sqlite3"
    raw_payload = json.dumps({"forecast_id": "old-id", "evaluation_date": "2026-11-09", "window_id": "morning",
                              "generated_at": "2026-11-06T13:29:00Z", "predicted_direction": "DOWN"})
    db = sqlite3.connect(db_path)
    db.executescript("""CREATE TABLE forecasts(forecast_id TEXT PRIMARY KEY,evaluation_date TEXT,window_id TEXT,
        generated_at TEXT,payload TEXT,actual_payload TEXT,UNIQUE(evaluation_date,window_id));
        CREATE TABLE pipeline_logs(log_id INTEGER PRIMARY KEY,forecast_id TEXT,evaluation_date TEXT,window_id TEXT,
        stage TEXT,timestamp TEXT,input_summary TEXT,output_summary TEXT,model_version TEXT,data_cutoff TEXT,rule TEXT,
        status TEXT,error_details TEXT);""")
    db.execute("INSERT INTO forecasts VALUES(?,?,?,?,?,?)", ("old-id", "2026-11-09", "morning",
               "2026-11-06T13:29:00Z", raw_payload, json.dumps({"direction_correct": True})))
    db.execute("INSERT INTO pipeline_logs VALUES(1,'old-id','2026-11-09','morning','model_prediction','2026-11-06T13:29:00Z','{}','{}',NULL,NULL,NULL,'ok',NULL)")
    db.commit(); db.close()

    migrated = forecast_store.list_forecasts("2026-11-09", db_path)
    assert migrated[0]["forecast_type"] == "LEGACY"
    assert migrated[0]["actual"]["direction_correct"] is True
    with forecast_store.connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM pipeline_logs").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM forecast_versions").fetchone()[0] == 1
    backups = list(tmp_path.glob("legacy.sqlite3.pre-v*.bak"))
    assert len(backups) == 1
    backup = sqlite3.connect(backups[0])
    assert backup.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0] == 1
    backup.close()


def test_scheduler_schedule_accounts_for_dst_and_early_close():
    # Friday close remains 16:00 EST before Monday's 09:30 EDT session.
    now = datetime(2026, 3, 6, 22, 0, tzinfo=timezone.utc)
    jobs = scheduler._scheduled_jobs(now, {"windows": [{"id": "morning", "start": "09:30", "end": "12:30"},
                                               {"id": "afternoon", "start": "12:30", "end": "15:30"}],
                                "scheduler": {"advance_after_close_minutes": 5}})
    monday = next(job for job in jobs if job["evaluation_date"] == "2026-03-09" and job["kind"] == "next_day_advance")
    assert monday["run_at"].astimezone(timezone.utc).hour == 21
    assert session_close(date(2026, 11, 27)).hour == 13
    assert not any(job["evaluation_date"] == "2026-11-27" and job["kind"] == "afternoon_update"
                   for job in scheduler._scheduled_jobs(datetime(2026, 11, 26, 22, tzinfo=timezone.utc),
                      {"windows": [{"id": "morning", "start": "09:30", "end": "12:30"},
                                   {"id": "afternoon", "start": "12:30", "end": "15:30"}], "scheduler": {}}))


def test_scheduler_jobs_are_persisted_idempotently_and_lease_is_single_owner(tmp_path, monkeypatch):
    monkeypatch.setattr(forecast_store, "DB_PATH", tmp_path / "scheduler.sqlite3")
    monkeypatch.setattr(scheduler, "OWNER", "host:100")
    config = {"windows": [{"id": "morning", "start": "09:30", "end": "12:30"},
                          {"id": "afternoon", "start": "12:30", "end": "15:30"}],
              "scheduler": {"advance_after_close_minutes": 5, "morning_update_delay_minutes": 2,
                            "comparison_delay_minutes": 2}}
    now = datetime(2026, 11, 9, 22, 0, tzinfo=timezone.utc)
    scheduler._persist_schedule(now, config)
    scheduler._persist_schedule(now, config)
    with forecast_store.connect() as db:
        count = db.execute("SELECT COUNT(*) FROM scheduler_jobs").fetchone()[0]
    assert count == len(scheduler._scheduled_jobs(now, config))
    assert scheduler._claim_lease(now)
    monkeypatch.setattr(scheduler, "OWNER", "other:200")
    assert not scheduler._claim_lease(now)


def test_workflow_events_are_persistent_and_terminal_state_is_recoverable(tmp_path, monkeypatch):
    monkeypatch.setattr(forecast_store, "DB_PATH", tmp_path / "workflow.sqlite3")
    operation = start_workflow("test_workflow", "2026-11-10",
                               lambda progress: (progress("forecast_generation", .7, "model loaded"), {"ok": True})[1])
    deadline = time.time() + 3
    state = get_workflow(operation["operation_id"])
    while state and state["status"] == "RUNNING" and time.time() < deadline:
        time.sleep(.01)
        state = get_workflow(operation["operation_id"])
    assert state and state["status"] == "COMPLETED"
    events = list_workflow_events(operation["operation_id"])
    assert [event["stage"] for event in events] == ["started", "forecast_generation", "completed"]
    assert events[1]["message"] == "model loaded"


def test_failed_workflow_requires_explicit_idempotent_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(forecast_store, "DB_PATH", tmp_path / "retry.sqlite3")
    operation = start_workflow("retry_test", "2026-11-10", lambda _progress: (_ for _ in ()).throw(ValueError("expected failure")))
    deadline = time.time() + 3
    while get_workflow(operation["operation_id"])["status"] == "RUNNING" and time.time() < deadline:
        time.sleep(.01)
    assert get_workflow(operation["operation_id"])["status"] == "FAILED"
    with pytest.raises(RuntimeError, match="explicitly request a retry"):
        start_workflow("retry_test", "2026-11-10", lambda _progress: {"ok": True})
    retry = start_workflow("retry_test", "2026-11-10", lambda _progress: {"ok": True}, retry_failed=True)
    assert retry["operation_id"] != operation["operation_id"]
    deadline = time.time() + 3
    while get_workflow(retry["operation_id"])["status"] == "RUNNING" and time.time() < deadline:
        time.sleep(.01)
    assert get_workflow(retry["operation_id"])["status"] == "COMPLETED"
    assert start_workflow("retry_test", "2026-11-10", lambda _progress: {"duplicate": True})["operation_id"] == retry["operation_id"]


def test_target_sse_route_is_registered_and_event_ids_support_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(forecast_store, "DB_PATH", tmp_path / "sse.sqlite3")
    operation = start_workflow("sse_test", "2026-11-10", lambda progress: (progress("stage_one", .5), {"ok": True})[1])
    deadline = time.time() + 3
    while get_workflow(operation["operation_id"])["status"] == "RUNNING" and time.time() < deadline:
        time.sleep(.01)
    events = list_target_events("2026-11-10")
    assert [event["event_id"] for event in events] == sorted(event["event_id"] for event in events)
    assert list_target_events("2026-11-10", events[0]["event_id"]) == events[1:]
    route = next(route for route in api_routes.router.routes if getattr(route, "path", None) == "/events")
    assert route.methods == {"GET"}


def test_next_day_orchestrator_issues_both_advance_forecasts_after_refresh(tmp_path, monkeypatch):
    fixed_now = datetime(2026, 11, 9, 17, 0, tzinfo=ET)
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now.astimezone(tz) if tz else fixed_now.replace(tzinfo=None)
    config = {"windows": [{"id": "morning", "start": "09:30", "end": "12:30"},
                          {"id": "afternoon", "start": "12:30", "end": "15:30"}],
              "timezone": "America/New_York", "support_lookback_sessions": 20,
              "swing_left_bars": 2, "swing_right_bars": 2, "level_atr_distance": 3,
              "minimum_calibration_samples": 30}
    monkeypatch.setattr(orchestrator, "datetime", FrozenDateTime)
    monkeypatch.setattr(orchestrator, "load_config", lambda: config)
    monkeypatch.setattr(orchestrator, "ROOT", tmp_path)
    monkeypatch.setattr(orchestrator, "refresh_latest_market_data", lambda progress: {"processed_path": str(tmp_path / "processed.csv")})
    monkeypatch.setattr(orchestrator, "latest_processed_path", lambda: tmp_path / "processed.csv")
    monkeypatch.setattr(orchestrator, "load_local_candles", lambda _path: _candles())
    monkeypatch.setattr(orchestrator, "train_for_evaluation_date", lambda *_args: {"model_version": "v1", "training_cutoff": "2026-11-09T21:00:00Z"})
    monkeypatch.setattr(orchestrator, "load_daily_model", lambda *_args: ({}, {}))
    monkeypatch.setattr(orchestrator, "list_forecasts", lambda _date: [])
    issued = []
    def issue(day, window_id, candles, generated_at, **kwargs):
        issued.append((window_id, kwargs["forecast_type"], generated_at))
        return {"forecast_id": window_id, "window_id": window_id, "forecast_type": kwargs["forecast_type"]}
    monkeypatch.setattr(orchestrator, "issue_window_forecast", issue)
    result = orchestrator._next_day_work("2026-11-10", lambda *_args: None)
    assert [(window, kind) for window, kind, _ in issued] == [("morning", "ADVANCE"), ("afternoon", "ADVANCE")]
    assert result["model_version"] == "v1"


def test_afternoon_update_uses_complete_morning_and_never_trains(tmp_path, monkeypatch):
    fixed_now = datetime(2026, 11, 10, 12, 32, tzinfo=ET)
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now.astimezone(tz) if tz else fixed_now.replace(tzinfo=None)
    config = {"windows": [{"id": "morning", "start": "09:30", "end": "12:30"},
                          {"id": "afternoon", "start": "12:30", "end": "15:30"}], "scheduler": {}}
    advance = {"forecast_id": "advance-afternoon", "window_id": "afternoon", "forecast_type": "ADVANCE", "actual": None}
    monkeypatch.setattr(orchestrator, "datetime", FrozenDateTime)
    monkeypatch.setattr(orchestrator, "load_config", lambda: config)
    monkeypatch.setattr(orchestrator, "fetch_alpaca_candles", lambda **_kwargs: _candles())
    monkeypatch.setattr(orchestrator, "list_forecasts", lambda _date: [advance])
    monkeypatch.setattr(orchestrator, "load_daily_model", lambda *_args: ({}, {"training_cutoff": "2026-11-09T21:00:00Z",
                                                                               "dataset_end": "2026-11-09T20:55:00Z",
                                                                               "model_version": "frozen-afternoon"}))
    monkeypatch.setattr(orchestrator, "train_for_evaluation_date", lambda *_args: pytest.fail("UPDATED inference must not retrain"))
    captured = {}
    def issue(day, window, candles, generated_at, **kwargs):
        captured.update({"window": window, "type": kwargs["forecast_type"], "parent": kwargs["parent_forecast_id"],
                         "cutoff": pd.to_datetime(candles.timestamp, utc=True).max()})
        return {"forecast_id": "updated-afternoon", "forecast_type": "UPDATED"}
    monkeypatch.setattr(orchestrator, "issue_window_forecast", issue)
    result = orchestrator._afternoon_update_work("2026-11-10", lambda *_args: None)
    assert captured == {"window": "afternoon", "type": "UPDATED", "parent": "advance-afternoon",
                        "cutoff": pd.Timestamp("2026-11-10T17:25:00Z")}
    assert result["forecasts"][0]["forecast_type"] == "UPDATED"


def test_afternoon_update_rejects_incomplete_morning_before_model_load(monkeypatch):
    fixed_now = datetime(2026, 11, 10, 12, 32, tzinfo=ET)
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now.astimezone(tz) if tz else fixed_now.replace(tzinfo=None)
    monkeypatch.setattr(orchestrator, "datetime", FrozenDateTime)
    monkeypatch.setattr(orchestrator, "load_config", lambda: {"windows": [{"id": "morning", "start": "09:30", "end": "12:30"},
                                                                               {"id": "afternoon", "start": "12:30", "end": "15:30"}]})
    monkeypatch.setattr(orchestrator, "fetch_alpaca_candles", lambda **_kwargs: _candles().drop(index=78))
    monkeypatch.setattr(orchestrator, "list_forecasts", lambda _date: [])
    monkeypatch.setattr(orchestrator, "load_daily_model", lambda *_args: pytest.fail("Must reject incomplete data first"))
    with pytest.raises(RuntimeError, match="36 expected completed bars"):
        orchestrator._afternoon_update_work("2026-11-10", lambda *_args: None)


def test_advance_and_updated_forecasts_compare_separately_to_same_actual_window(tmp_path, monkeypatch):
    monkeypatch.setattr(forecast_store, "DB_PATH", tmp_path / "paired.sqlite3")
    base = {"evaluation_date": "2026-11-10", "window_id": "afternoon",
            "window_start": "2026-11-10T12:30:00-05:00", "window_end": "2026-11-10T15:30:00-05:00",
            "training_cutoff": "2026-11-09T21:00:00+00:00", "model_version": "frozen-v1",
            "reference_price": 250.0, "predicted_range": {"lower": 249.0, "upper": 252.0},
            "support_resistance": {"support": 249.5, "resistance": 251.0}}
    advance = {**base, "forecast_id": "advance-pair", "forecast_type": "ADVANCE", "revision": 1,
               "generated_at": "2026-11-09T21:05:00Z", "expected_price_movement": .25, "predicted_direction": "UP"}
    updated = {**base, "forecast_id": "updated-pair", "forecast_type": "UPDATED", "revision": 1,
               "parent_forecast_id": "advance-pair", "generated_at": "2026-11-10T17:32:00Z",
               "expected_price_movement": .20, "predicted_direction": "DOWN"}
    forecast_store.issue_forecast(advance)
    forecast_store.issue_forecast(updated)
    stamps = pd.date_range("2026-11-10T17:30:00Z", "2026-11-10T20:25:00Z", freq="5min")
    closes = 250.0 + np.arange(len(stamps)) * .01
    actual = pd.DataFrame({"timestamp": stamps, "open": closes - .02, "high": closes + .1,
                           "low": closes - .1, "close": closes, "volume": 1000})
    now = datetime(2026, 11, 10, 15, 32, tzinfo=ET)
    a_result = compare_completed_forecast("advance-pair", actual, now)["actual"]
    u_result = compare_completed_forecast("updated-pair", actual, now)["actual"]
    assert a_result["actual_range"] == u_result["actual_range"]
    assert a_result["direction_correct"] is True and u_result["direction_correct"] is False
    records = forecast_store.list_forecasts("2026-11-10")
    assert {item["forecast_type"] for item in records} == {"ADVANCE", "UPDATED"}
    assert all(item["actual"] is not None for item in records)
    analysis = api_routes.forecasts("2026-11-10")["afternoon_revision_analysis"]
    assert analysis["same_actual_window"] is True
    assert analysis["direction_accuracy_change"] == -1


def test_next_day_orchestrator_issues_both_advance_forecasts_after_refresh(tmp_path, monkeypatch):
    fixed_now = datetime(2026, 11, 9, 17, 0, tzinfo=ET)
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now.astimezone(tz) if tz else fixed_now.replace(tzinfo=None)
    config = {"windows": [{"id": "morning", "start": "09:30", "end": "12:30"},
                          {"id": "afternoon", "start": "12:30", "end": "15:30"}],
              "timezone": "America/New_York", "support_lookback_sessions": 20,
              "swing_left_bars": 2, "swing_right_bars": 2, "level_atr_distance": 3,
              "minimum_calibration_samples": 30}
    monkeypatch.setattr(orchestrator, "datetime", FrozenDateTime)
    monkeypatch.setattr(orchestrator, "load_config", lambda: config)
    monkeypatch.setattr(orchestrator, "ROOT", tmp_path)
    monkeypatch.setattr(orchestrator, "refresh_latest_market_data", lambda progress: {"processed_path": str(tmp_path / "processed.csv")})
    monkeypatch.setattr(orchestrator, "latest_processed_path", lambda: tmp_path / "processed.csv")
    monkeypatch.setattr(orchestrator, "load_local_candles", lambda _path: _candles())
    monkeypatch.setattr(orchestrator, "train_for_evaluation_date", lambda *_args: {"model_version": "v1", "training_cutoff": "2026-11-09T21:00:00Z"})
    monkeypatch.setattr(orchestrator, "load_daily_model", lambda *_args: ({}, {}))
    monkeypatch.setattr(orchestrator, "list_forecasts", lambda _date: [])
    issued = []
    def issue(day, window_id, candles, generated_at, **kwargs):
        issued.append((window_id, kwargs["forecast_type"], generated_at))
        return {"forecast_id": window_id, "window_id": window_id, "forecast_type": kwargs["forecast_type"]}
    monkeypatch.setattr(orchestrator, "issue_window_forecast", issue)
    result = orchestrator._next_day_work("2026-11-10", lambda *_args: None)
    assert [(window, kind) for window, kind, _ in issued] == [("morning", "ADVANCE"), ("afternoon", "ADVANCE")]
    assert result["model_version"] == "v1"


def test_afternoon_update_uses_complete_morning_and_never_trains(tmp_path, monkeypatch):
    fixed_now = datetime(2026, 11, 10, 12, 32, tzinfo=ET)
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now.astimezone(tz) if tz else fixed_now.replace(tzinfo=None)
    config = {"windows": [{"id": "morning", "start": "09:30", "end": "12:30"},
                          {"id": "afternoon", "start": "12:30", "end": "15:30"}], "scheduler": {}}
    advance = {"forecast_id": "advance-afternoon", "window_id": "afternoon", "forecast_type": "ADVANCE", "actual": None}
    monkeypatch.setattr(orchestrator, "datetime", FrozenDateTime)
    monkeypatch.setattr(orchestrator, "load_config", lambda: config)
    monkeypatch.setattr(orchestrator, "fetch_alpaca_candles", lambda **_kwargs: _candles())
    monkeypatch.setattr(orchestrator, "list_forecasts", lambda _date: [advance])
    monkeypatch.setattr(orchestrator, "load_daily_model", lambda *_args: ({}, {"training_cutoff": "2026-11-09T21:00:00Z",
                                                                               "dataset_end": "2026-11-09T20:55:00Z",
                                                                               "model_version": "frozen-afternoon"}))
    monkeypatch.setattr(orchestrator, "train_for_evaluation_date", lambda *_args: pytest.fail("UPDATED inference must not retrain"))
    captured = {}
    def issue(day, window, candles, generated_at, **kwargs):
        captured.update({"window": window, "type": kwargs["forecast_type"], "parent": kwargs["parent_forecast_id"],
                         "cutoff": pd.to_datetime(candles.timestamp, utc=True).max()})
        return {"forecast_id": "updated-afternoon", "forecast_type": "UPDATED"}
    monkeypatch.setattr(orchestrator, "issue_window_forecast", issue)
    result = orchestrator._afternoon_update_work("2026-11-10", lambda *_args: None)
    assert captured == {"window": "afternoon", "type": "UPDATED", "parent": "advance-afternoon",
                        "cutoff": pd.Timestamp("2026-11-10T17:25:00Z")}
    assert result["forecasts"][0]["forecast_type"] == "UPDATED"


def test_afternoon_update_rejects_incomplete_morning_before_model_load(monkeypatch):
    fixed_now = datetime(2026, 11, 10, 12, 32, tzinfo=ET)
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now.astimezone(tz) if tz else fixed_now.replace(tzinfo=None)
    monkeypatch.setattr(orchestrator, "datetime", FrozenDateTime)
    monkeypatch.setattr(orchestrator, "load_config", lambda: {"windows": [{"id": "morning", "start": "09:30", "end": "12:30"},
                                                                               {"id": "afternoon", "start": "12:30", "end": "15:30"}]})
    monkeypatch.setattr(orchestrator, "fetch_alpaca_candles", lambda **_kwargs: _candles().drop(index=78))
    monkeypatch.setattr(orchestrator, "list_forecasts", lambda _date: [])
    monkeypatch.setattr(orchestrator, "load_daily_model", lambda *_args: pytest.fail("Must reject incomplete data first"))
    with pytest.raises(RuntimeError, match="36 expected completed bars"):
        orchestrator._afternoon_update_work("2026-11-10", lambda *_args: None)
