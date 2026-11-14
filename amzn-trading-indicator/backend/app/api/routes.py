"""Dashboard, frozen forecast, comparison, and pipeline-log endpoints."""
from __future__ import annotations

from datetime import date, datetime, timezone
import json
from pathlib import Path
from zoneinfo import ZoneInfo
import asyncio
import time
import logging
import math
import numpy as np

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.forecasts.service import compare_completed_forecast, issue_window_forecast
from src.forecasts.store import list_forecasts, list_logs
from src.forecasts.store import append_log
from src.forecasts.training import load_config
from src.data.ingestion import (dataset_status, fetch_alpaca_candles, latest_processed_path,
                                load_local_candles, refresh_latest_market_data)
from src.forecasts.operations import get_operation, list_operations, start_operation
from src.forecasts.calendar import is_trading_day, previous_trading_day, cutoff_for_evaluation, market_timestamp, session_close
from src.forecasts.training import train_for_evaluation_date, load_daily_model
from src.forecasts.store import get_forecast
from src.forecasts.operations import list_operation_events
from backend.app.services.forecast_orchestrator import (retry_workflow,
    start_afternoon_update, start_next_day_forecasts, start_post_window_comparison)
from backend.app.services.workflow_operations import get_workflow, list_target_events, list_workflow_events
from backend.app.scheduler import scheduler_status
from src.forecasts.legacy_sessions import (generate_session_predictions, list_session_predictions,
    evaluate_session_predictions, interval_schedule, LEGACY_MODELS, session_actual_candles,
    _session_decision, _actual_session)
from backend.app.services.analysis.service import analyze_candles
from backend.app.services import market_store
from backend.app.services.analysis.outcomes import resolve_signals
from backend.app.services.analysis.persistence import get_signal, list_signals, signal_events, active_signals, latest_signal
from fastapi.responses import StreamingResponse, Response
import csv
import io

router = APIRouter()
ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / "data" / "processed" / "AMZN_clean.csv"
logger = logging.getLogger(__name__)


def _json_safe_optional(value):
    """Normalize optional metadata only, after required candle fields are validated."""
    if isinstance(value, dict):
        return {str(key): _json_safe_optional(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_optional(item) for item in value]
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _analysis_local_candles() -> pd.DataFrame:
    """Read snapshot without deduplication so analysis can reject corrupt timestamps."""
    try:
        path = latest_processed_path()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="No local processed market snapshot is available") from exc
    frame = pd.read_csv(path)
    if "timestamp" not in frame:
        raise HTTPException(status_code=503, detail="Processed market snapshot has no timestamp column")
    return frame


@router.post("/analysis/run")
def run_analysis(source: str = "local"):
    if source != "local":
        raise HTTPException(status_code=422, detail="Analysis currently accepts source='local'; no live request was made")
    try:
        result = analyze_candles(_analysis_local_candles(), source="local")
        return {"status": "ok", **result}
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Analysis could not complete: {type(exc).__name__}: {exc}") from exc


@router.get("/analysis/signals")
def analysis_signals(start: str | None = None, end: str | None = None,
                     status: str | None = None, limit: int = Query(100, ge=1, le=1000),
                     offset: int = Query(0, ge=0)):
    return {"signals": list_signals(start, end, status, limit, offset), "limit": limit, "offset": offset,
            "hypothetical_only": True}


@router.get("/analysis/latest")
def latest_analysis():
    item = latest_signal()
    if item is None:
        return {"status": "empty", "detail": "No analysis has been stored yet. Run POST /api/analysis/run.", "analysis": None}
    return {"status": "ok", **item}


@router.get("/analysis/active")
def active_analysis_signals():
    return {"signals": active_signals(), "hypothetical_only": True}


@router.get("/analysis/outcomes")
def analysis_outcomes(limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)):
    records = list_signals(limit=limit, offset=offset)
    return {"outcomes": [item for item in records if item["status"] not in {"ACTIVE", "NOT_ACTIONABLE"}],
            "hypothetical_only": True, "limit": limit, "offset": offset}


@router.get("/analysis/signals/{signal_id}")
def analysis_signal(signal_id: str):
    item = get_signal(signal_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Analysis signal not found")
    item["events"] = signal_events(signal_id)
    return item


@router.post("/analysis/outcomes/resolve")
def resolve_analysis_outcomes(source: str = "local"):
    if source != "local":
        raise HTTPException(status_code=422, detail="Outcome resolution currently accepts source='local'")
    try:
        return resolve_signals(_analysis_local_candles())
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


class IssueRequest(BaseModel):
    evaluation_date: date
    window_id: str


class CompareRequest(BaseModel):
    forecast_id: str


class WorkflowRequest(BaseModel):
    evaluation_date: date
    retry_failed: bool = False


class CompareWorkflowRequest(BaseModel):
    evaluation_date: date
    retry_failed: bool = False
    window_id: str | None = Field(default=None, pattern="^(morning|afternoon)$")


class SessionPredictionRequest(BaseModel):
    evaluation_date: date
    session_id: str = Field(pattern="^(morning|afternoon)$")
    forecast_type: str = Field(default="ADVANCE", pattern="^(ADVANCE|UPDATED)$")


def _candles(source: str) -> pd.DataFrame:
    if source == "alpaca":
        try:
            return fetch_alpaca_candles()
        except Exception as exc:
            detail = str(exc)
            import os
            for name in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"):
                secret = os.getenv(name)
                if secret:
                    detail = detail.replace(secret, "[redacted]")
            raise HTTPException(status_code=503, detail=detail) from exc
    if source != "local":
        raise HTTPException(status_code=422, detail="source must be 'alpaca' or 'local'")
    try:
        path = latest_processed_path()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="No local market data is available") from exc
    data = load_local_candles(path)
    if data.empty:
        raise HTTPException(status_code=503, detail="Local market data is empty")
    return data


@router.get("/market/candles")
def market_candles(limit: int = Query(1200, ge=1, le=10000), source: str = "local", after: str | None = None):
    # Capture cursor before reading; SSE replays any event concurrent with this snapshot.
    cursor = market_store.latest_event_id()
    full = _candles(source) if not after else pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    forming = None
    if source == "local":
        stream_rows = market_store.get_candles(limit=10000, after=after)
        if stream_rows:
            full = pd.concat([full, pd.DataFrame(stream_rows)], ignore_index=True, sort=False)
            full["timestamp"] = pd.to_datetime(full["timestamp"], utc=True, errors="raise")
            full = full.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        forming = _current_forming_candle()
    if "symbol" not in full.columns:
        # The local endpoint is AMZN-only. Make that configured identity explicit
        # for older/single-symbol snapshot files that predate a symbol column.
        logger.info("AMZN candle response supplied without a symbol column; applying configured symbol %s", market_store.MARKET_SYMBOL)
        full = full.copy()
        full["symbol"] = market_store.MARKET_SYMBOL
    invalid_timestamp = 0
    if "timestamp" in full.columns:
        full = full.copy()
        full["timestamp"] = pd.to_datetime(full["timestamp"], utc=True, errors="coerce")
        invalid_timestamp = int(full["timestamp"].isna().sum())
        full = full.loc[full["timestamp"].notna()]
        full = full.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    else:
        full = pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])
    if invalid_timestamp:
        logger.warning("Excluded %d AMZN candle record(s) with invalid timestamps", invalid_timestamp)
    valid_records = []
    invalid_records = []
    for record in full.to_dict(orient="records"):
        try:
            core = market_store.validate_candle_record(record)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            invalid_records.append((record.get("timestamp"), str(exc)))
            continue
        record.update(core)
        valid_records.append({key: _json_safe_optional(value) for key, value in record.items()})
    if invalid_records:
        logger.warning("Excluded %d invalid AMZN candle record(s) from API response; samples=%s",
                       len(invalid_records), invalid_records[:5])
    full_len = len(valid_records)
    response_records = valid_records[-limit:]
    status = dataset_status(limit) if source == "local" and not after else {"rows": full_len, "source": "Alpaca IEX stream" if source == "local" else "Alpaca IEX LIVE"}
    return {"symbol": "AMZN", "feed": "IEX completed 5-minute candles", "source": status.get("source", source),
            "dataset_rows": status.get("rows", full_len), "chart_rows": len(response_records),
            "dataset_start": status.get("dataset_start"), "dataset_end": status.get("dataset_end"),
            "latest_timestamp": response_records[-1]["timestamp"] if response_records else None, "candles": response_records,
            "forming_candle": _json_safe_optional(forming), "stream_status": _json_safe_optional(_market_stream_status()), "event_cursor": cursor}


def _market_stream_status() -> dict:
    status = market_store.get_status()
    now = datetime.now(ZoneInfo("America/New_York"))
    try:
        close = session_close(now.date()) if is_trading_day(now.date()) else None
        opened = bool(close and now.weekday() < 5 and (now.hour, now.minute) >= (9, 30) and now.time() < close)
    except Exception:
        opened = False
    status["market_state"] = "OPEN" if opened else "CLOSED"
    latest = status.get("last_completed_candle")
    if latest:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(latest).astimezone(timezone.utc)).total_seconds()
        status["freshness"] = "FRESH" if age <= 15 * 60 else "STALE"
    else:
        status["freshness"] = "UNAVAILABLE"
    return status


def _current_forming_candle() -> dict | None:
    candle = market_store.latest_forming()
    if not candle or _market_stream_status().get("market_state") != "OPEN":
        return None
    updated = datetime.fromisoformat(candle["updated_at"]).astimezone(timezone.utc)
    return candle if (datetime.now(timezone.utc) - updated).total_seconds() <= 5 * 60 else None


@router.get("/market/stream/status")
def market_stream_status():
    return _json_safe_optional(_market_stream_status())


@router.get("/market/events")
async def market_events(request: Request, last_event_id: int = Query(0, ge=0)):
    header_id = request.headers.get("last-event-id")
    try:
        cursor = max(last_event_id, int(header_id or 0))
    except ValueError:
        cursor = last_event_id
    async def stream():
        nonlocal cursor
        while not await request.is_disconnected():
            events = market_store.list_events(cursor, limit=250)
            if events:
                for event in events:
                    cursor = event["event_id"]
                    payload = {**event["payload"], "event_id": cursor,
                               "event_type": event["event_type"], "event_timestamp": event["event_timestamp"]}
                    yield f"id: {cursor}\nevent: {event['event_type']}\ndata: {json.dumps(payload, separators=(',', ':'), allow_nan=False)}\n\n"
            else:
                yield ": keep-alive\n\n"
                await asyncio.sleep(1)
    return StreamingResponse(stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/data/status")
def data_status(chart_limit: int = Query(2000, ge=1, le=10000)):
    try:
        status = dataset_status(chart_limit)
        status["stream_status"] = _market_stream_status()
        status["forming_candle"] = _current_forming_candle()
        return status
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/data/refresh", status_code=202)
def data_refresh():
    try:
        return start_operation("data_refresh", None, refresh_latest_market_data)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/operations")
def operations(kind: str | None = None):
    return {"operations": list_operations(kind)}


@router.get("/operations/{operation_id}")
def operation(operation_id: str):
    result = get_operation(operation_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Operation not found")
    return result


@router.get("/forecasts")
def forecasts(evaluation_date: str | None = None):
    records = list_forecasts(evaluation_date)
    by_date: dict[str, list[dict]] = {}
    for record in records:
        by_date.setdefault(record["evaluation_date"], []).append(record)
    analyses = {}
    for day, items in by_date.items():
        afternoon = [x for x in items if x.get("window_id") == "afternoon" and x.get("actual")]
        advance = next((x for x in afternoon if x.get("forecast_type") == "ADVANCE"), None)
        updated = next((x for x in afternoon if x.get("forecast_type") == "UPDATED"), None)
        if advance and updated:
            analyses[day] = {
                "advance_forecast_id": advance["forecast_id"], "updated_forecast_id": updated["forecast_id"],
                "same_actual_window": advance["actual"].get("actual_range") == updated["actual"].get("actual_range"),
                "direction_accuracy_change": int(bool(updated["actual"].get("direction_correct"))) - int(bool(advance["actual"].get("direction_correct"))),
                "movement_absolute_error_change": abs(float(updated["actual"].get("movement_error", 0))) - abs(float(advance["actual"].get("movement_error", 0))),
                "boundary_mae_change": float(updated["actual"].get("mae_boundary", 0)) - float(advance["actual"].get("mae_boundary", 0)),
                "range_coverage_change": int(bool(updated["actual"].get("predicted_range_coverage"))) - int(bool(advance["actual"].get("predicted_range_coverage"))),
                "interpretation": "Negative error changes mean the UPDATED forecast had lower error; these are paired outcomes for one target window, not a statistical performance claim.",
            }
    analysis = analyses.get(evaluation_date) if evaluation_date else analyses
    return {"forecasts": records, "afternoon_revision_analysis": analysis}


@router.get("/evaluations")
def evaluation_history(evaluation_date: date | None = Query(default=None)):
    """Return comparison-backed trading dates and records for one selected date."""
    all_records = list_forecasts()
    by_date: dict[str, list[dict]] = {}
    for record in all_records:
        day = record.get("evaluation_date")
        if day and record.get("actual"):
            try:
                valid_session = is_trading_day(date.fromisoformat(day))
            except ValueError:
                valid_session = False
            if valid_session:
                by_date.setdefault(day, []).append(record)
    available_dates = sorted(by_date, reverse=True)
    latest = available_dates[0] if available_dates else None
    selected = evaluation_date.isoformat() if evaluation_date else (latest or datetime.now(ZoneInfo("America/New_York")).date().isoformat())
    selected_records = [item for item in all_records if item.get("evaluation_date") == selected]
    return {
        "latest_completed_date": latest,
        "available_dates": [{"evaluation_date": day, "comparison_count": len(by_date[day])} for day in available_dates],
        "evaluation_date": selected,
        "comparison_count": sum(item.get("actual") is not None for item in selected_records),
        "forecasts": selected_records,
    }


@router.get("/forecasts/{forecast_id}")
def forecast_detail(forecast_id: str):
    result = get_forecast(forecast_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Forecast not found")
    return result


@router.get("/session-predictions")
def session_predictions(evaluation_date: date, session_id: str = Query(pattern="^(morning|afternoon)$"),
                        forecast_type: str = Query(default="ADVANCE", pattern="^(ADVANCE|UPDATED)$")):
    rows = list_session_predictions(evaluation_date.isoformat(), session_id, forecast_type)
    completed = [item for item in rows if item.get("actual")]
    return {"evaluation_date": evaluation_date.isoformat(), "session_id": session_id,
            "forecast_type": forecast_type, "model_version": LEGACY_MODELS.get("model_lightgbm.joblib"),
            "expected_intervals": len(interval_schedule(evaluation_date.isoformat(), session_id)),
            "predictions": rows, "final_decision": _session_decision(rows) if rows else None,
            "actual_session": _actual_session(completed) if rows and len(completed)==len(rows) else None,
            "pending_count": len(rows)-len(completed)}


@router.get("/session-predictions/actual-candles")
def session_prediction_actual_candles(evaluation_date: date, session_id: str = Query(pattern="^(morning|afternoon)$")):
    try:
        return {"evaluation_date": evaluation_date.isoformat(), "session_id": session_id,
                "timeframe": "10m", "source": "aggregated genuine local 5-minute OHLCV",
                "candles": session_actual_candles(evaluation_date.isoformat(), session_id)}
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/session-predictions/generate", status_code=201)
def generate_session(request: SessionPredictionRequest):
    try:
        return generate_session_predictions(request.evaluation_date.isoformat(), request.session_id, request.forecast_type)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/session-predictions/evaluate")
def evaluate_session(request: SessionPredictionRequest):
    try:
        return evaluate_session_predictions(request.evaluation_date.isoformat(), request.session_id, request.forecast_type)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/session-predictions/export.csv")
def export_session_predictions(evaluation_date: date, session_id: str = Query(pattern="^(morning|afternoon)$"),
                               forecast_type: str = Query(default="ADVANCE", pattern="^(ADVANCE|UPDATED)$")):
    records = list_session_predictions(evaluation_date.isoformat(), session_id, forecast_type)
    buffer = io.StringIO(); writer = csv.writer(buffer)
    writer.writerow(["evaluation_date", "session", "forecast_type", "interval_start", "interval_end", "reference_price", "signal", "raw_probability_up", "confidence", "support", "resistance", "model_version", "outcome", "actual_return_pct"])
    for row in records:
        actual = row.get("actual") or {}
        writer.writerow([row["evaluation_date"], row["session_id"], row["forecast_type"], row["interval_start"], row["interval_end"], row["reference_price"], row["predicted_signal"], row.get("raw_probability_up"), row.get("confidence"), row.get("support"), row.get("resistance"), row["model_version"], actual.get("outcome"), actual.get("actual_return_pct")])
    return Response(buffer.getvalue(), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=AMZN_{evaluation_date}_{session_id}_{forecast_type}.csv"})


@router.post("/workflows/next-day", status_code=202)
def next_day_workflow(request: WorkflowRequest):
    try:
        return start_next_day_forecasts(request.evaluation_date.isoformat(), request.retry_failed)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/workflows/afternoon-update", status_code=202)
def afternoon_update_workflow(request: WorkflowRequest):
    try:
        return start_afternoon_update(request.evaluation_date.isoformat(), request.retry_failed)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/workflows/compare", status_code=202)
def comparison_workflow(request: CompareWorkflowRequest):
    try:
        return start_post_window_comparison(request.evaluation_date.isoformat(), request.retry_failed,
                                            request.window_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/workflows/{operation_id}")
def workflow_status(operation_id: str):
    result = get_workflow(operation_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Workflow operation not found")
    result["events"] = list_workflow_events(operation_id)
    return result


@router.post("/workflows/{operation_id}/retry", status_code=202)
def workflow_retry(operation_id: str):
    try:
        return retry_workflow(operation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Workflow operation not found") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _sse(operation_id: str, request: Request, workflow: bool, last_event_id: int):
    status_getter = get_workflow if workflow else get_operation
    event_getter = list_workflow_events if workflow else list_operation_events
    if status_getter(operation_id) is None:
        raise HTTPException(status_code=404, detail="Operation not found")

    async def stream():
        cursor = last_event_id
        heartbeat_at = time.monotonic()
        while True:
            rows = event_getter(operation_id, cursor)
            for row in rows:
                cursor = int(row["event_id"])
                payload = json.dumps(row, default=str)
                yield f"id: {cursor}\nevent: progress\ndata: {payload}\n\n"
            state = status_getter(operation_id)
            if state and state.get("status") in {"COMPLETED", "FAILED"}:
                yield f"event: terminal\ndata: {json.dumps(state, default=str)}\n\n"
                break
            if time.monotonic() - heartbeat_at >= 15:
                yield ": heartbeat\n\n"
                heartbeat_at = time.monotonic()
            if await request.is_disconnected():
                break
            await asyncio.sleep(1)
    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/workflows/{operation_id}/events")
def workflow_events(operation_id: str, request: Request, last_event_id: int = Query(0, alias="lastEventId")):
    header_id = request.headers.get("last-event-id")
    return _sse(operation_id, request, True, int(header_id) if header_id and header_id.isdigit() else last_event_id)


@router.get("/events")
async def target_workflow_events(request: Request, evaluation_date: date = Query(...),
                                 last_event_id: int = Query(0, alias="lastEventId")):
    header_id = request.headers.get("last-event-id")
    cursor = int(header_id) if header_id and header_id.isdigit() else last_event_id

    async def stream():
        nonlocal cursor
        heartbeat_at = time.monotonic()
        while not await request.is_disconnected():
            for event in list_target_events(evaluation_date.isoformat(), cursor):
                cursor = int(event["event_id"])
                yield f"id: {cursor}\nevent: workflow\ndata: {json.dumps(event, default=str)}\n\n"
                if event["status"] in {"COMPLETED", "FAILED"}:
                    yield f"event: refresh\ndata: {json.dumps({'operation_id': event['operation_id'], 'status': event['status']})}\n\n"
            if time.monotonic() - heartbeat_at >= 15:
                yield ": heartbeat\n\n"
                heartbeat_at = time.monotonic()
            await asyncio.sleep(1)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/operations/{operation_id}/events")
def operation_events(operation_id: str, request: Request, last_event_id: int = Query(0, alias="lastEventId")):
    header_id = request.headers.get("last-event-id")
    return _sse(operation_id, request, False, int(header_id) if header_id and header_id.isdigit() else last_event_id)


@router.get("/scheduler/status")
def get_scheduler_status():
    return scheduler_status()


@router.get("/config")
def forecast_config():
    return load_config()


@router.post("/forecasts/issue", status_code=201)
def issue(request: IssueRequest, source: str = "alpaca"):
    try:
        return issue_window_forecast(request.evaluation_date.isoformat(), request.window_id, _candles(source))
    except HTTPException as exc:
        append_log(f"attempt-{request.evaluation_date}-{request.window_id}", request.evaluation_date.isoformat(),
                   request.window_id, "forecast_issuance", {"source": source}, {}, None, None,
                   "pre-window-valid-data-required", "failed", str(exc.detail))
        raise
    except FileNotFoundError as exc:
        append_log(f"attempt-{request.evaluation_date}-{request.window_id}", request.evaluation_date.isoformat(),
                   request.window_id, "forecast_issuance", {"source": source}, {}, None, None,
                   "registered-model-required", "failed", str(exc))
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, FileExistsError) as exc:
        append_log(f"attempt-{request.evaluation_date}-{request.window_id}", request.evaluation_date.isoformat(),
                   request.window_id, "forecast_issuance", {"source": source}, {}, None, None,
                   "pre-window-valid-data-required", "failed", str(exc))
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        append_log(f"attempt-{request.evaluation_date}-{request.window_id}", request.evaluation_date.isoformat(),
                   request.window_id, "forecast_issuance", {"source": source}, {}, None, None,
                   "issuance_validation_and_persistence", "failed", type(exc).__name__)
        raise HTTPException(status_code=500, detail=f"Forecast issuance failed: {exc}") from exc


@router.post("/forecasts/compare")
def compare(request: CompareRequest, source: str = "alpaca"):
    try:
        return compare_completed_forecast(request.forecast_id, _candles(source))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/pipeline-logs")
def pipeline_logs(forecast_id: str | None = None):
    return {"logs": list_logs(forecast_id)}


@router.get("/model/status")
def model_status(evaluation_date: date = Query(...)):
    path = ROOT / "artifacts" / "window_models" / evaluation_date.isoformat() / "metadata.json"
    if not path.exists():
        return {"evaluation_date": evaluation_date.isoformat(), "registered": False,
                "eligible": False, "status": "MODEL NOT READY", "reason": "No model version registered"}
    metadata = json.loads(path.read_text(encoding="utf-8"))
    cfg = load_config()
    artifacts = {}
    integrity = True
    for window in cfg["windows"]:
        try:
            load_daily_model(evaluation_date.isoformat(), window["id"])
            artifacts[window["id"]] = "VERIFIED"
        except Exception as exc:
            artifacts[window["id"]] = f"FAILED: {exc}"
            integrity = False
    cutoff = pd.Timestamp(metadata["training_cutoff"])
    eligible = integrity and pd.Timestamp(metadata["dataset_end"]) <= cutoff
    first_window = load_config()["windows"][0]
    from src.forecasts.calendar import market_timestamp
    freeze_deadline = market_timestamp(evaluation_date, first_window["start"])
    trained_at = pd.Timestamp(metadata["trained_at"])
    eligible = eligible and trained_at < pd.Timestamp(freeze_deadline)
    if list_forecasts(evaluation_date.isoformat()):
        eligible = eligible and integrity
    return {**metadata, "registered": True, "artifact_status": artifacts,
            "integrity_verified": integrity, "eligible": eligible,
            "status": "READY" if eligible else "MODEL NOT READY",
            "cutoff_for_evaluation": cutoff_for_evaluation(evaluation_date).isoformat()}


@router.get("/model/versions")
def model_versions(evaluation_date: date | None = None):
    root = ROOT / "artifacts" / "window_models"
    versions = []
    for path in sorted(root.glob("*/metadata.json")) if root.exists() else []:
        if evaluation_date and path.parent.name != evaluation_date.isoformat():
            continue
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            versions.append({k: metadata.get(k) for k in ("model_version", "evaluation_date", "trained_at", "training_cutoff", "dataset_start", "dataset_end", "dataset_rows")})
        except (OSError, ValueError):
            continue
    return {"versions": versions}


@router.post("/model/train", status_code=202)
def model_train(evaluation_date: date = Query(...)):
    if not is_trading_day(evaluation_date):
        raise HTTPException(status_code=422, detail="Evaluation date is not a scheduled trading day")
    if list_forecasts(evaluation_date.isoformat()):
        raise HTTPException(status_code=409, detail="Forecasts have already been issued for this date; model training is blocked to preserve their frozen model association")
    try:
        data_path = latest_processed_path()
        metadata_path = ROOT / "artifacts" / "window_models" / evaluation_date.isoformat() / "metadata.json"
        if metadata_path.exists():
            raise HTTPException(status_code=409, detail="A model version is already registered for this evaluation date; registered artifacts are never overwritten")
        prior_day = previous_trading_day(evaluation_date)
        expected = pd.Timestamp(market_timestamp(prior_day, session_close(prior_day).strftime("%H:%M")) - pd.Timedelta(minutes=5)).tz_convert("UTC")
        candles = load_local_candles(data_path)
        latest = pd.Timestamp(candles.timestamp.iloc[-1]) if not candles.empty else None
        if latest != expected:
            raise HTTPException(status_code=409, detail=f"Training is blocked: previous-session final completed candle is missing. Expected {expected.isoformat()}, found {latest.isoformat() if latest is not None else 'no candles'}")
        return start_operation("model_training", evaluation_date.isoformat(),
                               lambda progress: train_for_evaluation_date(evaluation_date.isoformat(), data_path, progress))
    except (RuntimeError, FileNotFoundError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/model/{evaluation_date}")
def model_metadata(evaluation_date: date):
    path = ROOT / "artifacts" / "window_models" / evaluation_date.isoformat() / "metadata.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No model registered for that evaluation date")
    return json.loads(path.read_text(encoding="utf-8"))
