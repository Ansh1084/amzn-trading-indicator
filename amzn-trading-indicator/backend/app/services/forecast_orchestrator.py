"""Single orchestration path for manual and scheduled forecast workflows."""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd

from src.data.ingestion import (fetch_alpaca_candles, latest_processed_path,
                                load_local_candles, refresh_latest_market_data)
from src.forecasts.calendar import (is_trading_day, market_timestamp,
                                    previous_trading_day, session_close)
from src.forecasts.service import compare_completed_forecast, issue_window_forecast
from src.forecasts.store import get_forecast, list_forecasts
from src.forecasts.training import load_config, load_daily_model, train_for_evaluation_date
from backend.app.services.workflow_operations import start_workflow

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[3]


def _progress(progress: Callable[..., None], stage: str, fraction: float,
              message: str, data: Any = None) -> None:
    progress(stage, fraction, message, data)


def _model_ready(target: date, candles: pd.DataFrame, progress: Callable[..., None]) -> dict[str, Any]:
    directory = ROOT / "artifacts" / "window_models" / target.isoformat()
    metadata_path = directory / "metadata.json"
    config = load_config()
    if metadata_path.exists():
        for window in config["windows"]:
            load_daily_model(target.isoformat(), window["id"])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if pd.Timestamp(metadata["dataset_end"]) > pd.Timestamp(metadata["training_cutoff"]):
            raise RuntimeError("Registered model dataset exceeds its training cutoff")
        first_start = market_timestamp(target, config["windows"][0]["start"])
        if pd.Timestamp(metadata["trained_at"]) >= pd.Timestamp(first_start):
            raise RuntimeError("Registered model was not trained before the target session began")
        first_start = market_timestamp(target, config["windows"][0]["start"])
        if pd.Timestamp(metadata["trained_at"]) >= pd.Timestamp(first_start):
            raise RuntimeError("Registered model was not trained before the target session began")
        _progress(progress, "model_verified", .42, "Existing registered model and hashes verified",
                  {"model_version": metadata["model_version"]})
        return metadata
    _progress(progress, "model_training", .34, "No registered model; fitting from the previous-session cutoff")
    path = latest_processed_path()
    metadata = train_for_evaluation_date(target.isoformat(), path)
    for window in config["windows"]:
        load_daily_model(target.isoformat(), window["id"])
    _progress(progress, "model_verified", .48, "New model artifacts trained and hashes verified",
              {"model_version": metadata["model_version"]})
    return metadata


def _next_day_work(target_text: str, progress: Callable[..., None]) -> dict[str, Any]:
    target = date.fromisoformat(target_text)
    if not is_trading_day(target):
        raise ValueError(f"Target date {target} is not a scheduled trading day")
    now_et = datetime.now(timezone.utc).astimezone(ET)
    prior = previous_trading_day(target)
    prior_close = market_timestamp(prior, session_close(prior).strftime("%H:%M"))
    first_start = market_timestamp(target, load_config()["windows"][0]["start"])
    if not (prior_close <= now_et < first_start):
        raise ValueError("Advance forecasts can only be generated after the previous session close and before the first target window")
    _progress(progress, "data_refresh", .08, "Refreshing completed Alpaca IEX history")
    refreshed = refresh_latest_market_data(lambda stage, value: progress(f"data_{stage}", .08 + .20 * value,
                                                                         "Refreshing and validating market history"))
    path = Path(refreshed["processed_path"])
    candles = load_local_candles(path)
    metadata = _model_ready(target, candles, progress)
    output = []
    for index, window in enumerate(load_config()["windows"]):
        if market_timestamp(target, window["end"]) > market_timestamp(target, session_close(target).strftime("%H:%M")):
            _progress(progress, f"forecast_{window['id']}_skipped", .50 + .18 * index,
                      "Configured forecast window extends beyond this scheduled early close")
            continue
        if market_timestamp(target, window["end"]) > market_timestamp(target, session_close(target).strftime("%H:%M")):
            _progress(progress, f"forecast_{window['id']}_skipped", .50 + .18 * index,
                      "Configured forecast window extends beyond this scheduled early close")
            continue
        existing = next((item for item in list_forecasts(target_text)
                         if item.get("window_id") == window["id"] and item.get("forecast_type") == "ADVANCE"), None)
        if existing:
            output.append(existing)
            continue
        _progress(progress, f"forecast_{window['id']}", .50 + .18 * index,
                  f"Generating frozen ADVANCE {window['id']} forecast")
        payload = issue_window_forecast(target_text, window["id"], candles,
                                        datetime.now(timezone.utc), forecast_type="ADVANCE")
        output.append(payload)
    return {"evaluation_date": target_text, "model_version": metadata["model_version"],
            "forecasts": output, "data_path": str(path)}


def start_next_day_forecasts(evaluation_date: str, retry_failed: bool = False) -> dict[str, Any]:
    return start_workflow("next_day_advance", evaluation_date,
                          lambda progress: _next_day_work(evaluation_date, progress), retry_failed)


def _afternoon_update_work(target_text: str, progress: Callable[..., None]) -> dict[str, Any]:
    target = date.fromisoformat(target_text)
    if not is_trading_day(target):
        raise ValueError(f"Target date {target} is not a scheduled trading day")
    now = datetime.now(timezone.utc)
    now_et = now.astimezone(ET)
    start = market_timestamp(target, "12:30")
    end = market_timestamp(target, load_config()["windows"][1]["end"])
    if end > market_timestamp(target, session_close(target).strftime("%H:%M")):
        raise ValueError("Configured afternoon window extends beyond this scheduled early close")
    if end > market_timestamp(target, session_close(target).strftime("%H:%M")):
        raise ValueError("Configured afternoon window extends beyond this scheduled early close")
    if not (start <= now_et < end):
        raise ValueError("UPDATED afternoon forecast requires a completed morning and must run before the afternoon window ends")
    existing = next((item for item in list_forecasts(target_text)
                     if item.get("window_id") == "afternoon" and item.get("forecast_type") == "UPDATED"), None)
    if existing:
        return {"evaluation_date": target_text, "forecasts": [existing], "already_exists": True}
    _progress(progress, "fetching_morning_data", .12, "Fetching completed candles for the morning validation")
    candles = fetch_alpaca_candles(lookback_days=35)
    morning_start = pd.Timestamp(market_timestamp(target, "09:30")).tz_convert("UTC")
    morning_last = pd.Timestamp(market_timestamp(target, "12:25")).tz_convert("UTC")
    data = candles.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="raise")
    data = data.loc[(data.timestamp >= morning_start) & (data.timestamp <= morning_last)]
    expected = pd.date_range(morning_start, morning_last, freq="5min")
    if data.timestamp.duplicated().any() or set(data.timestamp) != set(expected):
        missing = len(set(expected) - set(data.timestamp))
        raise RuntimeError(f"Morning candles are not complete; {missing} of 36 expected completed bars are missing. Retry after data arrives.")
    _progress(progress, "morning_validated", .35, "All 36 morning candles are complete")
    bundle, metadata = load_daily_model(target_text, "afternoon")
    del bundle
    cutoff = pd.Timestamp(metadata["training_cutoff"])
    if pd.Timestamp(metadata["dataset_end"]) > cutoff:
        raise RuntimeError("Afternoon model metadata violates its previous-session training cutoff")
    advance = next((item for item in list_forecasts(target_text)
                    if item.get("window_id") == "afternoon" and item.get("forecast_type") == "ADVANCE"), None)
    _progress(progress, "model_verified", .48, "Previous-session afternoon model verified; no retraining")
    payload = issue_window_forecast(target_text, "afternoon", candles, now,
                                    forecast_type="UPDATED",
                                    parent_forecast_id=advance["forecast_id"] if advance else None)
    return {"evaluation_date": target_text, "model_version": metadata["model_version"], "forecasts": [payload]}


def start_afternoon_update(evaluation_date: str, retry_failed: bool = False) -> dict[str, Any]:
    # Type/date idempotency is also enforced by the forecast_versions unique key.
    return start_workflow("afternoon_update", evaluation_date,
                          lambda progress: _afternoon_update_work(evaluation_date, progress), retry_failed)


def _compare_work(target_text: str, progress: Callable[..., None], window_id: str | None = None) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    eligible = [item for item in list_forecasts(target_text)
                if item.get("actual") is None and (window_id is None or item.get("window_id") == window_id)
                and pd.Timestamp(item["window_end"]).tz_convert("UTC") <= pd.Timestamp(now)]
    if not eligible:
        return {"evaluation_date": target_text, "comparison_count": 0, "comparisons": []}
    _progress(progress, "fetching_actuals", .15, "Fetching completed Alpaca candles")
    candles = fetch_alpaca_candles(lookback_days=35)
    compared = []
    for index, forecast in enumerate(eligible):
        _progress(progress, "comparing_forecast", .25 + .65 * (index / max(1, len(eligible))),
                  f"Comparing {forecast.get('forecast_type')} {forecast['window_id']}",
                  {"forecast_id": forecast["forecast_id"]})
        compared.append(compare_completed_forecast(forecast["forecast_id"], candles, now)["actual"])
    return {"evaluation_date": target_text, "comparison_count": len(compared), "comparisons": compared}


def start_post_window_comparison(evaluation_date: str, retry_failed: bool = False,
                                 window_id: str | None = None) -> dict[str, Any]:
    suffix = f"_{window_id}" if window_id else ""
    return start_workflow(f"post_window_comparison{suffix}", evaluation_date,
                          lambda progress: _compare_work(evaluation_date, progress, window_id), retry_failed)


def retry_workflow(operation_id: str) -> dict[str, Any]:
    from backend.app.services.workflow_operations import get_workflow
    current = get_workflow(operation_id)
    if not current:
        raise KeyError(operation_id)
    if current["status"] == "RUNNING":
        raise RuntimeError("Workflow is already running")
    kind, target = current["kind"], current["evaluation_date"]
    if kind == "next_day_advance":
        return start_next_day_forecasts(target, retry_failed=True)
    if kind == "afternoon_update":
        return start_afternoon_update(target, retry_failed=True)
    if kind.startswith("post_window_comparison"):
        window_id = kind.rsplit("_", 1)[-1] if kind.endswith(("_morning", "_afternoon")) else None
        return start_post_window_comparison(target, retry_failed=True, window_id=window_id)
    raise ValueError(f"Workflow kind {kind} cannot be retried")
