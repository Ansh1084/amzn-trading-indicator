"""Leakage-gated issuance and post-window evaluation services."""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.features.engineering import generate_features
from src.forecasts.calendar import is_trading_day, market_timestamp, previous_trading_day, session_close
from src.forecasts.levels import downside_scenario, find_levels, support_hold_probability
from src.forecasts.store import issue_forecast, list_forecasts, record_actual
from src.forecasts.store import get_forecast
from src.forecasts.training import load_config, load_daily_model

ROOT = Path(__file__).resolve().parents[2]
ET = ZoneInfo("America/New_York")


def issue_window_forecast(evaluation_date: str, window_id: str, candles: pd.DataFrame,
                          generated_at: datetime | None = None, forecast_type: str = "INTRADAY",
                          parent_forecast_id: str | None = None) -> dict[str, Any]:
    generated_at = generated_at or datetime.now(timezone.utc)
    if generated_at.tzinfo is None:
        raise ValueError("generated_at must be timezone-aware")
    eval_day = date.fromisoformat(evaluation_date)
    if not is_trading_day(eval_day):
        raise ValueError("Evaluation date is not a trading day")
    cfg = load_config()
    window = next((w for w in cfg["windows"] if w["id"] == window_id), None)
    if window is None:
        raise ValueError(f"Unknown window_id: {window_id}")
    forecast_type = forecast_type.upper()
    if forecast_type not in {"ADVANCE", "UPDATED", "INTRADAY"}:
        raise ValueError("forecast_type must be ADVANCE, UPDATED, or INTRADAY")
    if forecast_type == "UPDATED" and window_id != "afternoon":
        raise ValueError("UPDATED forecasts are supported only for the afternoon window")
    start = market_timestamp(eval_day, window["start"])
    end = market_timestamp(eval_day, window["end"])
    close = market_timestamp(eval_day, session_close(eval_day).strftime("%H:%M"))
    if end > close:
        raise ValueError("Configured forecast window extends beyond this session close")
    local_generated_at = generated_at.astimezone(ET)
    lead = int(cfg.get("issuance_lead_minutes", 2))
    if forecast_type == "ADVANCE":
        previous_close = market_timestamp(previous_trading_day(eval_day),
                                          session_close(previous_trading_day(eval_day)).strftime("%H:%M"))
        if local_generated_at < previous_close or local_generated_at >= start:
            raise ValueError("ADVANCE forecasts must be generated after the previous session close and before the target window")
    elif forecast_type == "UPDATED":
        morning_end = market_timestamp(eval_day, "12:30")
        if local_generated_at < morning_end or local_generated_at >= end:
            raise ValueError("UPDATED afternoon forecasts require completed morning data and must precede the afternoon window end")
    else:
        if local_generated_at >= start:
            raise ValueError("Forecast must be generated strictly before its window begins")
        if local_generated_at < start - pd.Timedelta(minutes=lead):
            raise ValueError(f"Forecast issuance opens {lead} minute(s) before the configured window start")

    if candles.empty or "timestamp" not in candles:
        raise ValueError("No candle history available")
    df = candles.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    df = df.sort_values("timestamp").reset_index(drop=True)
    if df.timestamp.duplicated().any():
        raise ValueError("Duplicate candle timestamps; refusing forecast")
    if forecast_type == "ADVANCE":
        prev_day = previous_trading_day(eval_day)
        prev_close = pd.Timestamp(market_timestamp(prev_day, session_close(prev_day).strftime("%H:%M"))).tz_convert("UTC")
        df = df.loc[df.timestamp + pd.Timedelta(minutes=5) <= prev_close].reset_index(drop=True)
    elif forecast_type == "UPDATED":
        window_start_utc = pd.Timestamp(start).tz_convert("UTC")
        df = df.loc[df.timestamp < window_start_utc].reset_index(drop=True)
    now_utc = pd.Timestamp(generated_at.astimezone(timezone.utc))
    candle_ends = df.timestamp + pd.Timedelta(minutes=5)
    complete = candle_ends <= now_utc
    df = df.loc[complete].reset_index(drop=True)
    if df.empty:
        raise ValueError("No completed candles available")
    latest = df.timestamp.iloc[-1]
    if latest.tz_convert(ET).date() == eval_day and latest + pd.Timedelta(minutes=5) > pd.Timestamp(start).tz_convert("UTC"):
        raise ValueError("Input includes a candle from inside the forecast window")
    if latest >= pd.Timestamp(start).tz_convert("UTC"):
        raise ValueError("Latest candle timestamp must precede the forecast window")
    latest_end_et = (latest + pd.Timedelta(minutes=5)).tz_convert(ET)
    window_start_minutes = int(window["start"].split(":")[0]) * 60 + int(window["start"].split(":")[1])
    if forecast_type == "ADVANCE" or window_start_minutes == 9 * 60 + 30:
        expected_day = previous_trading_day(eval_day)
        close_et = market_timestamp(expected_day, session_close(expected_day).strftime("%H:%M"))
        minimum_end = close_et
        required_start = market_timestamp(expected_day, "09:30")
        required_end = close_et - pd.Timedelta(minutes=5)
    else:
        expected_day = eval_day
        minimum_end = (market_timestamp(eval_day, "12:30") if forecast_type == "UPDATED"
                       else start - pd.Timedelta(minutes=5))
        required_start = market_timestamp(eval_day, "09:30")
        required_end = (market_timestamp(eval_day, "12:25") if forecast_type == "UPDATED"
                        else latest.tz_convert(ET))
    if forecast_type == "UPDATED":
        morning_times = pd.date_range(pd.Timestamp(market_timestamp(eval_day, "09:30")).tz_convert("UTC"),
                                      pd.Timestamp(market_timestamp(eval_day, "12:25")).tz_convert("UTC"), freq="5min")
        observed = set(df.loc[pd.to_datetime(df.timestamp, utc=True).dt.tz_convert(ET).dt.date == eval_day, "timestamp"])
        if set(morning_times) - observed:
            raise ValueError(f"Morning window data is incomplete: missing {len(set(morning_times)-observed)} completed 5-minute candle(s)")
    if latest_end_et.date() != expected_day or latest_end_et < minimum_end:
        raise ValueError("Latest completed candle is stale or delayed for this forecast window")
    session_rows = df.loc[pd.to_datetime(df.timestamp, utc=True).dt.tz_convert(ET).dt.date == expected_day, "timestamp"]
    expected_session = pd.date_range(start=pd.Timestamp(required_start).tz_convert("UTC"),
                                     end=pd.Timestamp(required_end).tz_convert("UTC"), freq="5min")
    available = set(session_rows.tolist())
    missing = [t for t in expected_session if t not in available]
    if missing:
        raise ValueError(f"Market data has {len(missing)} missing 5-minute candle(s) before forecast issuance")

    bundle, metadata = load_daily_model(evaluation_date, window_id)
    cutoff = pd.Timestamp(metadata["training_cutoff"])
    if pd.Timestamp(metadata["dataset_end"]) > cutoff:
        raise RuntimeError("Registered model metadata violates its training cutoff")
    feat = generate_features(df, reset_daily=False)
    feature_cols = bundle["feature_columns"]
    x = feat.iloc[[-1]][feature_cols].replace([np.inf, -np.inf], np.nan)
    medians = pd.Series(bundle["medians"])
    x = x.fillna(medians).fillna(0.0)
    models = bundle["models"]
    ref = float(df.iloc[-1]["close"])
    end_return = float(models["end_return"].predict(x.to_numpy(float))[0])
    low_return = float(models["low_return"].predict(x.to_numpy(float))[0])
    high_return = float(models["high_return"].predict(x.to_numpy(float))[0])
    raw_direction_probability = float(models["direction"].predict_proba(x.to_numpy(float))[0, 1])
    direction_calibrator = bundle.get("direction_calibrator")
    if direction_calibrator is not None:
        clipped = float(np.clip(raw_direction_probability, 1e-6, 1.0 - 1e-6))
        logit = np.asarray([[np.log(clipped / (1.0 - clipped))]])
        direction_probability: float | None = float(direction_calibrator.predict_proba(logit)[0, 1])
        confidence = {"direction_up_probability": direction_probability, "method": "platt_calibrated_expanding_fold_predictions",
                      "calibration_evaluation": metadata.get("walk_forward_evaluation", {}).get(window_id, {}).get("direction_probability_calibration")}
    else:
        direction_probability = None
        confidence = {"direction_up_probability": None, "method": "unavailable_no_calibrator",
                      "raw_model_score_up": raw_direction_probability}
    expected_price = ref * (1.0 + end_return)
    lower, upper = sorted((ref * (1.0 + low_return), ref * (1.0 + high_return)))
    direction = "UP" if (direction_probability if direction_probability is not None else raw_direction_probability) >= 0.5 else "DOWN"

    # The history passed to level selection ends at latest completed candle, before window start.
    history = df.iloc[:-1].copy()
    atr = float(feat.iloc[-1].get("atr_14", np.nan))
    levels = find_levels(history, ref, atr, cfg["support_lookback_sessions"],
                         cfg["swing_left_bars"], cfg["swing_right_bars"], cfg["level_atr_distance"])
    hold = support_hold_probability(bundle["calibration"], levels.get("support_distance"), atr,
                                    bundle["calibration_minimum"])
    if levels.get("support") is not None:
        levels["expected_to_hold"] = hold["probability"] is not None and hold["probability"] >= 0.5
    else:
        levels["expected_to_hold"] = None
    downside = downside_scenario(history, levels.get("support"), ref, atr, levels)

    metadata_hash = metadata.get("model_artifact_sha256", {}).get(window_id)
    payload = {
        "forecast_id": str(uuid4()), "evaluation_date": evaluation_date, "window_id": window_id,
        "forecast_type": forecast_type, "revision": 1, "parent_forecast_id": parent_forecast_id,
        "status": "FROZEN", "model_artifact_sha256": metadata_hash,
        "window_start": start.isoformat(), "window_end": end.isoformat(),
        "generated_at": generated_at.astimezone(timezone.utc).isoformat(),
        "data_cutoff": latest.isoformat(), "feature_generation_timestamp": latest.isoformat(),
        "training_cutoff": metadata["training_cutoff"], "model_version": metadata["model_version"],
        "reference_price": ref, "predicted_direction": direction,
        "direction_probability_up": direction_probability,
        "confidence_estimates": confidence,
        "expected_price_movement": expected_price - ref, "expected_price": expected_price,
        "predicted_range": {"lower": lower, "upper": upper},
        "support_resistance": levels, "support_holding": hold,
        "conditional_downside": downside, "estimate_only": True,
    }
    stages = [
        ("data_cutoff", {"evaluation_date": evaluation_date}, {"training_cutoff": metadata["training_cutoff"]}, "previous_trading_day_close"),
        ("data_retrieval", {"latest_candle": latest.isoformat()}, {"completed_candles": len(df)}, "completed_5m_candles_only"),
        ("preprocessing", {"rows": len(df)}, {"timezone": "UTC", "sorted": True}, "existing_validation_and_causal_features"),
        ("feature_generation", {"latest_timestamp": latest.isoformat()}, {"feature_count": len(feature_cols)}, "36_feature_whitelist"),
        ("model_prediction", {"reference_price": ref}, {"direction": direction, "range": payload["predicted_range"]}, "direct_window_model"),
        ("support_resistance", {"reference_price": ref}, levels, "confirmed_prior_swings_atr_filtered"),
        ("confidence", {"support_distance": levels.get("support_distance")}, hold, "conditional_historical_frequency"),
        ("chart_output", {"forecast_id": payload["forecast_id"], "forecast_type": forecast_type}, {"status": "frozen"}, "append_only_forecast"),
    ]
    issue_forecast(payload, logs=stages)
    return payload


def compare_completed_forecast(forecast_id: str, candles: pd.DataFrame,
                               now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    forecast = get_forecast(forecast_id)
    if forecast is None:
        raise KeyError(f"Unknown forecast_id: {forecast_id}")
    if forecast.get("actual"):
        raise ValueError("This forecast already has an immutable actual comparison")
    start, end = pd.Timestamp(forecast["window_start"]), pd.Timestamp(forecast["window_end"])
    if pd.Timestamp(now.astimezone(timezone.utc)) < end.tz_convert("UTC"):
        raise ValueError("Cannot compare before the forecast window has ended")
    df = candles.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    df = df.loc[(df.timestamp >= start.tz_convert("UTC")) & (df.timestamp < end.tz_convert("UTC"))]
    expected = int((end - start).total_seconds() // 300)
    expected_times = set(pd.date_range(start=start.tz_convert("UTC"), periods=expected, freq="5min"))
    if len(df) != expected or df.timestamp.duplicated().any() or set(df.timestamp) != expected_times:
        raise ValueError(f"Actual window incomplete: expected {expected} candles, received {len(df)}")
    ref = float(forecast["reference_price"])
    actual_low, actual_high, actual_end = float(df.low.min()), float(df.high.max()), float(df.close.iloc[-1])
    predicted_direction = forecast["predicted_direction"]
    actual_direction = "UP" if actual_end >= ref else "DOWN"
    bounds = forecast["predicted_range"]
    sr = forecast["support_resistance"]
    actual = {
        "compared_at": now.astimezone(timezone.utc).isoformat(), "actual_range": {"lower": actual_low, "upper": actual_high},
        "actual_end_price": actual_end, "actual_direction": actual_direction,
        "direction_correct": predicted_direction == actual_direction,
        "predicted_range_coverage": actual_low >= bounds["lower"] and actual_high <= bounds["upper"],
        "range_boundary_errors": {"lower": actual_low - bounds["lower"], "upper": actual_high - bounds["upper"]},
        "movement_error": (actual_end - ref) - forecast["expected_price_movement"],
        "support_held": bool(sr.get("support") is None or float(df.close.min()) >= sr["support"]),
        "resistance_reached": bool(sr.get("resistance") is not None and float(df.high.max()) >= sr["resistance"]),
        "mae_boundary": (abs(actual_low - bounds["lower"]) + abs(actual_high - bounds["upper"])) / 2.0,
        "actual_candle_count": len(df),
    }
    record_actual(forecast_id, actual, log=("actual_comparison",
                                            {"forecast_id": forecast_id, "actual_candles": len(df)}, actual,
                                            "completed_window_only"))
    return {"forecast": forecast, "actual": actual}
