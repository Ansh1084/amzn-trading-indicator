"""Immutable 10-minute directional session forecasts using the recovered 40-feature model."""
from __future__ import annotations

import json
import hashlib
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from src.data.ingestion import latest_processed_path, load_local_candles
from src.features.legacy_engineering import FEATURE_COLUMNS, generate_features
from src.forecasts.calendar import ET, is_trading_day, market_timestamp, previous_trading_day, session_close
from src.forecasts.levels import find_levels
from src.forecasts.store import connect

ROOT = Path(__file__).resolve().parents[2]
LEGACY_FEATURES_PATH = ROOT / "artifacts" / "feature_columns.json"
MODEL_PATH = ROOT / "artifacts" / "model_lightgbm.joblib"
MODEL_VERSION = "legacy-lightgbm-10m-e1b5cf0"
LEGACY_MODELS = {"best_model.joblib": "legacy-histgradient-40f", "model_logisticregression.joblib": "legacy-logistic-40f",
                 "model_randomforest.joblib": "legacy-randomforest-40f", "model_histgradientboosting.joblib": "legacy-histgradient-40f",
                 "model_lightgbm.joblib": MODEL_VERSION}


def load_legacy_model(model_path: Path = MODEL_PATH):
    columns = json.loads(LEGACY_FEATURES_PATH.read_text(encoding="utf-8-sig"))
    if columns != FEATURE_COLUMNS or len(columns) != 40:
        raise ValueError("Recovered legacy feature pipeline does not match the saved ordered feature list")
    model = joblib.load(model_path)
    if int(model.n_features_in_) != len(columns) or not hasattr(model, "predict_proba"):
        raise ValueError("Legacy artifact has incompatible feature count or no class probabilities")
    classes = list(model.classes_)
    if set(classes) != {0.0, 1.0}:
        raise ValueError(f"Expected binary DOWN/UP classes, got {classes}")
    return model, columns, classes


def _artifact_sha256(path: Path = MODEL_PATH) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def interval_schedule(evaluation_date: str, session_id: str, interval_minutes: int = 10) -> list[tuple[datetime, datetime]]:
    day = date.fromisoformat(evaluation_date)
    if not is_trading_day(day):
        raise ValueError(f"{evaluation_date} is not a trading session")
    if session_id not in {"morning", "afternoon"}:
        raise ValueError("session_id must be morning or afternoon")
    start = market_timestamp(day, "09:30" if session_id == "morning" else "12:30")
    standard_end = market_timestamp(day, "12:30" if session_id == "morning" else "15:30")
    close = datetime.combine(day, session_close(day), tzinfo=ET)
    end = min(standard_end, close)
    intervals = []
    while start < end:
        next_start = min(start + timedelta(minutes=interval_minutes), end)
        intervals.append((start, next_start))
        start = next_start
    return intervals


def list_session_predictions(evaluation_date: str, session_id: str, forecast_type: str = "ADVANCE") -> list[dict]:
    with connect() as db:
        rows = db.execute("SELECT payload,actual_payload FROM session_predictions WHERE evaluation_date=? AND session_id=? AND forecast_type=? ORDER BY interval_start DESC,revision DESC",
                          (evaluation_date, session_id, forecast_type)).fetchall()
        return [{**json.loads(row[0]), "actual": json.loads(row[1]) if row[1] else None} for row in rows]


def generate_session_predictions(evaluation_date: str, session_id: str, forecast_type: str = "ADVANCE") -> dict:
    forecast_type = forecast_type.upper()
    if forecast_type not in {"ADVANCE", "UPDATED"}:
        raise ValueError("forecast_type must be ADVANCE or UPDATED")
    day = date.fromisoformat(evaluation_date)
    if forecast_type == "UPDATED" and session_id != "afternoon":
        raise ValueError("UPDATED forecasts are supported only for afternoon")
    session_start = market_timestamp(day, "09:30" if session_id == "morning" else "12:30")
    if forecast_type == "ADVANCE" and datetime.now(ET) >= session_start:
        raise ValueError("ADVANCE session predictions must be generated before the target session starts")
    candles = load_local_candles(latest_processed_path()).sort_values("timestamp").reset_index(drop=True)
    candles["timestamp"] = pd.to_datetime(candles.timestamp, utc=True)
    if forecast_type == "ADVANCE":
        # Both ADVANCE schedules are based exclusively on the completed prior
        # session. Select the cutoff before filtering so afternoon's same-day
        # morning candles cannot slip into the training snapshot.
        prior = previous_trading_day(day)
        cutoff = datetime.combine(prior, session_close(prior), tzinfo=ET)
    else:
        cutoff = market_timestamp(day, "12:30")
    eligible = candles.loc[candles.timestamp < cutoff.astimezone(timezone.utc)]
    if eligible.empty:
        raise ValueError("No completed market data is available before this forecast cutoff")
    if forecast_type == "ADVANCE":
        expected_last = (cutoff - timedelta(minutes=5)).astimezone(timezone.utc)
        if pd.Timestamp(eligible.timestamp.iloc[-1]) != pd.Timestamp(expected_last):
            raise ValueError("ADVANCE requires the previous session's final completed 5-minute candle")
    elif datetime.now(ET) < cutoff:
        raise ValueError("UPDATED afternoon predictions require completed morning-session data")
    if forecast_type == "UPDATED":
        morning_start = market_timestamp(day, "09:30").astimezone(timezone.utc)
        morning_end = market_timestamp(day, "12:30").astimezone(timezone.utc)
        morning = candles.loc[(candles.timestamp >= morning_start) & (candles.timestamp < morning_end)]
        expected_morning = pd.date_range(morning_start, morning_end - pd.Timedelta(minutes=5), freq="5min")
        if len(morning) != len(expected_morning) or not morning.timestamp.reset_index(drop=True).equals(pd.Series(expected_morning)):
            raise ValueError(f"UPDATED forecast requires 36 completed morning 5-minute bars; found {len(morning)}")
    model, columns, classes = load_legacy_model()
    model_sha256 = _artifact_sha256()
    features = generate_features(eligible, reset_daily=False)
    row = features.iloc[-1]
    if not np.isfinite(row[columns].to_numpy(dtype=float)).all():
        raise ValueError("Latest causal legacy feature row contains missing values")
    base = row[columns].to_dict()
    history = eligible
    atr = float(row.get("atr_14", np.nan))
    levels = find_levels(history, float(row.close), atr)
    intervals = interval_schedule(evaluation_date, session_id)
    generated = datetime.now(timezone.utc).isoformat()
    version = f"{MODEL_VERSION}:{model_sha256[:12]}"
    results = []
    for start, end in intervals:
        local = start.astimezone(ET)
        values = dict(base)
        values["hour"], values["minute"], values["day_of_week"] = local.hour, local.minute, local.weekday()
        minute = local.hour * 60 + local.minute
        values["time_of_day_sin"] = float(np.sin(2 * np.pi * minute / 1440))
        values["time_of_day_cos"] = float(np.cos(2 * np.pi * minute / 1440))
        x = pd.DataFrame([[values[c] for c in columns]], columns=columns, dtype=float)
        probabilities = model.predict_proba(x)[0]
        up = float(probabilities[classes.index(1.0)])
        signal = "UP" if model.predict(x)[0] == 1 else "DOWN"
        p = {"prediction_id": str(uuid4()), "evaluation_date": evaluation_date, "session_id": session_id,
             "forecast_type": forecast_type, "revision": 1, "interval_start": start.isoformat(), "interval_end": end.isoformat(),
             "generated_at": generated, "feature_timestamp": eligible.timestamp.iloc[-1].isoformat(),
             "reference_price": float(row.close), "predicted_signal": signal, "predicted_direction": signal,
             "raw_probability_up": up,
             "confidence": None, "confidence_status": "uncalibrated_raw_model_score_not_confidence",
             "support": levels.get("support"), "resistance": levels.get("resistance"),
             "model_name": type(model).__name__, "model_version": version, "model_sha256": model_sha256,
             "input_feature_count": len(columns),
             "data_cutoff": eligible.timestamp.iloc[-1].isoformat(), "actual": None}
        results.append(p)
    with connect() as db:
        for p in results:
            db.execute("INSERT OR IGNORE INTO session_predictions(prediction_id,evaluation_date,session_id,forecast_type,revision,interval_start,interval_end,generated_at,feature_timestamp,model_name,model_version,payload) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                       (p["prediction_id"],evaluation_date,session_id,forecast_type,1,p["interval_start"],p["interval_end"],generated,p["feature_timestamp"],p["model_name"],version,json.dumps(p)))
    stored = list_session_predictions(evaluation_date, session_id, forecast_type)
    return {"evaluation_date": evaluation_date, "session_id": session_id, "forecast_type": forecast_type,
            "model_version": version, "interval_count": len(intervals), "predictions": stored, "idempotent": True,
            "confidence_note": "Legacy model probabilities are uncalibrated and are shown as raw UP scores, not confidence."}


def evaluate_session_predictions(evaluation_date: str, session_id: str, forecast_type: str = "ADVANCE") -> dict:
    candles = load_local_candles(latest_processed_path()).copy()
    candles["timestamp"] = pd.to_datetime(candles.timestamp, utc=True)
    predictions = list(reversed(list_session_predictions(evaluation_date, session_id, forecast_type)))
    results = []
    with connect() as db:
        for prediction in predictions:
            if prediction["actual"]:
                results.append(prediction); continue
            start, end = pd.Timestamp(prediction["interval_start"]).tz_convert("UTC"), pd.Timestamp(prediction["interval_end"]).tz_convert("UTC")
            # Candle timestamps mark the bar's opening time. The endpoint bar
            # at ``end`` must itself finish before its close can be used, so a
            # 10-minute target based on close[t] and close[t+10m] is not
            # evaluable until five minutes after the endpoint timestamp.
            endpoint_complete_at = end + pd.Timedelta(minutes=5)
            if datetime.now(timezone.utc) < endpoint_complete_at.to_pydatetime():
                continue
            # Match the original target exactly: close at t versus close at t+10m.
            # This requires a real endpoint candle; absent/delayed endpoints stay pending.
            bars = candles.loc[(candles.timestamp >= start) & (candles.timestamp <= end)]
            expected = int((end - start).total_seconds() // 300) + 1
            if len(bars) != expected or bars.timestamp.iloc[0] != start or bars.timestamp.iloc[-1] != end:
                continue
            first, last = float(bars.close.iloc[0]), float(bars.close.iloc[-1])
            actual_signal = "UP" if last > first else "DOWN" if last < first else "FLAT"
            actual = {"actual_signal": actual_signal, "actual_direction": actual_signal, "actual_close": last,
                      "actual_return_pct": (last / first - 1) * 100,
                      "outcome": "CORRECT" if prediction["predicted_signal"] == actual_signal else "INCORRECT",
                      "correct": prediction["predicted_signal"] == actual_signal,
                      "evaluated_at": datetime.now(timezone.utc).isoformat(), "actual_bars": len(bars)}
            cursor = db.execute("UPDATE session_predictions SET actual_payload=? WHERE prediction_id=? AND actual_payload IS NULL",
                                (json.dumps(actual), prediction["prediction_id"]))
            if cursor.rowcount != 1:
                continue
            prediction["actual"] = actual
            results.append({**prediction, "actual": actual})
    final = _session_decision(predictions)
    completed = [p for p in predictions if p.get("actual")]
    return {"evaluation_date": evaluation_date, "session_id": session_id, "forecast_type": forecast_type,
            "predictions": results, "final_decision": final, "actual_session": _actual_session(completed) if len(completed)==len(predictions) else None,
            "completed_count": len(completed), "pending_count": len(predictions)-len(completed)}


def session_actual_candles(evaluation_date: str, session_id: str) -> list[dict]:
    """Aggregate only genuine, complete pairs of 5-minute bars into 10-minute OHLC."""
    candles = load_local_candles(latest_processed_path()).copy()
    candles["timestamp"] = pd.to_datetime(candles.timestamp, utc=True)
    result = []
    for start, end in interval_schedule(evaluation_date, session_id):
        bars = candles.loc[(candles.timestamp >= start.astimezone(timezone.utc)) &
                           (candles.timestamp < end.astimezone(timezone.utc))]
        expected = int((end-start).total_seconds() // 300)
        if len(bars) != expected:
            continue
        result.append({"time": start.isoformat(), "interval_end": end.isoformat(),
                       "open": float(bars.open.iloc[0]), "high": float(bars.high.max()),
                       "low": float(bars.low.min()), "close": float(bars.close.iloc[-1]),
                       "volume": float(bars.volume.sum())})
    return result


def auto_evaluate_available(evaluation_date: str | None = None) -> int:
    """Poll persisted predictions against local completed candles; missing intervals stay pending."""
    with connect() as db:
        if evaluation_date:
            rows = db.execute("SELECT DISTINCT evaluation_date,session_id,forecast_type FROM session_predictions WHERE evaluation_date=?", (evaluation_date,)).fetchall()
        else:
            rows = db.execute("SELECT DISTINCT evaluation_date,session_id,forecast_type FROM session_predictions ORDER BY evaluation_date DESC LIMIT 40").fetchall()
    completed = 0
    for row in rows:
        before = len([p for p in list_session_predictions(row[0], row[1], row[2]) if p.get("actual")])
        result = evaluate_session_predictions(row[0], row[1], row[2])
        completed += max(0, result["completed_count"] - before)
    return completed


def _session_decision(predictions: list[dict]) -> dict:
    counts = {signal: sum(p["predicted_signal"] == signal for p in predictions) for signal in ("UP", "DOWN")}
    decision = "UP" if counts["UP"] > counts["DOWN"] else "DOWN" if counts["DOWN"] > counts["UP"] else "NEUTRAL"
    return {"decision": decision, "method": "majority_interval_vote_tie_is_neutral", "up_count": counts["UP"],
            "down_count": counts["DOWN"], "neutral_count": 0, "confidence": None,
            "confidence_status": "unavailable_uncalibrated_model"}


def _actual_session(completed: list[dict]) -> dict | None:
    if not completed: return None
    return_pct = (float(np.prod([1.0 + p["actual"]["actual_return_pct"] / 100.0 for p in completed])) - 1.0) * 100.0
    actual = "UP" if return_pct > 0 else "DOWN" if return_pct < 0 else "FLAT"
    return {"decision": actual, "actual_intervals": len(completed),
            "return_pct": return_pct}
