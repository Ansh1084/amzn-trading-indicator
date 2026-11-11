"""Train separate direct models for configured intraday forecast windows."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from uuid import uuid4
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score, mean_absolute_error
from sklearn.linear_model import LogisticRegression

from src.features.engineering import generate_features, get_feature_columns
from src.forecasts.calendar import cutoff_for_evaluation, is_trading_day, market_timestamp, previous_trading_day, session_close
from src.forecasts.levels import find_levels

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "forecast_windows.json"
ARTIFACTS = ROOT / "artifacts" / "window_models"


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _examples(df: pd.DataFrame, features: pd.DataFrame, window: dict[str, str], cutoff_day: date,
              config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    tz = config["timezone"]
    lead = int(config.get("issuance_lead_minutes", 2))
    local = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(tz)
    rows, labels, calibration = [], [], []
    for day in sorted(set(local.dt.date)):
        if day > cutoff_day or not is_trading_day(day):
            continue
        start = market_timestamp(day, window["start"])
        end = market_timestamp(day, window["end"])
        close = market_timestamp(day, session_close(day).strftime("%H:%M"))
        if end > close:
            continue
        # Candle timestamps mark candle starts. The anchor candle must be completed by start.
        issue_at = start - pd.Timedelta(minutes=lead)
        anchor_candidates = np.flatnonzero((local + pd.Timedelta(minutes=5) <= issue_at) & (local.dt.date == day))
        if not len(anchor_candidates):
            anchor_candidates = np.flatnonzero((local + pd.Timedelta(minutes=5) <= issue_at) & (local.dt.date < day))
        if not len(anchor_candidates):
            continue
        anchor = int(anchor_candidates[-1])
        if start.hour == 9 and start.minute == 30:
            anchor_day = previous_trading_day(day)
            anchor_close = market_timestamp(anchor_day, session_close(anchor_day).strftime("%H:%M"))
            expected_anchor = pd.Timestamp(anchor_close - pd.Timedelta(minutes=5)).tz_convert("UTC")
            session_open = pd.Timestamp(market_timestamp(anchor_day, "09:30")).tz_convert("UTC")
        else:
            anchor_day = day
            bar_offset = ((lead + 5 + 4) // 5) * 5
            expected_anchor = pd.Timestamp(start - pd.Timedelta(minutes=bar_offset)).tz_convert("UTC")
            session_open = pd.Timestamp(market_timestamp(day, "09:30")).tz_convert("UTC")
        if pd.Timestamp(df.iloc[anchor]["timestamp"]) != expected_anchor:
            continue
        expected_anchor_candles = set(pd.date_range(session_open, expected_anchor, freq="5min"))
        observed_anchor_candles = set(pd.to_datetime(df.loc[(local.dt.date == anchor_day) &
                                                              (local >= session_open) & (local <= expected_anchor), "timestamp"], utc=True))
        if expected_anchor_candles != observed_anchor_candles:
            continue
        future_ix = np.flatnonzero((local >= start) & (local < end))
        if not len(future_ix):
            continue
        # Require all expected five-minute bars so incomplete historical windows are excluded.
        expected = int((end - start).total_seconds() // 300)
        if len(future_ix) != expected:
            continue
        ref = float(df.iloc[anchor]["close"])
        future = df.iloc[future_ix]
        last = float(future.iloc[-1]["close"])
        lo, hi = float(future["low"].min()), float(future["high"].max())
        if ref <= 0:
            continue
        rows.append(features.iloc[anchor][get_feature_columns()].to_dict())
        labels.append({"evaluation_date": day.isoformat(), "end_return": (last / ref) - 1,
                       "low_return": (lo / ref) - 1, "high_return": (hi / ref) - 1,
                       "direction": int(last >= ref)})
        history = df.iloc[max(0, anchor - 4000):anchor]
        atr = float(features.iloc[anchor].get("atr_14", np.nan))
        levels = find_levels(history, ref, atr, config["support_lookback_sessions"],
                             config["swing_left_bars"], config["swing_right_bars"], config["level_atr_distance"])
        support = levels.get("support")
        down_close = float(future["close"].min())
        calibration.append({"evaluation_date": day.isoformat(),
                            "break": bool(support is not None and down_close < support),
                            "distance_atr": ((ref-support)/atr if support is not None and np.isfinite(atr) and atr > 0 else None)})
    return pd.DataFrame(rows), pd.DataFrame(labels), calibration


def _walk_forward_support_calibration(samples: list[dict[str, Any]], minimum: int) -> dict[str, Any]:
    scored = []
    eligible = [s for s in samples if s.get("distance_atr") is not None]
    for i, sample in enumerate(eligible):
        past = [x for x in eligible[:i] if abs(x["distance_atr"] - sample["distance_atr"]) <= 0.5]
        if len(past) < minimum:
            continue
        holds = sum(not bool(x["break"]) for x in past)
        probability = (holds + 0.5) / (len(past) + 1.0)
        outcome = float(not bool(sample["break"]))
        scored.append((probability - outcome) ** 2)
    return {"method": "expanding_conditional_frequency", "scored_samples": len(scored),
            "coverage": len(scored) / len(eligible) if eligible else 0.0,
            "brier_score": float(np.mean(scored)) if scored else None,
            "minimum_prior_samples": minimum}


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def _train_for_evaluation_date_impl(evaluation_date: str, data_path: str | Path | None = None,
                                   progress=None) -> dict[str, Any]:
    eval_day = date.fromisoformat(evaluation_date)
    if not is_trading_day(eval_day):
        raise ValueError(f"{evaluation_date} is not a scheduled trading day")
    cutoff_day = previous_trading_day(eval_day)
    cutoff = cutoff_for_evaluation(eval_day)
    cfg = load_config()
    source = Path(data_path) if data_path else ROOT / "data" / "processed" / "AMZN_clean.csv"
    candles = pd.read_csv(source)
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True, errors="raise")
    candles = candles.loc[candles["timestamp"] <= pd.Timestamp(cutoff)].sort_values("timestamp").reset_index(drop=True)
    if candles.empty:
        raise ValueError("No training observations are available by the required cutoff")
    previous_close = market_timestamp(cutoff_day, session_close(cutoff_day).strftime("%H:%M"))
    required_last_bar = pd.Timestamp(previous_close - pd.Timedelta(minutes=5)).tz_convert("UTC")
    if candles["timestamp"].iloc[-1] != required_last_bar:
        raise ValueError(f"Training data is stale or incomplete: expected last completed bar {required_last_bar.isoformat()}, "
                         f"received {candles['timestamp'].iloc[-1].isoformat()}")
    if progress: progress("training_data_and_cutoff_verified", .12)
    eng = generate_features(candles, reset_daily=False)
    feature_cols = get_feature_columns()
    version = f"{eval_day.isoformat()}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output_dir = ARTIFACTS / eval_day.isoformat()
    if output_dir.exists():
        raise FileExistsError(f"Forecast model artifacts already exist for {eval_day}; refusing replacement")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    staging_dir = ARTIFACTS / f".{eval_day.isoformat()}.{uuid4().hex}.staging"
    staging_dir.mkdir()
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        raise FileExistsError(f"A model is already registered for {eval_day}; refusing silent replacement")
    evaluation_metrics: dict[str, Any] = {}
    model_hashes: dict[str, str] = {}
    for window_index, window in enumerate(cfg["windows"]):
        if progress: progress(f"training_{window['id']}_window_models", .15 + .7 * window_index / len(cfg["windows"]))
        X, y, calibration = _examples(candles, eng, window, cutoff_day, cfg)
        if len(X) < 50:
            raise ValueError(f"Only {len(X)} complete historical examples for {window['id']}; at least 50 required")
        dates = y.pop("evaluation_date").to_numpy()
        X = X.replace([np.inf, -np.inf], np.nan)
        medians = X.median().fillna(0.0)
        X = X.fillna(medians)
        X_np = X.to_numpy(float)
        # Expanding-window folds: each validation block occurs strictly after its training block.
        # Model settings are fixed in code and are not tuned on these fold outcomes.
        fold_results = []
        oof_probabilities: list[float] = []
        oof_directions: list[int] = []
        n = len(X)
        boundaries = [int(n * 0.55), int(n * 0.70), int(n * 0.85), n]
        previous = int(n * 0.40)
        for fold, boundary in enumerate(boundaries):
            train_end = previous if fold == 0 else boundaries[fold - 1]
            if boundary <= train_end or train_end < 30:
                previous = boundary
                continue
            clf = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15, l2_regularization=1.0, random_state=42)
            reg = HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=15, l2_regularization=1.0, random_state=42)
            low_q = GradientBoostingRegressor(loss="quantile", alpha=0.1, n_estimators=100, max_depth=2, random_state=42)
            high_q = GradientBoostingRegressor(loss="quantile", alpha=0.9, n_estimators=100, max_depth=2, random_state=42)
            clf.fit(X_np[:train_end], y["direction"].to_numpy()[:train_end])
            reg.fit(X_np[:train_end], y["end_return"].to_numpy()[:train_end])
            low_q.fit(X_np[:train_end], y["low_return"].to_numpy()[:train_end])
            high_q.fit(X_np[:train_end], y["high_return"].to_numpy()[:train_end])
            ix = slice(train_end, boundary)
            predicted = clf.predict(X_np[ix])
            fold_probabilities = clf.predict_proba(X_np[ix])[:, 1]
            truth = y["direction"].to_numpy()[ix]
            train_labels = y["direction"].to_numpy()[:train_end]
            majority_direction = int(np.mean(train_labels) >= 0.5)
            oof_probabilities.extend(fold_probabilities.tolist())
            oof_directions.extend(truth.astype(int).tolist())
            low_pred, high_pred = low_q.predict(X_np[ix]), high_q.predict(X_np[ix])
            low_true, high_true = y["low_return"].to_numpy()[ix], y["high_return"].to_numpy()[ix]
            fold_results.append({
                "fold": fold + 1, "train_end_date": dates[train_end - 1],
                "validation_start_date": dates[train_end], "validation_end_date": dates[boundary - 1],
                "n_validation": int(boundary - train_end), "direction_accuracy": float(accuracy_score(truth, predicted)),
                "majority_baseline_accuracy": float(np.mean(truth == majority_direction)),
                "direction_balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
                "direction_f1": float(f1_score(truth, predicted, zero_division=0)),
                "end_return_mae": float(mean_absolute_error(y["end_return"].to_numpy()[ix], reg.predict(X_np[ix]))),
                "range_coverage": float(np.mean((low_true >= low_pred) & (high_true <= high_pred))),
            })
            previous = boundary
            if progress: progress(f"validating_{window['id']}_fold_{fold + 1}",
                                  .2 + .65 * ((window_index + (fold + 1) / len(boundaries)) / len(cfg["windows"])))
        probability_calibrator = None
        calibration_report: dict[str, Any] = {"method": "platt_sigmoid_on_expanding_fold_predictions",
                                               "evaluation_samples": 0, "raw_brier_score": None,
                                               "calibrated_brier_score": None}
        if len(oof_probabilities) >= 40 and len(set(oof_directions)) == 2:
            oof_p = np.asarray(oof_probabilities, dtype=float)
            oof_y = np.asarray(oof_directions, dtype=int)
            split = max(1, int(len(oof_p) * 0.70))
            eval_p, eval_y = oof_p[split:], oof_y[split:]
            if len(eval_p) > 0:
                validation_calibrator = LogisticRegression(random_state=42)
                validation_calibrator.fit(_logit(oof_p[:split]), oof_y[:split])
                calibrated_eval = validation_calibrator.predict_proba(_logit(eval_p))[:, 1]
                calibration_report.update({"evaluation_samples": int(len(eval_p)),
                                           "raw_brier_score": float(brier_score_loss(eval_y, eval_p)),
                                           "calibrated_brier_score": float(brier_score_loss(eval_y, calibrated_eval))})
            probability_calibrator = LogisticRegression(random_state=42)
            probability_calibrator.fit(_logit(oof_p), oof_y)
        evaluation_metrics[window["id"]] = {"validation_method": "expanding_chronological_folds",
                                              "folds": fold_results,
                                              "fold_count": len(fold_results),
                                              "direction_probability_calibration": calibration_report,
                                              "support_probability": _walk_forward_support_calibration(
                                                  calibration, cfg["minimum_calibration_samples"])}
        bundle = {
            "end_return": HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=15, l2_regularization=1.0, random_state=42),
            "low_return": GradientBoostingRegressor(loss="quantile", alpha=0.1, n_estimators=100, max_depth=2, random_state=42),
            "high_return": GradientBoostingRegressor(loss="quantile", alpha=0.9, n_estimators=100, max_depth=2, random_state=42),
            "direction": HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15, l2_regularization=1.0, random_state=42),
        }
        for key, model in bundle.items():
            model.fit(X_np, y[key].to_numpy())
        artifact = {"models": bundle, "direction_calibrator": probability_calibrator,
                    "medians": medians.to_dict(), "feature_columns": feature_cols,
                    "window": window, "calibration": [x for x in calibration if x.get("distance_atr") is not None],
                    "calibration_minimum": cfg["minimum_calibration_samples"]}
        artifact_path = staging_dir / f"{window['id']}.joblib"
        joblib.dump(artifact, artifact_path)
        model_hashes[window["id"]] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    metadata = {"model_version": version, "evaluation_date": eval_day.isoformat(),
                "training_cutoff": cutoff.isoformat(), "dataset_start": candles.timestamp.min().isoformat(),
                "dataset_end": candles.timestamp.max().isoformat(), "dataset_rows": len(candles),
                "feature_columns": feature_cols, "config": cfg, "artifact_dir": str(output_dir),
                "model_artifact_sha256": model_hashes,
                "model_configuration": {"classifier": "HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=15,l2=1)",
                                        "expected_move": "HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=15,l2=1)",
                                        "range": "GradientBoostingRegressor quantiles 0.10 and 0.90"},
                "trained_at": datetime.now(timezone.utc).isoformat(), "target": "window_end_return_and_10_90pct_window_extremes"}
    metadata["walk_forward_evaluation"] = evaluation_metrics
    (staging_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    try:
        staging_dir.replace(output_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return metadata


def train_for_evaluation_date(evaluation_date: str, data_path: str | Path | None = None,
                             progress=None) -> dict[str, Any]:
    """Train atomically and remove only staging output created by this call on failure."""
    eval_day = date.fromisoformat(evaluation_date)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    before = set(ARTIFACTS.glob(f".{eval_day.isoformat()}.*.staging"))
    try:
        return _train_for_evaluation_date_impl(evaluation_date, data_path, progress)
    except Exception:
        for candidate in set(ARTIFACTS.glob(f".{eval_day.isoformat()}.*.staging")) - before:
            shutil.rmtree(candidate, ignore_errors=True)
        raise


def load_daily_model(evaluation_date: str, window_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    meta_path = ARTIFACTS / evaluation_date / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No model version is registered for evaluation date {evaluation_date}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    artifact_path = ARTIFACTS / evaluation_date / f"{window_id}.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError(f"No model artifact for window {window_id} on {evaluation_date}")
    expected_hash = metadata.get("model_artifact_sha256", {}).get(window_id)
    observed_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    if not expected_hash or expected_hash != observed_hash:
        raise RuntimeError("Model artifact integrity check failed; refusing to load a changed model")
    return joblib.load(artifact_path), metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation_date", help="YYYY-MM-DD; cutoff is the previous trading session")
    parser.add_argument("--data-path")
    args = parser.parse_args()
    print(json.dumps(train_for_evaluation_date(args.evaluation_date, args.data_path), indent=2))
