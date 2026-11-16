"""Live Out-of-Sample Production ML Model Performance Evaluation System.

Evaluates the frozen 40-feature LightGBM binary classifier (predicting next 10-minute
price movement direction: UP vs DOWN) on genuine completed live candles and subsequent
observed market outcomes.

Zero fabrication, zero data leakage, and strict out-of-sample resolution.
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, time, timezone
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from backend.app.services.analysis.config import BAR_MINUTES, TIMEZONE
from src.forecasts.calendar import is_trading_day, session_close
from src.forecasts.legacy_sessions import MODEL_PATH, MODEL_VERSION, _artifact_sha256
from src.forecasts.store import connect

logger = logging.getLogger(__name__)

PREDICTION_HORIZON_MINUTES = 10
ET = ZoneInfo(TIMEZONE)


def ensure_ml_predictions_table(db) -> None:
    """Ensure the additive ml_predictions table exists in the SQLite store."""
    db.executescript("""
      CREATE TABLE IF NOT EXISTS ml_predictions (
        prediction_id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL DEFAULT 'AMZN',
        model_name TEXT NOT NULL,
        model_version TEXT NOT NULL,
        artifact_sha256 TEXT,
        prediction_timestamp TEXT NOT NULL,
        candle_timestamp TEXT NOT NULL,
        reference_price REAL NOT NULL,
        horizon_minutes INTEGER NOT NULL DEFAULT 10,
        predicted_direction TEXT NOT NULL,
        probability_up REAL NOT NULL,
        probability_down REAL NOT NULL,
        target_candle_timestamp TEXT NOT NULL,
        actual_future_price REAL,
        actual_direction TEXT,
        forward_return_pct REAL,
        status TEXT NOT NULL DEFAULT 'PENDING',
        exclusion_reason TEXT,
        resolved_at TEXT,
        UNIQUE(candle_timestamp, model_version)
      );
      CREATE INDEX IF NOT EXISTS ix_ml_predictions_status
        ON ml_predictions(status, candle_timestamp DESC);
      CREATE INDEX IF NOT EXISTS ix_ml_predictions_resolved
        ON ml_predictions(resolved_at DESC);
    """)


def save_ml_prediction(
    ml_evidence: dict,
    reference_price: float,
    symbol: str = "AMZN",
    db_path: str | Path | None = None,
) -> str | None:
    """Persist a live ML prediction with PENDING outcome status.

    Idempotent: unique on (candle_timestamp, model_version).
    """
    if not ml_evidence or not ml_evidence.get("available"):
        return None

    candle_ts = ml_evidence.get("candle_timestamp")
    if not candle_ts:
        return None

    candle_iso = pd.Timestamp(candle_ts).tz_convert("UTC").isoformat()
    target_iso = (pd.Timestamp(candle_ts).tz_convert("UTC") + pd.Timedelta(minutes=PREDICTION_HORIZON_MINUTES)).isoformat()
    gen_iso = pd.Timestamp(ml_evidence.get("generated_at", datetime.now(timezone.utc))).tz_convert("UTC").isoformat()

    model_name = str(ml_evidence.get("model_name") or "LGBMClassifier")
    model_version = str(ml_evidence.get("model_version") or MODEL_VERSION)
    sha256 = ml_evidence.get("artifact_sha256") or _artifact_sha256(MODEL_PATH)
    pred_dir = str(ml_evidence.get("direction")).upper()
    prob_up = float(ml_evidence.get("raw_probability_up"))
    prob_down = 1.0 - prob_up
    ref_price = float(reference_price)

    with connect(db_path) as db:
        ensure_ml_predictions_table(db)
        existing = db.execute(
            "SELECT prediction_id FROM ml_predictions WHERE candle_timestamp = ? AND model_version = ?",
            (candle_iso, model_version),
        ).fetchone()
        if existing:
            return existing[0]

        pred_id = str(uuid4())
        db.execute(
            """INSERT INTO ml_predictions (
                prediction_id, symbol, model_name, model_version, artifact_sha256,
                prediction_timestamp, candle_timestamp, reference_price, horizon_minutes,
                predicted_direction, probability_up, probability_down, target_candle_timestamp,
                status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')""",
            (
                pred_id, symbol, model_name, model_version, sha256,
                gen_iso, candle_iso, ref_price, PREDICTION_HORIZON_MINUTES,
                pred_dir, prob_up, prob_down, target_iso,
            ),
        )
        return pred_id


def resolve_ml_predictions(
    candles: pd.DataFrame | list[dict] | None = None,
    as_of: datetime | None = None,
    db_path: str | Path | None = None,
) -> dict:
    """Resolve pending live ML predictions using completed subsequent market candles.

    Strict 10-minute horizon:
      - Feature candle at T0
      - Target candle at T0 + 10m
      - Outcome: UP if target_close > ref_close, DOWN if target_close < ref_close
      - Handles market sessions, closures, early closes, missing bars, and data gaps.
    """
    with connect(db_path) as db:
        ensure_ml_predictions_table(db)
        pending = db.execute(
            "SELECT * FROM ml_predictions WHERE status = 'PENDING' ORDER BY candle_timestamp ASC"
        ).fetchall()

        if not pending:
            return {"resolved_count": 0, "excluded_count": 0, "still_pending_count": 0}

        # Build candle price lookup map: {utc_timestamp_iso: close_price}
        candle_map: dict[str, float] = {}
        candle_timestamps: list[pd.Timestamp] = []

        if candles is not None:
            if isinstance(candles, pd.DataFrame):
                for row in candles.itertuples(index=False):
                    ts = pd.Timestamp(row.timestamp).tz_convert("UTC")
                    candle_map[ts.isoformat()] = float(row.close)
                    candle_timestamps.append(ts)
            elif isinstance(candles, list):
                for item in candles:
                    ts = pd.Timestamp(item["timestamp"]).tz_convert("UTC")
                    candle_map[ts.isoformat()] = float(item["close"])
                    candle_timestamps.append(ts)

        if not candle_map:
            # Fallback: load available candles from market store
            from backend.app.services.market_store import get_candles
            try:
                db_candles = get_candles(limit=25000)
                for item in db_candles:
                    ts = pd.Timestamp(item["timestamp"]).tz_convert("UTC")
                    candle_map[ts.isoformat()] = float(item["close"])
                    candle_timestamps.append(ts)
            except Exception as exc:
                logger.warning("Could not fetch candles from market_store: %s", exc)

        if not candle_map:
            return {"resolved_count": 0, "excluded_count": 0, "still_pending_count": len(pending)}

        candle_timestamps.sort()
        latest_available_ts = candle_timestamps[-1]
        now_utc = pd.Timestamp(as_of or datetime.now(timezone.utc)).tz_convert("UTC")

        resolved_count = 0
        excluded_count = 0
        still_pending_count = 0

        for row in pending:
            pred_id = row["prediction_id"]
            t0 = pd.Timestamp(row["candle_timestamp"]).tz_convert("UTC")
            t_target = pd.Timestamp(row["target_candle_timestamp"]).tz_convert("UTC")
            ref_price = float(row["reference_price"])
            pred_dir = str(row["predicted_direction"]).upper()

            # Exchange time validation (America/New_York)
            t0_local = t0.astimezone(ET)
            t_target_local = t_target.astimezone(ET)

            # Condition 1: Same calendar trading date
            if t0_local.date() != t_target_local.date():
                db.execute(
                    """UPDATE ml_predictions
                       SET status = 'EXCLUDED', exclusion_reason = 'SESSION_BOUNDARY', resolved_at = ?
                       WHERE prediction_id = ?""",
                    (now_utc.isoformat(), pred_id),
                )
                excluded_count += 1
                continue

            # Condition 2: Regular trading session validation
            date_val = t0_local.date()
            if not is_trading_day(date_val):
                db.execute(
                    """UPDATE ml_predictions
                       SET status = 'EXCLUDED', exclusion_reason = 'NOT_TRADING_DAY', resolved_at = ?
                       WHERE prediction_id = ?""",
                    (now_utc.isoformat(), pred_id),
                )
                excluded_count += 1
                continue

            sess_close_time = session_close(date_val)
            rth_start_time = time(9, 30)

            # Candle must open >= 09:30 and complete by session close
            # Target candle opens at t_target_local and closes at t_target_local + 5m
            target_close_time = (t_target_local + pd.Timedelta(minutes=BAR_MINUTES)).time()

            if t0_local.time() < rth_start_time or target_close_time > sess_close_time:
                db.execute(
                    """UPDATE ml_predictions
                       SET status = 'EXCLUDED', exclusion_reason = 'SESSION_BOUNDARY', resolved_at = ?
                       WHERE prediction_id = ?""",
                    (now_utc.isoformat(), pred_id),
                )
                excluded_count += 1
                continue

            # Condition 3: Has the market data progressed past the target timestamp?
            # A 5-minute candle starting at t_target is completed at t_target + 5m
            target_completion_ts = t_target + pd.Timedelta(minutes=BAR_MINUTES)
            if latest_available_ts < t_target:
                # The target candle has not yet arrived
                still_pending_count += 1
                continue

            # Condition 4: Target candle presence in dataset
            t_target_iso = t_target.isoformat()
            if t_target_iso not in candle_map:
                # The dataset is already past the target candle, but this specific bar is missing
                db.execute(
                    """UPDATE ml_predictions
                       SET status = 'EXCLUDED', exclusion_reason = 'MISSING_FUTURE_CANDLE', resolved_at = ?
                       WHERE prediction_id = ?""",
                    (now_utc.isoformat(), pred_id),
                )
                excluded_count += 1
                continue

            target_close = candle_map[t_target_iso]
            if not np.isfinite(target_close) or target_close <= 0:
                db.execute(
                    """UPDATE ml_predictions
                       SET status = 'EXCLUDED', exclusion_reason = 'INVALID_PRICE', resolved_at = ?
                       WHERE prediction_id = ?""",
                    (now_utc.isoformat(), pred_id),
                )
                excluded_count += 1
                continue

            # Return calculation and directional determination
            forward_return = ((target_close - ref_price) / ref_price) * 100.0

            if target_close > ref_price:
                actual_dir = "UP"
            elif target_close < ref_price:
                actual_dir = "DOWN"
            else:
                # Exactly 0 bps return is neutral (NaN target in training labeling)
                db.execute(
                    """UPDATE ml_predictions
                       SET actual_future_price = ?, actual_direction = 'NEUTRAL',
                           forward_return_pct = 0.0, status = 'EXCLUDED',
                           exclusion_reason = 'NEUTRAL_RETURN', resolved_at = ?
                       WHERE prediction_id = ?""",
                    (target_close, now_utc.isoformat(), pred_id),
                )
                excluded_count += 1
                continue

            outcome_status = "CORRECT" if pred_dir == actual_dir else "INCORRECT"
            db.execute(
                """UPDATE ml_predictions
                   SET actual_future_price = ?, actual_direction = ?,
                       forward_return_pct = ?, status = ?,
                       resolved_at = ?
                   WHERE prediction_id = ?""",
                (target_close, actual_dir, forward_return, outcome_status, now_utc.isoformat(), pred_id),
            )
            resolved_count += 1

        return {
            "resolved_count": resolved_count,
            "excluded_count": excluded_count,
            "still_pending_count": still_pending_count,
        }


def calculate_ml_performance(
    window: str = "50",
    db_path: str | Path | None = None,
) -> dict:
    """Calculate comprehensive live out-of-sample ML classification and probability metrics.

    Windows: '50', '100', '500', 'all'.
    UP is the positive class.
    All scores are evaluated as raw, uncalibrated LightGBM model scores.
    """
    valid_windows = {"50", "100", "500", "all"}
    if window not in valid_windows:
        raise ValueError(f"Invalid evaluation window '{window}'. Allowed: {sorted(valid_windows)}")

    requested_size = int(window) if window != "all" else None

    with connect(db_path) as db:
        ensure_ml_predictions_table(db)

        # Overview counts across entire live store
        total_tracked = db.execute("SELECT count(*) FROM ml_predictions").fetchone()[0]
        total_pending = db.execute("SELECT count(*) FROM ml_predictions WHERE status = 'PENDING'").fetchone()[0]
        total_excluded = db.execute("SELECT count(*) FROM ml_predictions WHERE status = 'EXCLUDED'").fetchone()[0]
        total_resolved = db.execute("SELECT count(*) FROM ml_predictions WHERE status IN ('CORRECT', 'INCORRECT')").fetchone()[0]

        # Exclusion reason breakdown
        excl_rows = db.execute(
            "SELECT exclusion_reason, count(*) FROM ml_predictions WHERE status = 'EXCLUDED' GROUP BY exclusion_reason"
        ).fetchall()
        exclusion_breakdown = {str(r[0] or "UNSPECIFIED"): int(r[1]) for r in excl_rows}

        # Query resolved samples for the requested window
        if window == "all":
            query = """
                SELECT * FROM ml_predictions
                WHERE status IN ('CORRECT', 'INCORRECT')
                ORDER BY candle_timestamp ASC
            """
            rows = db.execute(query).fetchall()
        else:
            query = """
                SELECT * FROM (
                    SELECT * FROM ml_predictions
                    WHERE status IN ('CORRECT', 'INCORRECT')
                    ORDER BY candle_timestamp DESC
                    LIMIT ?
                ) ORDER BY candle_timestamp ASC
            """
            rows = db.execute(query, (requested_size,)).fetchall()

        sample_count = len(rows)
        is_partial = requested_size is not None and sample_count < requested_size and sample_count > 0

        # Model identifier metadata
        sha256 = _artifact_sha256(MODEL_PATH)
        model_identifier = {
            "model_name": "LGBMClassifier",
            "model_version": MODEL_VERSION,
            "artifact_sha256": sha256,
            "horizon_minutes": PREDICTION_HORIZON_MINUTES,
            "input_feature_count": 40,
        }

        # Empty state handling
        if sample_count == 0:
            return {
                "has_data": False,
                "empty_reason": "Insufficient live evaluation data. Metrics will appear after enough predictions have matured.",
                "window": window,
                "requested_window_size": requested_size,
                "sample_count": 0,
                "is_partial": False,
                "total_predictions_tracked": total_tracked,
                "pending_count": total_pending,
                "excluded_count": total_excluded,
                "total_resolved_count": total_resolved,
                "exclusion_breakdown": exclusion_breakdown,
                "start_date": None,
                "end_date": None,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "model_identifier": model_identifier,
                "summary": None,
                "metrics": None,
                "confusion_matrix": None,
                "probability_evaluation": None,
                "detailed_table": [],
            }

        start_date = rows[0]["candle_timestamp"]
        end_date = rows[-1]["candle_timestamp"]

        # ── Confusion Matrix Elements ─────────────────────────────────────────
        # Positive class: UP
        # Negative class: DOWN
        tp = sum(1 for r in rows if r["actual_direction"] == "UP" and r["predicted_direction"] == "UP")
        fn = sum(1 for r in rows if r["actual_direction"] == "UP" and r["predicted_direction"] == "DOWN")
        fp = sum(1 for r in rows if r["actual_direction"] == "DOWN" and r["predicted_direction"] == "UP")
        tn = sum(1 for r in rows if r["actual_direction"] == "DOWN" and r["predicted_direction"] == "DOWN")

        actual_up = tp + fn
        actual_down = tn + fp
        pred_up = tp + fp
        pred_down = tn + fn
        correct_count = tp + tn
        incorrect_count = fp + fn

        accuracy = correct_count / sample_count if sample_count > 0 else 0.0

        # UP class metrics
        precision_up = tp / pred_up if pred_up > 0 else 0.0
        recall_up = tp / actual_up if actual_up > 0 else 0.0
        f1_up = (2.0 * precision_up * recall_up) / (precision_up + recall_up) if (precision_up + recall_up) > 0 else 0.0

        # DOWN class metrics
        precision_down = tn / pred_down if pred_down > 0 else 0.0
        recall_down = tn / actual_down if actual_down > 0 else 0.0
        f1_down = (2.0 * precision_down * recall_down) / (precision_down + recall_down) if (precision_down + recall_down) > 0 else 0.0

        # Aggregated metrics
        macro_f1 = (f1_up + f1_down) / 2.0
        macro_precision = (precision_up + precision_down) / 2.0
        weighted_f1 = ((actual_up * f1_up) + (actual_down * f1_down)) / sample_count if sample_count > 0 else 0.0

        # Balanced Accuracy = average recall across classes
        if actual_up > 0 and actual_down > 0:
            balanced_accuracy = (recall_up + recall_down) / 2.0
        elif actual_up > 0:
            balanced_accuracy = recall_up
        elif actual_down > 0:
            balanced_accuracy = recall_down
        else:
            balanced_accuracy = 0.0

        # ── Probability Evaluation on Raw Scores ──────────────────────────────
        # Binary Cross-Entropy (Log Loss) and Brier Score
        eps = 1e-15
        log_loss_sum = 0.0
        brier_sum = 0.0

        probs: list[float] = []
        labels: list[int] = []

        for r in rows:
            y = 1 if r["actual_direction"] == "UP" else 0
            p_raw = float(r["probability_up"])
            probs.append(p_raw)
            labels.append(y)

            p_clipped = max(eps, min(1.0 - eps, p_raw))
            log_loss_sum += -(y * math.log(p_clipped) + (1 - y) * math.log(1.0 - p_clipped))
            brier_sum += (p_raw - y) ** 2

        log_loss = log_loss_sum / sample_count
        brier_score = brier_sum / sample_count

        # Calibration Diagnostics (5 equal bins: 0.0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0)
        bin_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        calibration_bins = []
        ece_sum = 0.0

        for b_idx in range(len(bin_edges) - 1):
            b_lower = bin_edges[b_idx]
            b_upper = bin_edges[b_idx + 1]
            bin_indices = [
                i for i, p in enumerate(probs)
                if (b_lower <= p <= b_upper if b_idx == len(bin_edges) - 2 else b_lower <= p < b_upper)
            ]
            bin_count = len(bin_indices)
            if bin_count > 0:
                mean_p = sum(probs[i] for i in bin_indices) / bin_count
                observed_rate = sum(labels[i] for i in bin_indices) / bin_count
                diff = abs(mean_p - observed_rate)
                ece_sum += (bin_count / sample_count) * diff
            else:
                mean_p = None
                observed_rate = None

            calibration_bins.append({
                "bin_label": f"{b_lower:.1f}–{b_upper:.1f}",
                "lower": b_lower,
                "upper": b_upper,
                "sample_count": bin_count,
                "mean_predicted_prob": round(mean_p, 4) if mean_p is not None else None,
                "observed_positive_rate": round(observed_rate, 4) if observed_rate is not None else None,
            })

        # Format Detailed Metrics Table
        detailed_table = [
            {
                "metric": "Accuracy",
                "up": "—",
                "down": "—",
                "overall": f"{accuracy * 100:.1f}%",
                "tooltip": "Overall directional hit rate across resolved predictions",
            },
            {
                "metric": "Precision",
                "up": f"{precision_up * 100:.1f}%",
                "down": f"{precision_down * 100:.1f}%",
                "overall": f"{macro_precision * 100:.1f}%",
                "tooltip": "Positive predictive value for each direction",
            },
            {
                "metric": "Recall",
                "up": f"{recall_up * 100:.1f}%",
                "down": f"{recall_down * 100:.1f}%",
                "overall": f"{balanced_accuracy * 100:.1f}%",
                "tooltip": "Sensitivity: share of actual moves correctly captured",
            },
            {
                "metric": "F1 Score",
                "up": f"{f1_up:.3f}",
                "down": f"{f1_down:.3f}",
                "overall": f"{macro_f1:.3f}",
                "tooltip": "Harmonic mean of precision and recall per class",
            },
            {
                "metric": "Macro F1",
                "up": "—",
                "down": "—",
                "overall": f"{macro_f1:.3f}",
                "tooltip": "Unweighted arithmetic average of UP and DOWN F1 scores",
            },
            {
                "metric": "Weighted F1",
                "up": "—",
                "down": "—",
                "overall": f"{weighted_f1:.3f}",
                "tooltip": "Class-prevalence weighted average F1 score",
            },
            {
                "metric": "Balanced Accuracy",
                "up": "—",
                "down": "—",
                "overall": f"{balanced_accuracy * 100:.1f}%",
                "tooltip": "Macro-average recall across directional classes",
            },
            {
                "metric": "Log Loss (Raw Score)",
                "up": "—",
                "down": "—",
                "overall": f"{log_loss:.4f}",
                "tooltip": "Cross-entropy loss evaluated on raw uncalibrated probabilities",
            },
            {
                "metric": "Brier Score (Raw Score)",
                "up": "—",
                "down": "—",
                "overall": f"{brier_score:.4f}",
                "tooltip": "Mean squared probability error relative to actual binary outcomes",
            },
        ]

        return {
            "has_data": True,
            "window": window,
            "requested_window_size": requested_size,
            "sample_count": sample_count,
            "is_partial": is_partial,
            "total_predictions_tracked": total_tracked,
            "pending_count": total_pending,
            "excluded_count": total_excluded,
            "total_resolved_count": total_resolved,
            "exclusion_breakdown": exclusion_breakdown,
            "start_date": start_date,
            "end_date": end_date,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "model_identifier": model_identifier,
            "summary": {
                "accuracy": round(accuracy, 4),
                "macro_f1": round(macro_f1, 4),
                "weighted_f1": round(weighted_f1, 4),
                "balanced_accuracy": round(balanced_accuracy, 4),
                "precision_overall": round(macro_precision, 4),
                "recall_overall": round(balanced_accuracy, 4),
                "total_resolved": sample_count,
                "correct": correct_count,
                "incorrect": incorrect_count,
            },
            "metrics": {
                "accuracy": round(accuracy, 4),
                "up": {
                    "precision": round(precision_up, 4),
                    "recall": round(recall_up, 4),
                    "f1": round(f1_up, 4),
                    "actual_count": actual_up,
                    "predicted_count": pred_up,
                },
                "down": {
                    "precision": round(precision_down, 4),
                    "recall": round(recall_down, 4),
                    "f1": round(f1_down, 4),
                    "actual_count": actual_down,
                    "predicted_count": pred_down,
                },
                "macro_f1": round(macro_f1, 4),
                "weighted_f1": round(weighted_f1, 4),
                "balanced_accuracy": round(balanced_accuracy, 4),
            },
            "confusion_matrix": {
                "true_up_pred_up": tp,
                "true_up_pred_down": fn,
                "true_down_pred_up": fp,
                "true_down_pred_down": tn,
                "total_actual_up": actual_up,
                "total_actual_down": actual_down,
                "total_pred_up": pred_up,
                "total_pred_down": pred_down,
                "total_resolved": sample_count,
            },
            "probability_evaluation": {
                "log_loss": round(log_loss, 4),
                "brier_score": round(brier_score, 4),
                "expected_calibration_error": round(ece_sum, 4),
                "is_uncalibrated": True,
                "raw_score_notice": (
                    "Evaluated strictly on raw LightGBM probability scores. "
                    "Scores are uncalibrated model scores and must not be interpreted as calibrated confidence."
                ),
                "calibration_bins": calibration_bins,
            },
            "detailed_table": detailed_table,
        }


def sync_existing_signals_to_ml_predictions(db_path: str | Path | None = None) -> int:
    """Import genuine live ML predictions from existing analysis_signals table.

    Runs once or periodically to track any signals previously produced by the terminal.
    """
    imported = 0
    with connect(db_path) as db:
        ensure_ml_predictions_table(db)
        rows = db.execute("SELECT candle_timestamp, payload FROM analysis_signals ORDER BY candle_timestamp").fetchall()
        for r in rows:
            try:
                payload = json.loads(r["payload"])
                ml = payload.get("ml_evidence")
                if not ml or not ml.get("available"):
                    continue
                candle_ts = ml.get("candle_timestamp") or payload.get("candle_timestamp")
                close_price = payload.get("indicators", {}).get("close")
                if not candle_ts or close_price is None:
                    continue
                saved = save_ml_prediction(ml, float(close_price), db_path=db_path)
                if saved:
                    imported += 1
            except Exception as exc:
                logger.warning("Error syncing signal to ml_predictions: %s", exc)

    # Immediately attempt resolution against available candles
    resolve_ml_predictions(db_path=db_path)
    return imported
