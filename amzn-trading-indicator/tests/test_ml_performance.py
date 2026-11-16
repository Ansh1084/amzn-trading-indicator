"""Unit tests for live out-of-sample ML model performance evaluation."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.app.services.analysis.ml_evaluation import (
    calculate_ml_performance,
    ensure_ml_predictions_table,
    resolve_ml_predictions,
    save_ml_prediction,
)
from src.forecasts.store import connect


def sample_ml_evidence(
    candle_timestamp: str = "2026-11-16T14:30:00+00:00",
    direction: str = "UP",
    prob_up: float = 0.62,
    model_version: str = "test-v1",
) -> dict:
    return {
        "available": True,
        "direction": direction,
        "raw_probability_up": prob_up,
        "raw_directional_score": prob_up if direction == "UP" else 1.0 - prob_up,
        "model_name": "LGBMClassifier",
        "model_version": model_version,
        "artifact_sha256": "sha-test",
        "candle_timestamp": candle_timestamp,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def make_test_candles(start_iso: str, closes: list[float]) -> pd.DataFrame:
    """Generate 5-minute candles starting at start_iso."""
    ts_start = pd.Timestamp(start_iso)
    records = []
    for i, c in enumerate(closes):
        ts = ts_start + pd.Timedelta(minutes=5 * i)
        records.append({
            "timestamp": ts.isoformat(),
            "open": c,
            "high": c + 0.5,
            "low": c - 0.5,
            "close": c,
            "volume": 1000.0,
        })
    return pd.DataFrame(records)


def test_duplicate_prediction_prevention(tmp_path: Path):
    db_file = tmp_path / "test.sqlite3"
    evidence = sample_ml_evidence("2026-11-16T14:30:00+00:00")

    id1 = save_ml_prediction(evidence, 100.0, db_path=db_file)
    id2 = save_ml_prediction(evidence, 100.0, db_path=db_file)

    assert id1 is not None
    assert id1 == id2

    with connect(db_file) as db:
        count = db.execute("SELECT count(*) FROM ml_predictions").fetchone()[0]
        assert count == 1


def test_correct_up_outcome_at_exact_10m_horizon(tmp_path: Path):
    db_file = tmp_path / "test.sqlite3"
    # 09:30 ET is 14:30 UTC in standard time
    t0 = "2026-11-16T14:30:00+00:00"
    evidence = sample_ml_evidence(t0, direction="UP", prob_up=0.70)
    save_ml_prediction(evidence, 100.0, db_path=db_file)

    # 3 candles: 14:30 (T0), 14:35 (T0+5m), 14:40 (T0+10m)
    # Target close is 102.5 (> 100.0) -> UP
    candles = make_test_candles(t0, [100.0, 101.0, 102.5])
    res = resolve_ml_predictions(candles=candles, db_path=db_file)

    assert res["resolved_count"] == 1
    assert res["still_pending_count"] == 0

    with connect(db_file) as db:
        row = dict(db.execute("SELECT * FROM ml_predictions").fetchone())
        assert row["status"] == "CORRECT"
        assert row["actual_direction"] == "UP"
        assert row["actual_future_price"] == 102.5
        assert pytest.approx(row["forward_return_pct"], 0.001) == 2.5
        assert row["resolved_at"] is not None


def test_correct_down_outcome(tmp_path: Path):
    db_file = tmp_path / "test.sqlite3"
    t0 = "2026-11-16T14:30:00+00:00"
    evidence = sample_ml_evidence(t0, direction="DOWN", prob_up=0.30)
    save_ml_prediction(evidence, 100.0, db_path=db_file)

    # Target close at 14:40 is 97.5 (< 100.0) -> DOWN
    candles = make_test_candles(t0, [100.0, 99.0, 97.5])
    res = resolve_ml_predictions(candles=candles, db_path=db_file)

    assert res["resolved_count"] == 1
    with connect(db_file) as db:
        row = dict(db.execute("SELECT * FROM ml_predictions").fetchone())
        assert row["status"] == "CORRECT"
        assert row["actual_direction"] == "DOWN"
        assert row["actual_future_price"] == 97.5


def test_incorrect_up_prediction(tmp_path: Path):
    db_file = tmp_path / "test.sqlite3"
    t0 = "2026-11-16T14:30:00+00:00"
    evidence = sample_ml_evidence(t0, direction="UP", prob_up=0.60)
    save_ml_prediction(evidence, 100.0, db_path=db_file)

    # Target close at 14:40 is 98.0 (< 100.0), but predicted UP -> INCORRECT
    candles = make_test_candles(t0, [100.0, 99.0, 98.0])
    res = resolve_ml_predictions(candles=candles, db_path=db_file)

    assert res["resolved_count"] == 1
    with connect(db_file) as db:
        row = dict(db.execute("SELECT * FROM ml_predictions").fetchone())
        assert row["status"] == "INCORRECT"
        assert row["actual_direction"] == "DOWN"


def test_incorrect_down_prediction(tmp_path: Path):
    db_file = tmp_path / "test.sqlite3"
    t0 = "2026-11-16T14:30:00+00:00"
    evidence = sample_ml_evidence(t0, direction="DOWN", prob_up=0.40)
    save_ml_prediction(evidence, 100.0, db_path=db_file)

    # Target close at 14:40 is 103.0 (> 100.0), but predicted DOWN -> INCORRECT
    candles = make_test_candles(t0, [100.0, 101.0, 103.0])
    res = resolve_ml_predictions(candles=candles, db_path=db_file)

    assert res["resolved_count"] == 1
    with connect(db_file) as db:
        row = dict(db.execute("SELECT * FROM ml_predictions").fetchone())
        assert row["status"] == "INCORRECT"
        assert row["actual_direction"] == "UP"


def test_future_candle_missing_remains_pending_until_passed(tmp_path: Path):
    db_file = tmp_path / "test.sqlite3"
    t0 = "2026-11-16T14:30:00+00:00"
    evidence = sample_ml_evidence(t0, direction="UP", prob_up=0.60)
    save_ml_prediction(evidence, 100.0, db_path=db_file)

    # Candles only up to 14:35 (target at 14:40 has not occurred yet)
    candles = make_test_candles(t0, [100.0, 101.0])
    res = resolve_ml_predictions(candles=candles, db_path=db_file)

    assert res["resolved_count"] == 0
    assert res["still_pending_count"] == 1
    with connect(db_file) as db:
        row = dict(db.execute("SELECT * FROM ml_predictions").fetchone())
        assert row["status"] == "PENDING"


def test_data_gap_causes_exclusion(tmp_path: Path):
    db_file = tmp_path / "test.sqlite3"
    t0 = "2026-11-16T14:30:00+00:00"
    evidence = sample_ml_evidence(t0, direction="UP", prob_up=0.60)
    save_ml_prediction(evidence, 100.0, db_path=db_file)

    # Dataset has moved past 14:40 to 14:50, but 14:40 itself is missing
    c1 = make_test_candles("2026-11-16T14:30:00+00:00", [100.0, 100.5])
    c2 = make_test_candles("2026-11-16T14:50:00+00:00", [102.0])
    candles = pd.concat([c1, c2]).reset_index(drop=True)

    res = resolve_ml_predictions(candles=candles, db_path=db_file)
    assert res["excluded_count"] == 1
    with connect(db_file) as db:
        row = dict(db.execute("SELECT * FROM ml_predictions").fetchone())
        assert row["status"] == "EXCLUDED"
        assert row["exclusion_reason"] == "MISSING_FUTURE_CANDLE"


def test_market_closure_and_session_boundary_exclusion(tmp_path: Path):
    db_file = tmp_path / "test.sqlite3"
    # 15:55 ET is 20:55 UTC. 10m target is 21:05 UTC (16:05 ET) which exceeds 16:00 ET close.
    t0 = "2026-11-16T20:55:00+00:00"
    evidence = sample_ml_evidence(t0, direction="DOWN", prob_up=0.45)
    save_ml_prediction(evidence, 100.0, db_path=db_file)

    candles = make_test_candles(t0, [100.0])
    res = resolve_ml_predictions(candles=candles, db_path=db_file)

    assert res["excluded_count"] == 1
    with connect(db_file) as db:
        row = dict(db.execute("SELECT * FROM ml_predictions").fetchone())
        assert row["status"] == "EXCLUDED"
        assert row["exclusion_reason"] == "SESSION_BOUNDARY"


def test_neutral_return_is_excluded(tmp_path: Path):
    db_file = tmp_path / "test.sqlite3"
    t0 = "2026-11-16T14:30:00+00:00"
    evidence = sample_ml_evidence(t0, direction="UP", prob_up=0.55)
    save_ml_prediction(evidence, 100.0, db_path=db_file)

    # Target close at 14:40 is exactly 100.0 (0 bps return)
    candles = make_test_candles(t0, [100.0, 100.2, 100.0])
    res = resolve_ml_predictions(candles=candles, db_path=db_file)

    assert res["excluded_count"] == 1
    with connect(db_file) as db:
        row = dict(db.execute("SELECT * FROM ml_predictions").fetchone())
        assert row["status"] == "EXCLUDED"
        assert row["exclusion_reason"] == "NEUTRAL_RETURN"


def test_zero_division_and_empty_state_handling(tmp_path: Path):
    db_file = tmp_path / "test.sqlite3"
    # No predictions in store
    res = calculate_ml_performance("50", db_path=db_file)
    assert res["has_data"] is False
    assert res["sample_count"] == 0
    assert "Insufficient live evaluation data" in res["empty_reason"]
    assert res["summary"] is None
    assert res["metrics"] is None


def test_exact_metrics_and_confusion_matrix(tmp_path: Path):
    db_file = tmp_path / "test.sqlite3"
    with connect(db_file) as db:
        ensure_ml_predictions_table(db)

        # Inject 10 resolved predictions:
        # TP=4 (actual UP, pred UP)
        # FN=1 (actual UP, pred DOWN)
        # FP=2 (actual DOWN, pred UP)
        # TN=3 (actual DOWN, pred DOWN)
        # Total UP = 5, Total DOWN = 5, Pred UP = 6, Pred DOWN = 4
        samples = [
            ("UP", "UP", 0.70, "CORRECT"),
            ("UP", "UP", 0.65, "CORRECT"),
            ("UP", "UP", 0.60, "CORRECT"),
            ("UP", "UP", 0.75, "CORRECT"),
            ("UP", "DOWN", 0.45, "INCORRECT"),
            ("DOWN", "UP", 0.55, "INCORRECT"),
            ("DOWN", "UP", 0.58, "INCORRECT"),
            ("DOWN", "DOWN", 0.35, "CORRECT"),
            ("DOWN", "DOWN", 0.40, "CORRECT"),
            ("DOWN", "DOWN", 0.30, "CORRECT"),
        ]

        t_base = pd.Timestamp("2026-11-16T14:30:00+00:00")
        for i, (act, pred, prob, status) in enumerate(samples):
            ts = (t_base + pd.Timedelta(minutes=5 * i)).isoformat()
            db.execute(
                """INSERT INTO ml_predictions (
                    prediction_id, symbol, model_name, model_version, prediction_timestamp,
                    candle_timestamp, reference_price, horizon_minutes, predicted_direction,
                    probability_up, probability_down, target_candle_timestamp, actual_direction,
                    status, resolved_at
                ) VALUES (?, 'AMZN', 'LGBMClassifier', 'test-v1', ?, ?, 100.0, 10, ?, ?, ?, ?, ?, ?, ?)""",
                (f"pred-{i}", ts, ts, pred, prob, 1.0 - prob, ts, act, status, ts),
            )

    res = calculate_ml_performance("50", db_path=db_file)
    assert res["has_data"] is True
    assert res["sample_count"] == 10

    cm = res["confusion_matrix"]
    assert cm["true_up_pred_up"] == 4
    assert cm["true_up_pred_down"] == 1
    assert cm["true_down_pred_up"] == 2
    assert cm["true_down_pred_down"] == 3

    # Accuracy: 7 / 10 = 0.70
    assert pytest.approx(res["summary"]["accuracy"]) == 0.70

    # UP Precision: 4 / (4 + 2) = 0.6667
    assert pytest.approx(res["metrics"]["up"]["precision"], 0.001) == 4 / 6
    # UP Recall: 4 / (4 + 1) = 0.80
    assert pytest.approx(res["metrics"]["up"]["recall"], 0.001) == 0.80
    # UP F1: 2 * (4/6 * 4/5) / (4/6 + 4/5) = 8/11 ~ 0.7273
    assert pytest.approx(res["metrics"]["up"]["f1"], 0.001) == 8 / 11

    # DOWN Precision: 3 / (3 + 1) = 0.75
    assert pytest.approx(res["metrics"]["down"]["precision"], 0.001) == 0.75
    # DOWN Recall: 3 / (3 + 2) = 0.60
    assert pytest.approx(res["metrics"]["down"]["recall"], 0.001) == 0.60
    # DOWN F1: 2 * (3/4 * 3/5) / (3/4 + 3/5) = 2/3 ~ 0.6667
    assert pytest.approx(res["metrics"]["down"]["f1"], 0.001) == 2 / 3

    # Macro F1: (8/11 + 2/3) / 2 ~ 0.6970
    assert pytest.approx(res["summary"]["macro_f1"], 0.001) == (8 / 11 + 2 / 3) / 2
    # Balanced Accuracy: (0.80 + 0.60) / 2 = 0.70
    assert pytest.approx(res["summary"]["balanced_accuracy"], 0.001) == 0.70


def test_log_loss_and_brier_score(tmp_path: Path):
    db_file = tmp_path / "test.sqlite3"
    with connect(db_file) as db:
        ensure_ml_predictions_table(db)

        # 2 predictions:
        # 1: Actual UP (y=1), Prob UP = 0.8 -> loss = -ln(0.8) ~ 0.2231, Brier = (0.8 - 1)^2 = 0.04
        # 2: Actual DOWN (y=0), Prob UP = 0.4 -> loss = -ln(0.6) ~ 0.5108, Brier = (0.4 - 0)^2 = 0.16
        # Expected Log Loss = (0.22314 + 0.51083) / 2 ~ 0.3670
        # Expected Brier = (0.04 + 0.16) / 2 = 0.10
        db.execute(
            """INSERT INTO ml_predictions (
                prediction_id, symbol, model_name, model_version, prediction_timestamp,
                candle_timestamp, reference_price, horizon_minutes, predicted_direction,
                probability_up, probability_down, target_candle_timestamp, actual_direction,
                status, resolved_at
            ) VALUES
            ('p1', 'AMZN', 'LGBM', 'v1', 't1', '2026-11-16T14:30:00+00:00', 100, 10, 'UP', 0.8, 0.2, 't2', 'UP', 'CORRECT', 't2'),
            ('p2', 'AMZN', 'LGBM', 'v1', 't1', '2026-11-16T14:35:00+00:00', 100, 10, 'DOWN', 0.4, 0.6, 't2', 'DOWN', 'CORRECT', 't2')
            """
        )

    res = calculate_ml_performance("50", db_path=db_file)
    prob = res["probability_evaluation"]

    expected_log_loss = -(math.log(0.8) + math.log(0.6)) / 2.0
    expected_brier = (0.04 + 0.16) / 2.0

    assert pytest.approx(prob["log_loss"], 0.001) == expected_log_loss
    assert pytest.approx(prob["brier_score"], 0.001) == expected_brier
    assert prob["is_uncalibrated"] is True


def test_window_slicing(tmp_path: Path):
    db_file = tmp_path / "test.sqlite3"
    with connect(db_file) as db:
        ensure_ml_predictions_table(db)
        t_base = pd.Timestamp("2026-11-16T14:30:00+00:00")
        for i in range(60):
            ts = (t_base + pd.Timedelta(minutes=5 * i)).isoformat()
            db.execute(
                """INSERT INTO ml_predictions (
                    prediction_id, symbol, model_name, model_version, prediction_timestamp,
                    candle_timestamp, reference_price, horizon_minutes, predicted_direction,
                    probability_up, probability_down, target_candle_timestamp, actual_direction,
                    status, resolved_at
                ) VALUES (?, 'AMZN', 'LGBM', 'v1', ?, ?, 100, 10, 'UP', 0.6, 0.4, ?, 'UP', 'CORRECT', ?)""",
                (f"p-{i}", ts, ts, ts, ts),
            )

    res_50 = calculate_ml_performance("50", db_path=db_file)
    assert res_50["sample_count"] == 50
    assert res_50["is_partial"] is False

    res_100 = calculate_ml_performance("100", db_path=db_file)
    assert res_100["sample_count"] == 60
    assert res_100["is_partial"] is True

    res_all = calculate_ml_performance("all", db_path=db_file)
    assert res_all["sample_count"] == 60
    assert res_all["is_partial"] is False
