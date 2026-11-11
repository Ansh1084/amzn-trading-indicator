from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.experiments.evaluation import classification_metrics, trading_metrics
from src.experiments.features import ADVANCED_FEATURES, generate_experimental_features
from src.experiments.pipeline import DEFAULT_DATA, build_folds, load_dataset, prepare_examples, sample_weights
from src.experiments.retraining import (drift_triggered, mature_training_rows,
                                       population_stability_index, promotion_review, retraining_due)


DATA = DEFAULT_DATA


def synthetic_candles(days=4):
    stamps = [stamp for day in pd.date_range("2026-01-05", periods=days, freq="B")
              for stamp in pd.date_range(f"{day:%Y-%m-%d} 09:30", periods=78, freq="5min", tz="America/New_York")]
    timestamps = pd.DatetimeIndex(stamps).tz_convert("UTC")
    close = 220 + np.arange(len(timestamps)) * 0.015 + np.sin(np.arange(len(timestamps)) / 5) * 0.3
    open_ = close - np.cos(np.arange(len(timestamps)) / 4) * 0.08
    return pd.DataFrame({"timestamp": timestamps, "open": open_, "high": np.maximum(open_, close) + .12,
                         "low": np.minimum(open_, close) - .12, "close": close,
                         "volume": 1000 + np.arange(len(timestamps)) % 300, "vwap": (open_ + close) / 2})


def test_data_snapshot_is_regular_session_and_calendar_clean():
    if not DATA.exists():
        pytest.skip(f"Optional local research snapshot is unavailable: {DATA}")
    candles, audit = load_dataset()
    assert audit["rows_outside_calendar_rth_removed"] == 36
    assert audit["missing_expected_rth_candles"] == []
    assert audit["unexpected_rth_candles"] == []
    assert audit["session_candle_count_distribution"] == {"42": 3, "78": 273}
    assert candles.timestamp.is_monotonic_increasing


def test_advanced_features_do_not_depend_on_future_rows():
    candles = synthetic_candles()
    full = generate_experimental_features(candles)
    prefix = generate_experimental_features(candles.iloc[:100].copy())
    pd.testing.assert_frame_equal(full.loc[:99, ADVANCED_FEATURES].reset_index(drop=True),
                                  prefix[ADVANCED_FEATURES].reset_index(drop=True),
                                  check_exact=False, rtol=1e-12, atol=1e-12)


def test_folds_are_chronological_and_train_samples_are_purged():
    if not DATA.exists():
        pytest.skip(f"Optional local research snapshot is unavailable: {DATA}")
    candles, _ = load_dataset()
    examples = prepare_examples(candles)["examples"]
    folds = build_folds(examples)
    assert len(folds) == 5
    for fold in folds:
        assert examples.timestamp.iloc[fold.train_indices].max() < examples.timestamp.iloc[fold.validation_indices].min()
        assert examples.trading_date.iloc[fold.train_indices[-1]] < examples.trading_date.iloc[fold.validation_indices[0]]


def test_decay_weights_halve_across_requested_trading_day_half_life():
    examples = pd.DataFrame({"trading_date": [f"2026-01-{day:02}" for day in range(1, 22)]})
    weights = sample_weights(examples, np.arange(21), "decay_10")
    assert weights[0] / weights[-1] == pytest.approx(0.25)
    assert weights.mean() == pytest.approx(1.0)


def test_metrics_include_per_class_confusion_and_train_majority_baseline():
    metrics = classification_metrics([0, 1, 0, 1], [0, 0, 1, 1], [.2, .4, .6, .8], [0, 0, 1])
    assert metrics["confusion_matrix_labels_down_up"] == [[1, 1], [1, 1]]
    assert metrics["majority_class_in_train"] == "DOWN"
    assert metrics["roc_auc"] == pytest.approx(0.75)


def test_trading_metrics_uses_nonoverlapping_10_minute_entries():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-05 14:30", periods=6, freq="5min", tz="UTC"),
        "forward_return_10m": [1.0] * 6, "target_10m": [1] * 6,
    })
    result = trading_metrics(frame, [1, 1, 1, 1, 1, 1])
    assert result["n_nonoverlap"] == 3
    assert result["correct_count"] == 3


def test_label_maturity_waits_until_exit_candle_is_complete():
    stamps = pd.date_range("2026-01-05 14:30", periods=4, freq="5min", tz="UTC")
    now = datetime(2026, 1, 5, 14, 50, tzinfo=timezone.utc)
    assert mature_training_rows(stamps, [True] * 4, now).tolist() == [0, 1]
    with pytest.raises(ValueError):
        mature_training_rows(stamps, [True] * 4, datetime(2026, 1, 5, 14, 50))


def test_retraining_cadence_and_manual_promotion_guardrails():
    assert retraining_due(None, date(2026, 1, 1), "weekly")
    assert retraining_due(date(2026, 1, 1), date(2026, 1, 8), "weekly")
    assert not retraining_due(date(2026, 1, 1), date(2026, 1, 7), "weekly")
    result = promotion_review(
        {"n": 500, "f1_macro": .51, "balanced_accuracy": .51, "trade_max_drawdown": -.04, "fold_f1_std": .03},
        {"n": 500, "f1_macro": .53, "balanced_accuracy": .52, "trade_max_drawdown": -.035, "fold_f1_std": .04})
    assert result["eligible_for_human_review"] is True
    assert result["automatic_promotion"] is False


def test_drift_helper_uses_reference_binned_psi_and_threshold():
    ref = np.linspace(0, 1, 1000)
    assert population_stability_index(ref, ref) == pytest.approx(0.0, abs=1e-8)
    shifted = population_stability_index(ref, ref + 1.5)
    assert shifted > 0.2
    assert drift_triggered([.03, shifted])
    assert not drift_triggered([.01, .09])
