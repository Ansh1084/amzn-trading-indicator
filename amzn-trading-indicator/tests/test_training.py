"""Tests for the model training pipeline.

Tests cover:
  - Feature column selection and leakage prevention
  - Session-aware chronological splitting
  - Purge/embargo at split boundaries
  - Missing value handling (causal, no future information)
  - Model training and evaluation end-to-end
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.splitting import get_trading_dates, split_by_sessions
from src.evaluation.metrics import build_comparison_table, compute_classification_metrics
from src.features.engineering import FEATURE_COLUMNS, get_feature_columns
from src.models.train import _get_safe_feature_columns, _handle_missing_values
from src.targets.labeling import get_leakage_target_columns, is_target_column


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_session_df(dates: list[str], candles_per_day: int = 78) -> pd.DataFrame:
    """Create a minimal DataFrame with realistic session structure for testing.

    Each date gets ``candles_per_day`` 5-minute candles starting at 09:30 ET.
    """
    rows = []
    for date_str in dates:
        base = pd.Timestamp(f"{date_str} 09:30", tz="America/New_York")
        for i in range(candles_per_day):
            ts = base + pd.Timedelta(minutes=5 * i)
            rows.append({
                "timestamp": ts.tz_convert("UTC"),
                "symbol": "AMZN",
                "open": 200.0 + np.random.randn() * 0.5,
                "high": 201.0 + np.random.randn() * 0.5,
                "low": 199.0 + np.random.randn() * 0.5,
                "close": 200.0 + np.random.randn() * 0.5,
                "volume": float(np.random.randint(1000, 50000)),
                "trade_count": float(np.random.randint(10, 500)),
                "vwap": 200.0 + np.random.randn() * 0.3,
            })
    df = pd.DataFrame(rows)
    # Ensure high >= max(open, close) and low <= min(open, close)
    df["high"] = df[["open", "high", "close"]].max(axis=1) + 0.01
    df["low"] = df[["open", "low", "close"]].min(axis=1) - 0.01
    return df


# ── Tests: Feature Column Selection ──────────────────────────────────────────


class TestFeatureColumnSelection:
    """Tests for safe feature column selection."""

    def test_feature_columns_are_defined(self):
        """FEATURE_COLUMNS list should be non-empty."""
        cols = get_feature_columns()
        assert len(cols) > 0
        assert isinstance(cols, list)

    def test_no_target_columns_in_features(self):
        """No target/leakage column should appear in FEATURE_COLUMNS."""
        leakage = set(get_leakage_target_columns())
        overlap = leakage & set(FEATURE_COLUMNS)
        assert len(overlap) == 0, f"Leakage columns in FEATURE_COLUMNS: {overlap}"

    def test_is_target_column_detects_targets(self):
        """is_target_column should flag all known target patterns."""
        assert is_target_column("future_close_10m")
        assert is_target_column("forward_return_10m")
        assert is_target_column("target_10m")
        assert is_target_column("target_valid_10m")

    def test_is_target_column_rejects_features(self):
        """is_target_column should not flag legitimate feature columns."""
        assert not is_target_column("rsi_14")
        assert not is_target_column("sma_5")
        assert not is_target_column("volume_ratio_20")

    def test_safe_feature_selection_excludes_targets(self):
        """_get_safe_feature_columns should exclude target columns."""
        # Create a DF with features AND target columns
        data = {col: np.random.randn(10) for col in FEATURE_COLUMNS}
        data["future_close_10m"] = np.random.randn(10)
        data["forward_return_10m"] = np.random.randn(10)
        data["target_10m"] = np.random.choice([0, 1], 10)
        data["target_valid_10m"] = np.ones(10, dtype=bool)
        data["timestamp"] = pd.date_range("2025-01-01", periods=10, freq="5min", tz="UTC")
        data["close"] = np.random.randn(10)
        data["symbol"] = "AMZN"

        df = pd.DataFrame(data)
        safe = _get_safe_feature_columns(df)

        assert "future_close_10m" not in safe
        assert "forward_return_10m" not in safe
        assert "target_10m" not in safe
        assert "target_valid_10m" not in safe
        assert "timestamp" not in safe
        assert "close" not in safe
        assert "symbol" not in safe

        # All returned columns should be in FEATURE_COLUMNS
        for col in safe:
            assert col in FEATURE_COLUMNS, f"{col} not in FEATURE_COLUMNS"


# ── Tests: Chronological Splitting ───────────────────────────────────────────


class TestChronologicalSplitting:
    """Tests for session-aware chronological splitting."""

    def test_basic_split_sizes(self):
        """Split should produce non-empty train, val, and test sets."""
        dates = [f"2025-11-{d:02d}" for d in range(3, 24)]  # 21 weekdays approx
        df = _make_session_df(dates, candles_per_day=10)
        splits = split_by_sessions(df, purge_candles=0)

        assert len(splits["train"]) > 0
        assert len(splits["val"]) > 0
        assert len(splits["test"]) > 0

    def test_chronological_order(self):
        """Train dates should precede val dates, which should precede test dates."""
        dates = [f"2025-11-{d:02d}" for d in range(3, 24)]
        df = _make_session_df(dates, candles_per_day=10)
        splits = split_by_sessions(df, purge_candles=0)

        train_max = pd.to_datetime(splits["train"]["timestamp"], utc=True).max()
        val_min = pd.to_datetime(splits["val"]["timestamp"], utc=True).min()
        val_max = pd.to_datetime(splits["val"]["timestamp"], utc=True).max()
        test_min = pd.to_datetime(splits["test"]["timestamp"], utc=True).min()

        assert train_max < val_min, "Train data must precede validation data."
        assert val_max < test_min, "Validation data must precede test data."

    def test_session_integrity(self):
        """All candles from a trading date should be in the same split."""
        dates = [f"2025-11-{d:02d}" for d in range(3, 14)]
        df = _make_session_df(dates, candles_per_day=10)
        splits = split_by_sessions(df, purge_candles=0)

        for split_name in ["train", "val", "test"]:
            split_df = splits[split_name]
            ts = pd.to_datetime(split_df["timestamp"], utc=True)
            local_dates = ts.dt.tz_convert("America/New_York").dt.date
            # Each date should have exactly candles_per_day candles
            date_counts = local_dates.value_counts()
            for date, count in date_counts.items():
                assert count == 10, (
                    f"Session {date} in {split_name} has {count} candles, expected 10."
                )

    def test_purge_removes_boundary_candles(self):
        """Purge should remove candles from the end of train and val."""
        dates = [f"2025-11-{d:02d}" for d in range(3, 24)]
        df = _make_session_df(dates, candles_per_day=10)

        splits_no_purge = split_by_sessions(df, purge_candles=0)
        splits_with_purge = split_by_sessions(df, purge_candles=2)

        assert len(splits_with_purge["train"]) == len(splits_no_purge["train"]) - 2
        assert len(splits_with_purge["val"]) == len(splits_no_purge["val"]) - 2
        # Test set is NOT purged
        assert len(splits_with_purge["test"]) == len(splits_no_purge["test"])

    def test_no_overlap_between_splits(self):
        """No timestamp should appear in more than one split."""
        dates = [f"2025-11-{d:02d}" for d in range(3, 24)]
        df = _make_session_df(dates, candles_per_day=10)
        splits = split_by_sessions(df, purge_candles=2)

        train_ts = set(pd.to_datetime(splits["train"]["timestamp"], utc=True))
        val_ts = set(pd.to_datetime(splits["val"]["timestamp"], utc=True))
        test_ts = set(pd.to_datetime(splits["test"]["timestamp"], utc=True))

        assert len(train_ts & val_ts) == 0, "Train and val overlap!"
        assert len(train_ts & test_ts) == 0, "Train and test overlap!"
        assert len(val_ts & test_ts) == 0, "Val and test overlap!"

    def test_metadata_is_populated(self):
        """Split metadata should contain session counts and date ranges."""
        dates = [f"2025-11-{d:02d}" for d in range(3, 14)]
        df = _make_session_df(dates, candles_per_day=10)
        splits = split_by_sessions(df, purge_candles=0)

        meta = splits["metadata"]
        assert meta["n_train_sessions"] > 0
        assert meta["n_val_sessions"] > 0
        assert meta["n_test_sessions"] > 0
        assert "train_date_range" in meta
        assert "val_date_range" in meta
        assert "test_date_range" in meta

    def test_empty_df_raises(self):
        """Splitting an empty DataFrame should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            split_by_sessions(pd.DataFrame({"timestamp": []}))

    def test_too_few_sessions_raises(self):
        """Splitting fewer than 3 sessions should raise ValueError."""
        df = _make_session_df(["2025-11-03", "2025-11-04"], candles_per_day=10)
        with pytest.raises(ValueError, match="at least 3"):
            split_by_sessions(df)


# ── Tests: Missing Value Handling ────────────────────────────────────────────


class TestMissingValueHandling:
    """Tests for causal missing value handling."""

    def test_no_nans_after_handling(self):
        """After handling, no NaN values should remain."""
        n = 100
        X_train = pd.DataFrame({
            "a": np.random.randn(n),
            "b": np.random.randn(n),
        })
        X_val = X_train.copy()
        X_test = X_train.copy()

        # Introduce NaNs
        X_train.iloc[:5, 0] = np.nan
        X_val.iloc[:3, 1] = np.nan

        X_train, X_val, X_test, info = _handle_missing_values(X_train, X_val, X_test)

        assert X_train.isna().sum().sum() == 0
        assert X_val.isna().sum().sum() == 0
        assert X_test.isna().sum().sum() == 0

    def test_inf_values_replaced(self):
        """Infinite values should be replaced."""
        n = 50
        X_train = pd.DataFrame({"a": np.random.randn(n), "b": np.random.randn(n)})
        X_val = X_train.copy()
        X_test = X_train.copy()

        X_train.iloc[0, 0] = np.inf
        X_train.iloc[1, 1] = -np.inf

        X_train, X_val, X_test, info = _handle_missing_values(X_train, X_val, X_test)

        assert not np.any(np.isinf(X_train.to_numpy()))


# ── Tests: Metrics Computation ───────────────────────────────────────────────


class TestMetrics:
    """Tests for classification metric computation."""

    def test_perfect_predictions(self):
        """Perfect predictions should yield accuracy=1, F1=1."""
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.1, 0.9, 0.9, 0.1])

        result = compute_classification_metrics(
            y_true, y_pred, y_prob, model_name="test_model"
        )

        assert result["accuracy"] == 1.0
        assert result["f1_score"] == 1.0
        assert result["roc_auc"] == 1.0

    def test_random_predictions(self):
        """Random predictions should have accuracy around 0.5."""
        np.random.seed(42)
        y_true = np.random.choice([0, 1], 1000)
        y_pred = np.random.choice([0, 1], 1000)

        result = compute_classification_metrics(y_true, y_pred)
        assert 0.3 < result["accuracy"] < 0.7

    def test_comparison_table(self):
        """build_comparison_table should produce a DataFrame with correct columns."""
        results = [
            compute_classification_metrics(
                np.array([0, 1, 0, 1]),
                np.array([0, 1, 1, 0]),
                model_name=f"model_{i}",
            )
            for i in range(3)
        ]
        table = build_comparison_table(results)
        assert len(table) == 3
        assert "model" in table.columns
        assert "accuracy" in table.columns
        assert "f1_score" in table.columns

