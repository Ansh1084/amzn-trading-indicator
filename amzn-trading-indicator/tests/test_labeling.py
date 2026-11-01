"""Tests for target generation in src.targets.labeling.

Covers:
  1. Correct UP label (forward_return > positive_threshold).
  2. Correct DOWN label (forward_return < -negative_threshold).
  3. Exactly unchanged price (forward_return == 0.0 -> NaN).
  4. Positive and negative neutral thresholds (symmetric and asymmetric bps).
  5. Exact 10-minute timestamp matching (matches t + 10m, not t + 5m).
  6. Missing future candle (target unavailable, does not fall back to shift(-2)).
  7. Missing intermediate candle (target matches if exact future exists).
  8. Final candles of a trading session (15:50 and 15:55 receive NaN).
  9. No cross-day target matching (Day 1 end never matches Day 2 start).
  10. No cross-session target matching (early-close session boundaries respected).
  11. Timezone-aware timestamps handling.
  12. UTC timestamps corresponding to New York trading sessions (EST and EDT).
  13. Invalid/missing columns error handling.
  14. Invalid close values (negative, zero, NaN, inf).
  15. Duplicate timestamps error handling.
  16. Unsorted timestamps error handling (no silent sorting).
  17. Input DataFrame immutability and index preservation.
  18. Correct behavior with different configured horizons (5m, 15m).
  19. No false target generation when future timestamp is unavailable.
  20. Threshold validation: rejection of negative, NaN, infinite, and boolean thresholds.
  21. Horizon validation: rejection of boolean, non-integer, zero, and negative values.
  22. Leakage column identification for configurable horizons (5m, 10m, 15m).
  23. Naive timestamp rejection by default and explicit assume_utc option.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import get_feature_columns
from src.targets.labeling import (
    DEFAULT_HORIZON_MINUTES,
    DEFAULT_NEUTRAL_THRESHOLD_BPS,
    LEAKAGE_TARGET_COLUMNS,
    generate_targets,
    get_leakage_target_columns,
    get_target_columns,
    is_target_column,
)


def _make_rth_df(
    closes: list[float],
    start: str = "2026-01-05 14:30:00",  # 09:30 ET (winter)
    freq: str = "5min",
) -> pd.DataFrame:
    """Helper to generate a clean synthetic OHLCV DataFrame in UTC."""
    n = len(closes)
    ts = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "symbol": ["AMZN"] * n,
            "timestamp": ts,
            "open": closes,
            "high": [c + 1.0 for c in closes],
            "low": [c - 1.0 for c in closes],
            "close": closes,
            "volume": [1000.0] * n,
            "vwap": closes,
        }
    )


class TestTargetClassification:
    def test_correct_up_label(self):
        """UP label: forward return strictly exceeds positive threshold."""
        # 09:30 close=100.0, 09:35 close=100.5, 09:40 close=102.0 (+2.0%)
        df = _make_rth_df([100.0, 100.5, 102.0])
        res = generate_targets(df, horizon_minutes=10, neutral_threshold_bps=0.0)

        assert res.loc[0, "future_close_10m"] == 102.0
        assert pytest.approx(res.loc[0, "forward_return_10m"], rel=1e-5) == 2.0
        assert res.loc[0, "target_10m"] == 1.0
        assert res.loc[0, "target_valid_10m"] is True or res.loc[0, "target_valid_10m"] == 1

    def test_correct_down_label(self):
        """DOWN label: forward return strictly below negative threshold."""
        # 09:30 close=100.0, 09:35 close=99.5, 09:40 close=98.0 (-2.0%)
        df = _make_rth_df([100.0, 99.5, 98.0])
        res = generate_targets(df, horizon_minutes=10, neutral_threshold_bps=0.0)

        assert res.loc[0, "future_close_10m"] == 98.0
        assert pytest.approx(res.loc[0, "forward_return_10m"], rel=1e-5) == -2.0
        assert res.loc[0, "target_10m"] == 0.0
        assert bool(res.loc[0, "target_valid_10m"]) is True

    def test_exactly_unchanged_price(self):
        """Unchanged price (forward return == 0.0) must receive target = NaN."""
        # 09:30 close=100.0, 09:35 close=101.0, 09:40 close=100.0 (0.0%)
        df = _make_rth_df([100.0, 101.0, 100.0])
        res = generate_targets(df, horizon_minutes=10, neutral_threshold_bps=0.0)

        assert res.loc[0, "future_close_10m"] == 100.0
        assert pytest.approx(res.loc[0, "forward_return_10m"], abs=1e-7) == 0.0
        assert np.isnan(res.loc[0, "target_10m"])
        assert bool(res.loc[0, "target_valid_10m"]) is False

    def test_positive_and_negative_neutral_thresholds(self):
        """Test symmetric neutral threshold in bps (1 bps = 0.01%)."""
        # neutral_threshold_bps = 10.0 bps = 0.10%
        # Row 0: 100.00 -> Row 2: 100.15 (+0.15% > +0.10%) -> UP (1.0)
        # Row 1: 100.00 -> Row 3: 100.05 (+0.05% in neutral zone) -> Neutral (NaN)
        # Row 2: 100.15 -> Row 4: 100.10 (-0.05% in neutral zone) -> Neutral (NaN)
        # Row 3: 100.05 -> Row 5: 99.85  (-0.20% < -0.10%) -> DOWN (0.0)
        closes = [100.0, 100.0, 100.15, 100.05, 100.10, 99.85]
        df = _make_rth_df(closes)
        res = generate_targets(df, horizon_minutes=10, neutral_threshold_bps=10.0)

        assert res.loc[0, "target_10m"] == 1.0
        assert np.isnan(res.loc[1, "target_10m"])
        assert np.isnan(res.loc[2, "target_10m"])
        assert res.loc[3, "target_10m"] == 0.0

    def test_asymmetric_thresholds(self):
        """Test explicit positive and negative thresholds in bps."""
        # pos_thresh = 15 bps (+0.15%), neg_thresh = 5 bps (-0.05%)
        # 09:30 (100.0) -> 09:40 (100.10) (+0.10% <= +0.15%) -> Neutral (NaN)
        # 09:35 (100.0) -> 09:45 (99.90)  (-0.10% < -0.05%) -> DOWN (0.0)
        closes = [100.0, 100.0, 100.10, 99.90]
        df = _make_rth_df(closes)
        res = generate_targets(
            df,
            horizon_minutes=10,
            positive_threshold_bps=15.0,
            negative_threshold_bps=5.0,
        )

        assert np.isnan(res.loc[0, "target_10m"])
        assert res.loc[1, "target_10m"] == 0.0


class TestExactTimestampMatching:
    def test_exact_10_minute_timestamp_matching(self):
        """Verify candle matches exactly t + 10m, not t + 5m."""
        # 09:30 close=100, 09:35 close=105, 09:40 close=110
        df = _make_rth_df([100.0, 105.0, 110.0])
        res = generate_targets(df, horizon_minutes=10)

        # Candle 09:30 must match 09:40 (110.0), NOT 09:35 (105.0)
        assert res.loc[0, "future_close_10m"] == 110.0
        assert pytest.approx(res.loc[0, "forward_return_10m"]) == 10.0

    def test_missing_future_candle_does_not_fall_back_to_shift(self):
        """Missing future candle must result in NaN, not a fallback shift(-2)."""
        # Timestamps: 09:30, 09:35, 09:45 (09:40 missing!)
        ts = pd.to_datetime(
            [
                "2026-01-05 14:30:00+00:00",
                "2026-01-05 14:35:00+00:00",
                "2026-01-05 14:45:00+00:00",
            ]
        )
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "close": [100.0, 101.0, 105.0],
            }
        )
        res = generate_targets(df, horizon_minutes=10)

        # For 09:30, t+10m is 09:40, which is MISSING.
        # Blind shift(-2) would have grabbed 09:45 (105.0). Exact matching must yield NaN!
        assert np.isnan(res.loc[0, "future_close_10m"])
        assert np.isnan(res.loc[0, "forward_return_10m"])
        assert np.isnan(res.loc[0, "target_10m"])

    def test_missing_intermediate_candle_matches_exact_future(self):
        """An intermediate missing candle does not break target if exact future exists."""
        # Timestamps: 09:30, 09:40 (09:35 missing!)
        ts = pd.to_datetime(
            [
                "2026-01-05 14:30:00+00:00",
                "2026-01-05 14:40:00+00:00",
            ]
        )
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "close": [100.0, 108.0],
            }
        )
        res = generate_targets(df, horizon_minutes=10)

        # For 09:30, t+10m is 09:40, which exists!
        assert res.loc[0, "future_close_10m"] == 108.0
        assert pytest.approx(res.loc[0, "forward_return_10m"]) == 8.0
        assert res.loc[0, "target_10m"] == 1.0


class TestSessionAndDayBoundaries:
    def test_final_candles_of_trading_session(self):
        """Final 2 candles of an RTH session (15:50 and 15:55) must have NaN targets."""
        # 15:45 ET = 20:45 UTC, 15:50 ET = 20:50 UTC, 15:55 ET = 20:55 UTC
        ts = pd.to_datetime(
            [
                "2026-01-05 20:45:00+00:00",
                "2026-01-05 20:50:00+00:00",
                "2026-01-05 20:55:00+00:00",
            ]
        )
        df = pd.DataFrame({"timestamp": ts, "close": [100.0, 101.0, 102.0]})
        res = generate_targets(df, horizon_minutes=10)

        # 15:45 + 10m = 15:55 (exists in session) -> valid!
        assert res.loc[0, "future_close_10m"] == 102.0
        assert res.loc[0, "target_10m"] == 1.0

        # 15:50 + 10m = 16:00 (market close, post-market) -> NaN
        assert np.isnan(res.loc[1, "future_close_10m"])
        assert np.isnan(res.loc[1, "target_10m"])

        # 15:55 + 10m = 16:05 (post-market) -> NaN
        assert np.isnan(res.loc[2, "future_close_10m"])
        assert np.isnan(res.loc[2, "target_10m"])

    def test_no_cross_day_target_matching(self):
        """Last candle of Day 1 must never match first candle of Day 2."""
        # Day 1: 15:55 ET (20:55 UTC)
        # Day 2: 09:30 ET (14:30 UTC next day)
        ts = pd.to_datetime(
            [
                "2026-01-05 20:55:00+00:00",
                "2026-01-06 14:30:00+00:00",
            ]
        )
        df = pd.DataFrame({"timestamp": ts, "close": [100.0, 105.0]})
        res = generate_targets(df, horizon_minutes=10)

        assert np.isnan(res.loc[0, "future_close_10m"])
        assert np.isnan(res.loc[0, "target_10m"])

    def test_no_cross_session_target_matching_early_close(self):
        """On early-close days (13:00 close), 12:50 and 12:55 must have NaN targets."""
        # 2025-11-28 is Black Friday (scheduled 13:00 ET close)
        # 12:45 ET = 17:45 UTC, 12:50 ET = 17:50 UTC, 12:55 ET = 17:55 UTC
        ts = pd.to_datetime(
            [
                "2025-11-28 17:45:00+00:00",  # 12:45 ET
                "2025-11-28 17:50:00+00:00",  # 12:50 ET
                "2025-11-28 17:55:00+00:00",  # 12:55 ET
                "2025-11-28 18:00:00+00:00",  # 13:00 ET (post-close)
                "2025-11-28 18:05:00+00:00",  # 13:05 ET (post-close)
            ]
        )
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "close": [100.0, 101.0, 102.0, 103.0, 104.0],
            }
        )
        res = generate_targets(df, horizon_minutes=10)

        # 12:45 + 10m = 12:55 (< 13:00) -> valid!
        assert res.loc[0, "future_close_10m"] == 102.0
        assert res.loc[0, "target_10m"] == 1.0

        # 12:50 + 10m = 13:00 (session close boundary) -> NaN
        assert np.isnan(res.loc[1, "future_close_10m"])
        assert np.isnan(res.loc[1, "target_10m"])

        # 12:55 + 10m = 13:05 (post-close) -> NaN
        assert np.isnan(res.loc[2, "future_close_10m"])
        assert np.isnan(res.loc[2, "target_10m"])


class TestTimezonesAndDST:
    def test_timezone_aware_and_dst_handling(self):
        """Correct session matching in both winter (EST, UTC-5) and summer (EDT, UTC-4)."""
        # Winter (EST): 09:30 ET is 14:30 UTC
        df_winter = _make_rth_df([100.0, 101.0, 102.0], start="2026-01-05 14:30:00")
        res_winter = generate_targets(df_winter, horizon_minutes=10)
        assert res_winter.loc[0, "future_close_10m"] == 102.0

        # Summer (EDT): 09:30 ET is 13:30 UTC
        df_summer = _make_rth_df([200.0, 201.0, 204.0], start="2026-07-06 13:30:00")
        res_summer = generate_targets(df_summer, horizon_minutes=10)
        assert res_summer.loc[0, "future_close_10m"] == 204.0
        assert res_summer.loc[0, "target_10m"] == 1.0

    def test_naive_timestamps_rejected_by_default(self):
        """Timezone-naive timestamps must be rejected by default (naive_tz_policy='reject')."""
        # String without offset
        df_naive_str = pd.DataFrame(
            {
                "timestamp": ["2026-01-05 09:30:00", "2026-01-05 09:35:00", "2026-01-05 09:40:00"],
                "close": [100.0, 101.0, 102.0],
            }
        )
        with pytest.raises(ValueError, match="timezone-naive"):
            generate_targets(df_naive_str)

        # Datetime series without tz
        df_naive_dt = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-05 09:30:00", periods=3, freq="5min"),
                "close": [100.0, 101.0, 102.0],
            }
        )
        with pytest.raises(ValueError, match="timezone-naive"):
            generate_targets(df_naive_dt)

    def test_naive_timestamps_assume_utc_option(self):
        """Setting naive_tz_policy='assume_utc' explicitly parses naive timestamps as UTC."""
        df_naive = pd.DataFrame(
            {
                "timestamp": ["2026-01-05 14:30:00", "2026-01-05 14:35:00", "2026-01-05 14:40:00"],
                "close": [100.0, 101.0, 102.0],
            }
        )
        res = generate_targets(df_naive, naive_tz_policy="assume_utc")
        assert res.loc[0, "future_close_10m"] == 102.0
        assert res.loc[0, "target_10m"] == 1.0


class TestValidationAndErrorHandling:
    def test_missing_required_columns(self):
        """Missing required columns must raise ValueError."""
        df = pd.DataFrame({"close": [100.0, 101.0]})
        with pytest.raises(ValueError, match="missing required column"):
            generate_targets(df)

    def test_empty_dataframe(self):
        """Empty DataFrame must raise ValueError."""
        df = pd.DataFrame(columns=["timestamp", "close"])
        with pytest.raises(ValueError, match="empty"):
            generate_targets(df)

    def test_invalid_close_values(self):
        """NaN, negative, zero, or non-numeric close must raise ValueError."""
        # Non-positive close
        df_neg = _make_rth_df([100.0, -5.0, 102.0])
        with pytest.raises(ValueError, match="zero or negative"):
            generate_targets(df_neg)

        df_zero = _make_rth_df([100.0, 0.0, 102.0])
        with pytest.raises(ValueError, match="zero or negative"):
            generate_targets(df_zero)

        # NaN close
        df_nan = _make_rth_df([100.0, 101.0, 102.0])
        df_nan.loc[1, "close"] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            generate_targets(df_nan)

        # Inf close
        df_inf = _make_rth_df([100.0, np.inf, 102.0])
        with pytest.raises(ValueError, match="non-finite"):
            generate_targets(df_inf)

    def test_duplicate_timestamps(self):
        """Duplicate timestamps must be rejected with ValueError."""
        df = _make_rth_df([100.0, 101.0, 102.0])
        df.loc[1, "timestamp"] = df.loc[0, "timestamp"]
        with pytest.raises(ValueError, match="duplicate timestamp"):
            generate_targets(df)

    def test_unsorted_timestamps(self):
        """Unsorted timestamps must be rejected without silent sorting."""
        df = _make_rth_df([100.0, 101.0, 102.0])
        df_unsorted = df.iloc[::-1].reset_index(drop=True)
        with pytest.raises(ValueError, match="not chronologically increasing"):
            generate_targets(df_unsorted)

    def test_invalid_timestamps(self):
        """Unparseable string or NaT timestamps must raise ValueError."""
        df = _make_rth_df([100.0, 101.0, 102.0])
        df.loc[1, "timestamp"] = pd.NaT
        with pytest.raises(ValueError, match="invalid, unparseable, or null/NaT"):
            generate_targets(df)

    def test_threshold_validation_rejects_negative_nan_inf_and_bool(self):
        """Thresholds must be numeric, finite, non-negative, and reject boolean."""
        df = _make_rth_df([100.0, 101.0, 102.0])

        # Negative neutral threshold
        with pytest.raises(ValueError, match="must be non-negative"):
            generate_targets(df, neutral_threshold_bps=-1.0)

        # NaN neutral threshold
        with pytest.raises(ValueError, match="cannot be NaN"):
            generate_targets(df, neutral_threshold_bps=np.nan)

        # Inf neutral threshold
        with pytest.raises(ValueError, match="cannot be infinite"):
            generate_targets(df, neutral_threshold_bps=np.inf)

        # Boolean neutral threshold
        with pytest.raises(ValueError, match="must be numeric"):
            generate_targets(df, neutral_threshold_bps=True)

        # Negative positive_threshold_bps
        with pytest.raises(ValueError, match="must be non-negative"):
            generate_targets(df, positive_threshold_bps=-5.0)

        # NaN positive_threshold_bps
        with pytest.raises(ValueError, match="cannot be NaN"):
            generate_targets(df, positive_threshold_bps=np.nan)

        # Negative negative_threshold_bps
        with pytest.raises(ValueError, match="must be non-negative"):
            generate_targets(df, negative_threshold_bps=-2.0)

        # Inf negative_threshold_bps
        with pytest.raises(ValueError, match="cannot be infinite"):
            generate_targets(df, negative_threshold_bps=np.inf)

    def test_horizon_validation_rejects_bool_and_non_positive_int(self):
        """horizon_minutes must be a positive int and explicitly reject bool."""
        df = _make_rth_df([100.0, 101.0, 102.0])

        with pytest.raises(ValueError, match="not boolean"):
            generate_targets(df, horizon_minutes=True)

        with pytest.raises(ValueError, match="not boolean"):
            generate_targets(df, horizon_minutes=False)

        with pytest.raises(ValueError, match="positive integer"):
            generate_targets(df, horizon_minutes=0)

        with pytest.raises(ValueError, match="positive integer"):
            generate_targets(df, horizon_minutes=-10)

        with pytest.raises(ValueError, match="positive integer"):
            generate_targets(df, horizon_minutes=10.5)

        with pytest.raises(ValueError, match="positive integer"):
            generate_targets(df, horizon_minutes="10")


class TestSafetyConfigAndLeakage:
    def test_input_dataframe_unchanged(self):
        """Input DataFrame must not be modified in-place."""
        df = _make_rth_df([100.0, 101.0, 102.0])
        df_copy = df.copy()
        custom_index = [10, 20, 30]
        df.index = custom_index
        df_copy.index = custom_index

        res = generate_targets(df)
        pd.testing.assert_frame_equal(df, df_copy)
        assert list(res.index) == custom_index

    def test_different_configured_horizons(self):
        """Configuring horizon_minutes=5 or 15 produces correctly named columns."""
        # 09:30, 09:35, 09:40, 09:45
        df = _make_rth_df([100.0, 101.0, 102.0, 103.0])
        res_5m = generate_targets(df, horizon_minutes=5)
        assert "target_5m" in res_5m.columns
        assert res_5m.loc[0, "future_close_5m"] == 101.0

        res_15m = generate_targets(df, horizon_minutes=15)
        assert "target_15m" in res_15m.columns
        assert res_15m.loc[0, "future_close_15m"] == 103.0

    def test_no_false_target_when_future_unavailable(self):
        """A lone single candle cannot generate a target."""
        df = _make_rth_df([100.0])
        res = generate_targets(df, horizon_minutes=10)
        assert np.isnan(res.loc[0, "future_close_10m"])
        assert np.isnan(res.loc[0, "forward_return_10m"])
        assert np.isnan(res.loc[0, "target_10m"])
        assert bool(res.loc[0, "target_valid_10m"]) is False

    def test_leakage_column_identification_configurable_horizons(self):
        """Verify leakage column cataloging across 5m, 10m, and 15m horizons."""
        # 1. 5-minute horizon columns
        cols_5m = get_target_columns(5)
        assert cols_5m == ["future_close_5m", "forward_return_5m", "target_5m", "target_valid_5m"]

        # 2. 10-minute horizon columns
        cols_10m = get_target_columns(10)
        assert cols_10m == ["future_close_10m", "forward_return_10m", "target_10m", "target_valid_10m"]

        # 3. 15-minute horizon columns
        cols_15m = get_target_columns(15)
        assert cols_15m == ["future_close_15m", "forward_return_15m", "target_15m", "target_valid_15m"]

        # 4. Multi-horizon leakage list
        multi_leakage = get_leakage_target_columns([5, 10, 15])
        assert set(cols_5m + cols_10m + cols_15m) == set(multi_leakage)

        # 5. is_target_column predicate
        assert is_target_column("future_close_5m")
        assert is_target_column("forward_return_10m")
        assert is_target_column("target_15m")
        assert is_target_column("target_valid_20m")

        # 6. Crucial check: none of the 40 engineered model features are target columns
        feat_cols = get_feature_columns()
        for f in feat_cols:
            assert not is_target_column(f), f"Feature '{f}' mistakenly identified as a target column!"
            assert f not in multi_leakage, f"Feature '{f}' in target leakage set!"
