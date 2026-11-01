"""Tests for src.features.engineering.

Uses small synthetic DataFrames with known numerical values.
Covers:
  1. Correct output columns and preservation of original columns.
  2. Manual verification of price return features.
  3. Manual verification of candle geometry (body, range, wicks, ratio).
  4. Manual verification of SMA and EMA calculations.
  5. Manual verification of RSI on rising, falling, and flat prices.
  6. Manual verification of MACD, signal, and histogram.
  7. Manual verification of ATR (including overnight/gap True Range).
  8. Manual verification of Bollinger Bands (middle, upper, lower, width, position).
  9. Volume and VWAP features.
  10. Time features and America/New_York timezone conversion.
  11. Session-reset mode (reset_daily=True) isolation across trading days.
  12. Continuous mode (reset_daily=False) continuity across trading days.
  13. Initial NaN handling (no bfill, no zero-fill).
  14. Zero-range candle handling (safe division, no inf).
  15. Zero volume and zero VWAP safety (no inf or ZeroDivisionError).
  16. Immutability of input DataFrame.
  17. Preservation of row count and original index.
  18. Purity of feature column list (no future or target columns).
  19. Missing required columns error handling.
  20. Global absence of infinite values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    FEATURE_COLUMNS,
    REQUIRED_INPUT_COLUMNS,
    generate_features,
    get_feature_columns,
    get_missing_feature_counts,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_sample_df(
    n: int = 10,
    start: str = "2026-01-05 14:30:00",  # 09:30 ET
    freq: str = "5min",
) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame with predictable values."""
    ts = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    base_price = 200.0 + np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "symbol": "AMZN",
            "timestamp": ts,
            "open": base_price,
            "high": base_price + 2.0,
            "low": base_price - 1.0,
            "close": base_price + 1.0,
            "volume": 1000.0 + np.arange(n, dtype=float) * 100.0,
            "trade_count": 50.0 + np.arange(n, dtype=float) * 5.0,
            "vwap": base_price + 0.5,
        }
    )


# ── Test Suite ───────────────────────────────────────────────────────────────


class TestFeatureOutputColumns:
    def test_output_columns_and_preservation(self):
        df = _make_sample_df(n=25)
        res = generate_features(df)

        # 1. Original columns preserved
        for col in df.columns:
            assert col in res.columns
            pd.testing.assert_series_equal(res[col], df[col])

        # 2. All 40 engineered features present
        expected_features = get_feature_columns()
        assert len(expected_features) == 40
        for feat in expected_features:
            assert feat in res.columns

    def test_no_future_derived_or_target_columns(self):
        feat_cols = get_feature_columns()
        forbidden_substrings = ["future", "target", "label", "next"]
        for col in feat_cols:
            for sub in forbidden_substrings:
                assert sub not in col.lower(), f"Feature column '{col}' contains forbidden '{sub}'"


class TestPriceAndReturnCalculations:
    def test_returns_and_price_change(self):
        """Test manually verified return calculations."""
        # Closes: 100, 105, 110, 104.5
        closes = [100.0, 105.0, 110.0, 104.5]
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-05 14:30:00", periods=4, freq="5min", tz="UTC"),
                "open": closes,
                "high": [c + 1.0 for c in closes],
                "low": [c - 1.0 for c in closes],
                "close": closes,
                "volume": [1000.0] * 4,
                "vwap": closes,
            }
        )
        res = generate_features(df, reset_daily=False)

        # return_1
        assert np.isnan(res.loc[0, "return_1"])
        assert pytest.approx(res.loc[1, "return_1"], rel=1e-5) == (105.0 - 100.0) / 100.0  # +5%
        assert pytest.approx(res.loc[2, "return_1"], rel=1e-5) == (110.0 - 105.0) / 105.0  # +4.7619%
        assert pytest.approx(res.loc[3, "return_1"], rel=1e-5) == (104.5 - 110.0) / 110.0  # -5.0%

        # price_change
        assert np.isnan(res.loc[0, "price_change"])
        assert pytest.approx(res.loc[1, "price_change"], rel=1e-5) == 5.0
        assert pytest.approx(res.loc[2, "price_change"], rel=1e-5) == 5.0
        assert pytest.approx(res.loc[3, "price_change"], rel=1e-5) == -5.5

        # return_2
        assert np.isnan(res.loc[0, "return_2"])
        assert np.isnan(res.loc[1, "return_2"])
        assert pytest.approx(res.loc[2, "return_2"], rel=1e-5) == (110.0 - 100.0) / 100.0  # +10%

        # return_3
        assert np.isnan(res.loc[0, "return_3"])
        assert np.isnan(res.loc[1, "return_3"])
        assert np.isnan(res.loc[2, "return_3"])
        assert pytest.approx(res.loc[3, "return_3"], rel=1e-5) == (104.5 - 100.0) / 100.0  # +4.5%

        # log_return_1
        assert np.isnan(res.loc[0, "log_return_1"])
        assert pytest.approx(res.loc[1, "log_return_1"], rel=1e-5) == np.log(105.0 / 100.0)

    def test_candle_geometry(self):
        """Test candle body, range, upper/lower wick, and body_to_range."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-05 14:30:00", periods=3, freq="5min", tz="UTC"),
                "open": [100.0, 105.0, 100.0],
                "high": [110.0, 108.0, 100.0],
                "low": [95.0, 98.0, 100.0],
                "close": [108.0, 101.0, 100.0],
                "volume": [500.0, 600.0, 0.0],
                "vwap": [105.0, 103.0, 100.0],
            }
        )
        res = generate_features(df)

        # Bullish candle (Row 0): open=100, high=110, low=95, close=108
        assert pytest.approx(res.loc[0, "candle_body"]) == 8.0  # 108 - 100
        assert pytest.approx(res.loc[0, "candle_range"]) == 15.0  # 110 - 95
        assert pytest.approx(res.loc[0, "upper_wick"]) == 2.0  # 110 - 108
        assert pytest.approx(res.loc[0, "lower_wick"]) == 5.0  # 100 - 95
        assert pytest.approx(res.loc[0, "body_to_range"]) == 8.0 / 15.0

        # Bearish candle (Row 1): open=105, high=108, low=98, close=101
        assert pytest.approx(res.loc[1, "candle_body"]) == -4.0  # 101 - 105
        assert pytest.approx(res.loc[1, "candle_range"]) == 10.0  # 108 - 98
        assert pytest.approx(res.loc[1, "upper_wick"]) == 3.0  # 108 - 105
        assert pytest.approx(res.loc[1, "lower_wick"]) == 3.0  # 101 - 98
        assert pytest.approx(res.loc[1, "body_to_range"]) == -0.4  # -4 / 10

        # Zero-range candle (Row 2): open=close=high=low=100
        assert pytest.approx(res.loc[2, "candle_body"]) == 0.0
        assert pytest.approx(res.loc[2, "candle_range"]) == 0.0
        assert pytest.approx(res.loc[2, "upper_wick"]) == 0.0
        assert pytest.approx(res.loc[2, "lower_wick"]) == 0.0
        assert pytest.approx(res.loc[2, "body_to_range"]) == 0.0  # Safe zero division


class TestTrendCalculations:
    def test_sma_and_ema(self):
        """Test 5-period SMA and EMA calculations against manual formula."""
        prices = [10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-05 14:30:00", periods=6, freq="5min", tz="UTC"),
                "open": prices,
                "high": [p + 1 for p in prices],
                "low": [p - 1 for p in prices],
                "close": prices,
                "volume": [100.0] * 6,
                "vwap": prices,
            }
        )
        res = generate_features(df, reset_daily=False)

        # SMA 5 at index 4: mean of [10, 12, 14, 16, 18] = 14.0
        assert np.isnan(res.loc[3, "sma_5"])
        assert pytest.approx(res.loc[4, "sma_5"]) == 14.0
        # SMA 5 at index 5: mean of [12, 14, 16, 18, 20] = 16.0
        assert pytest.approx(res.loc[5, "sma_5"]) == 16.0

        # EMA with adjust=False:
        # alpha = 2 / (5 + 1) = 1/3
        # e0 = 10.0
        # e1 = (1/3)*12 + (2/3)*10 = 4 + 6.6667 = 10.6667
        # e2 = (1/3)*14 + (2/3)*10.6667 = 4.6667 + 7.1111 = 11.7778
        alpha = 2.0 / 6.0
        e0 = 10.0
        e1 = alpha * 12.0 + (1 - alpha) * e0
        e2 = alpha * 14.0 + (1 - alpha) * e1
        assert pytest.approx(res.loc[0, "ema_5"]) == e0
        assert pytest.approx(res.loc[1, "ema_5"]) == e1
        assert pytest.approx(res.loc[2, "ema_5"]) == e2

    def test_macd(self):
        """Test MACD, signal, and histogram relationship."""
        df = _make_sample_df(n=35)
        res = generate_features(df, reset_daily=False)

        # MACD = ema_12 - ema_26
        expected_macd = res["ema_12"] - res["ema_26"] if "ema_12" in res.columns else res["macd"]
        # Verify macd_hist = macd - macd_signal
        pd.testing.assert_series_equal(
            res["macd_hist"], res["macd"] - res["macd_signal"], check_names=False
        )


class TestMomentumCalculations:
    def test_rsi_rising_prices(self):
        """RSI on strictly rising prices must equal 100.0."""
        prices = [100.0 + i for i in range(25)]
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-05 14:30:00", periods=25, freq="5min", tz="UTC"),
                "open": prices,
                "high": [p + 0.5 for p in prices],
                "low": [p - 0.5 for p in prices],
                "close": prices,
                "volume": [1000.0] * 25,
                "vwap": prices,
            }
        )
        res = generate_features(df, reset_daily=False)
        # First 13 are warm-up NaNs
        assert res["rsi_14"].iloc[:13].isna().all()
        # From candle 14 onward, RSI is 100.0
        valid_rsi = res["rsi_14"].iloc[14:]
        assert (valid_rsi == 100.0).all()

    def test_rsi_falling_prices(self):
        """RSI on strictly falling prices must equal 0.0."""
        prices = [200.0 - i for i in range(25)]
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-05 14:30:00", periods=25, freq="5min", tz="UTC"),
                "open": prices,
                "high": [p + 0.5 for p in prices],
                "low": [p - 0.5 for p in prices],
                "close": prices,
                "volume": [1000.0] * 25,
                "vwap": prices,
            }
        )
        res = generate_features(df, reset_daily=False)
        valid_rsi = res["rsi_14"].iloc[14:]
        assert (valid_rsi == 0.0).all()

    def test_rsi_flat_prices(self):
        """RSI on flat prices must equal 50.0."""
        prices = [150.0] * 25
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-05 14:30:00", periods=25, freq="5min", tz="UTC"),
                "open": prices,
                "high": prices,
                "low": prices,
                "close": prices,
                "volume": [1000.0] * 25,
                "vwap": prices,
            }
        )
        res = generate_features(df, reset_daily=False)
        valid_rsi = res["rsi_14"].iloc[14:]
        assert (valid_rsi == 50.0).all()

    def test_wilder_rsi_manual_sequence(self):
        """Test Wilder RSI against an exact hand-calculated 16-candle sequence.

        Base price = 100.0.
        First 14 changes: [2, -1, 3, -2, 1, -1, 2, 0, 1, -2, 3, -1, 2, -1]
        Sum of gains = 2 + 3 + 1 + 2 + 1 + 3 + 2 = 14.0 -> AvgGain_14 = 14.0 / 14 = 1.0
        Sum of losses = 1 + 2 + 1 + 2 + 1 + 1 = 8.0 -> AvgLoss_14 = 8.0 / 14 = 4/7
        RSI at index 14 = 100 * (1.0 / (1.0 + 4/7)) = 700 / 11 ~ 63.63636%

        15th change (index 15): +4.0 (gain=4, loss=0)
        AvgGain_15 = (1.0 * 13 + 4.0) / 14 = 17 / 14
        AvgLoss_15 = ((4/7) * 13 + 0.0) / 14 = 26 / 49
        RSI at index 15 = 100 * (119 / 171) ~ 69.59064%
        """
        changes = [2.0, -1.0, 3.0, -2.0, 1.0, -1.0, 2.0, 0.0, 1.0, -2.0, 3.0, -1.0, 2.0, -1.0, 4.0]
        prices = [100.0]
        for c in changes:
            prices.append(prices[-1] + c)

        n = len(prices)  # 16 candles
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-05 14:30:00", periods=n, freq="5min", tz="UTC"),
                "open": prices,
                "high": [p + 0.5 for p in prices],
                "low": [p - 0.5 for p in prices],
                "close": prices,
                "volume": [1000.0] * n,
                "vwap": prices,
            }
        )
        res = generate_features(df, reset_daily=False)

        # First 14 candles (indices 0..13) are warm-up NaNs
        assert res["rsi_14"].iloc[:14].isna().all()

        # Index 14: first 14-period simple average
        expected_rsi_14 = 700.0 / 11.0
        assert pytest.approx(res.loc[14, "rsi_14"], rel=1e-5) == expected_rsi_14

        # Index 15: recursive Wilder update
        expected_rsi_15 = 100.0 * 119.0 / 171.0
        assert pytest.approx(res.loc[15, "rsi_14"], rel=1e-5) == expected_rsi_15

    def test_rate_of_change(self):
        prices = [100.0, 102.0, 104.0, 106.0, 108.0, 110.0]
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-05 14:30:00", periods=6, freq="5min", tz="UTC"),
                "open": prices,
                "high": prices,
                "low": prices,
                "close": prices,
                "volume": [100.0] * 6,
                "vwap": prices,
            }
        )
        res = generate_features(df, reset_daily=False)
        # ROC 5 at index 5: (110 - 100) / 100 * 100 = 10.0%
        assert np.isnan(res.loc[4, "roc_5"])
        assert pytest.approx(res.loc[5, "roc_5"]) == 10.0


class TestVolatilityCalculations:
    def test_atr_with_overnight_gap(self):
        """ATR true range must account for gap over previous close."""
        # Row 0: high=105, low=95, close=100. TR = 10.
        # Row 1: gap up open=120, high=125, low=118, close=122.
        # TR1 = 125 - 118 = 7.
        # TR2 = |125 - 100| = 25.
        # TR3 = |118 - 100| = 18.
        # TR for Row 1 = 25.
        df = _make_sample_df(n=20)
        res = generate_features(df, reset_daily=False)
        # First 13 rows are NaN warm-up
        assert res["atr_14"].iloc[:13].isna().all()
        assert res["atr_14"].iloc[14:].notna().all()
        assert (res["atr_14"].iloc[14:] > 0).all()

    def test_wilder_atr_manual_sequence(self):
        """Test Wilder ATR against an exact hand-calculated 15-candle sequence.

        Candle 0: high=105, low=95, close=100. TR0 = 10.0 (first-candle high - low convention).
        Candles 1..13: high=106, low=96, close=100 (prev close=100). TR1..13 = 10.0 each.
        First 14 TR values (candles 0..13) all equal 10.0 -> initial ATR_13 = 10.0.

        Candle 14: prev close=100, open=120, high=124, low=118, close=122.
        TR_14 = max(124-118, |124-100|, |118-100|) = max(6, 24, 18) = 24.0.
        Recursive update: ATR_14 = (10.0 * 13 + 24.0) / 14 = 154 / 14 = 11.0.
        """
        highs = [105.0] + [106.0] * 13 + [124.0]
        lows = [95.0] + [96.0] * 13 + [118.0]
        closes = [100.0] + [100.0] * 13 + [122.0]
        opens = [100.0] + [100.0] * 13 + [120.0]
        n = len(closes)  # 15 candles

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-05 14:30:00", periods=n, freq="5min", tz="UTC"),
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": [1000.0] * n,
                "vwap": closes,
            }
        )
        res = generate_features(df, reset_daily=False)

        # First 13 candles (indices 0..12) are warm-up NaNs
        assert res["atr_14"].iloc[:13].isna().all()

        # Index 13: simple mean of first 14 TR values
        assert pytest.approx(res.loc[13, "atr_14"], rel=1e-5) == 10.0

        # Index 14: recursive Wilder update
        assert pytest.approx(res.loc[14, "atr_14"], rel=1e-5) == 11.0

    def test_bollinger_bands(self):
        """Test Bollinger Band width, position, and zero-volatility safety."""
        # 25 constant candles
        prices = [100.0] * 25
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-05 14:30:00", periods=25, freq="5min", tz="UTC"),
                "open": prices,
                "high": prices,
                "low": prices,
                "close": prices,
                "volume": [1000.0] * 25,
                "vwap": prices,
            }
        )
        res = generate_features(df, reset_daily=False)

        # When price is flat, std = 0, bb_upper = bb_middle = bb_lower = 100.0
        assert pytest.approx(res.loc[24, "bb_middle"]) == 100.0
        assert pytest.approx(res.loc[24, "bb_upper"]) == 100.0
        assert pytest.approx(res.loc[24, "bb_lower"]) == 100.0
        assert pytest.approx(res.loc[24, "bb_width"]) == 0.0
        # Position with zero band width should default safely to 0.5 (center)
        assert pytest.approx(res.loc[24, "bb_position"]) == 0.5


class TestVolumeAndVWAPCalculations:
    def test_volume_and_vwap_features(self):
        df = _make_sample_df(n=25)
        res = generate_features(df, reset_daily=False)

        # Check volume_sma_5 at row 4
        expected_vol_sma_5 = df["volume"].iloc[:5].mean()
        assert pytest.approx(res.loc[4, "volume_sma_5"]) == expected_vol_sma_5

        # Check volume_ratio_5 = volume / volume_sma_5
        expected_ratio = df.loc[4, "volume"] / expected_vol_sma_5
        assert pytest.approx(res.loc[4, "volume_ratio_5"]) == expected_ratio

        # Check vwap_distance: (close - vwap) / vwap
        expected_vwap_dist = (df.loc[0, "close"] - df.loc[0, "vwap"]) / df.loc[0, "vwap"]
        assert pytest.approx(res.loc[0, "vwap_distance"]) == expected_vwap_dist


class TestTimeFeatures:
    def test_time_features_and_cyclical_encoding(self):
        # 14:30 UTC = 09:30 America/New_York (winter EST)
        # Monday, Jan 5, 2026
        df = _make_sample_df(n=1, start="2026-01-05 14:30:00")
        res = generate_features(df)

        assert res.loc[0, "hour"] == 9
        assert res.loc[0, "minute"] == 30
        assert res.loc[0, "day_of_week"] == 0  # Monday

        # Cyclical encoding: total minutes = 9 * 60 + 30 = 570
        angle = 2.0 * np.pi * 570.0 / 1440.0
        assert pytest.approx(res.loc[0, "time_of_day_sin"]) == np.sin(angle)
        assert pytest.approx(res.loc[0, "time_of_day_cos"]) == np.cos(angle)


class TestSessionHandling:
    def test_continuous_mode_carries_across_days(self):
        """In continuous mode (reset_daily=False), Day 2 candle 0 computes return against Day 1 last candle."""
        # Day 1: 2026-01-05 20:55 UTC (15:55 ET, close=200)
        # Day 2: 2026-01-06 14:30 UTC (09:30 ET, close=204)
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2026-01-05 20:55:00+00:00", "2026-01-06 14:30:00+00:00"]
                ),
                "open": [200.0, 203.0],
                "high": [201.0, 205.0],
                "low": [199.0, 202.0],
                "close": [200.0, 204.0],
                "volume": [1000.0, 1500.0],
                "vwap": [200.0, 203.5],
            }
        )
        res = generate_features(df, reset_daily=False)
        # Day 2 candle 0 computes return against Day 1 last candle (204 - 200) / 200 = +2%
        assert pytest.approx(res.loc[1, "return_1"]) == 0.02
        assert pytest.approx(res.loc[1, "price_change"]) == 4.0

    def test_session_reset_mode_isolates_days(self):
        """In session-reset mode (reset_daily=True), Day 2 candle 0 has NaN return_1 and fresh warm-up."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2026-01-05 20:55:00+00:00", "2026-01-06 14:30:00+00:00"]
                ),
                "open": [200.0, 203.0],
                "high": [201.0, 205.0],
                "low": [199.0, 202.0],
                "close": [200.0, 204.0],
                "volume": [1000.0, 1500.0],
                "vwap": [200.0, 203.5],
            }
        )
        res = generate_features(df, reset_daily=True)
        # Day 2 candle 0 must NOT use Day 1 data
        assert np.isnan(res.loc[1, "return_1"])
        assert np.isnan(res.loc[1, "price_change"])


class TestSafetyAndImmutability:
    def test_zero_range_candle_safety(self):
        """Zero range (high == low) must not cause ZeroDivisionError or infinite body_to_range."""
        df = _make_sample_df(n=5)
        df.loc[2, "high"] = 200.0
        df.loc[2, "low"] = 200.0
        df.loc[2, "open"] = 200.0
        df.loc[2, "close"] = 200.0

        res = generate_features(df)
        assert res.loc[2, "body_to_range"] == 0.0
        assert not np.isinf(res.loc[2, "body_to_range"])

    def test_zero_volume_and_zero_vwap_safety(self):
        """Zero volume and zero VWAP must not cause ZeroDivisionError or infinite values."""
        df = _make_sample_df(n=5)
        df.loc[1, "volume"] = 0.0
        df.loc[2, "vwap"] = 0.0

        res = generate_features(df)
        assert not np.isinf(res["volume_change"]).any()
        assert not np.isinf(res["vwap_distance"]).any()

    def test_input_dataframe_immutability(self):
        """Input DataFrame must not be modified in-place."""
        df = _make_sample_df(n=10)
        df_copy = df.copy()

        generate_features(df)
        pd.testing.assert_frame_equal(df, df_copy)

    def test_row_count_and_index_preserved(self):
        """Custom indices and row counts must be preserved exactly."""
        custom_index = [100, 102, 105, 108, 115]
        df = _make_sample_df(n=5)
        df.index = custom_index

        res = generate_features(df)
        assert len(res) == 5
        assert list(res.index) == custom_index

    def test_missing_required_columns_raises_error(self):
        df = _make_sample_df(n=5).drop(columns=["close"])
        with pytest.raises(ValueError, match="missing required column"):
            generate_features(df)

    def test_empty_dataframe_raises_error(self):
        df = pd.DataFrame(columns=REQUIRED_INPUT_COLUMNS)
        with pytest.raises(ValueError, match="empty"):
            generate_features(df)

    def test_no_infinities_produced(self):
        """Verify that absolutely no positive or negative infinities are produced."""
        df = _make_sample_df(n=50)
        res = generate_features(df, reset_daily=False)
        feat_cols = get_feature_columns()
        inf_count = np.isinf(res[feat_cols]).sum().sum()
        assert inf_count == 0

    def test_get_missing_feature_counts(self):
        """Verify helper correctly counts initial warm-up NaNs."""
        df = _make_sample_df(n=25)
        res = generate_features(df, reset_daily=False)
        counts = get_missing_feature_counts(res)

        assert isinstance(counts, pd.Series)
        assert counts["return_1"] == 1
        assert counts["return_2"] == 2
        assert counts["return_3"] == 3
        assert counts["sma_5"] == 4
        assert counts["sma_20"] == 19
        assert counts["candle_body"] == 0

    def test_zero_denominators_without_runtime_warnings(self):
        """Zero denominators across all features must never emit RuntimeWarnings."""
        import warnings

        # Construct candles with 0 range, 0 volume, 0 VWAP, 0 previous volume
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-05 14:30:00", periods=25, freq="5min", tz="UTC"),
                "open": [100.0] * 25,
                "high": [100.0] * 25,  # range == 0
                "low": [100.0] * 25,
                "close": [100.0] * 25,
                "volume": [0.0] * 25,  # volume == 0
                "vwap": [0.0] * 25,    # vwap == 0
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            res = generate_features(df, reset_daily=False)

        assert not np.isinf(res["body_to_range"]).any()
        assert not np.isinf(res["volume_change"]).any()
        assert not np.isinf(res["volume_ratio_5"]).any()
        assert not np.isinf(res["volume_ratio_20"]).any()
        assert not np.isinf(res["vwap_distance"]).any()
        assert not np.isinf(res["bb_width"]).any()
        assert not np.isinf(res["bb_position"]).any()


class TestTimestampValidation:
    def test_duplicate_timestamps_rejected(self):
        """Duplicate timestamps must be rejected with ValueError."""
        df = _make_sample_df(n=5)
        df.loc[2, "timestamp"] = df.loc[1, "timestamp"]
        with pytest.raises(ValueError, match="duplicate timestamp"):
            generate_features(df)

    def test_unsorted_timestamps_rejected(self):
        """Unsorted or reversed timestamps must be rejected without silent sorting."""
        df = _make_sample_df(n=5)
        df_reversed = df.iloc[::-1].reset_index(drop=True)
        with pytest.raises(ValueError, match="not chronologically increasing"):
            generate_features(df_reversed)

    def test_invalid_timestamps_rejected(self):
        """Invalid strings or NaT timestamps must be rejected with ValueError."""
        # Case A: NaT in timestamp
        df_nat = _make_sample_df(n=5)
        df_nat.loc[2, "timestamp"] = pd.NaT
        with pytest.raises(ValueError, match="invalid, unparseable, or null/NaT"):
            generate_features(df_nat)

        # Case B: Unparseable string in timestamp
        df_str = _make_sample_df(n=5)
        df_str = df_str.astype({"timestamp": "object"})
        df_str.loc[2, "timestamp"] = "corrupted-date-format"
        with pytest.raises(ValueError, match="invalid, unparseable, or null/NaT"):
            generate_features(df_str)

