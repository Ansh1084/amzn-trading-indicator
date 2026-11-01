"""Tests for src.data.preprocessing.

Uses small synthetic DataFrames — does not depend on the full historical dataset.
Covers all edge cases: NaT, infinities, non-numeric values, conflicting duplicates,
identical duplicates, missing session first/last candles, partial dataset boundary gap detection,
full missing sessions, RTH boundaries, early-close sessions, mixed timezone handling with DST,
and explicit validation failure/cleanup.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import (
    CANDLE_INTERVAL,
    DEFAULT_US_EARLY_CLOSES,
    REQUIRED_COLUMNS,
    GapInfo,
    ValidationReport,
    detect_gaps,
    filter_rth,
    load_raw_csv,
    preprocess,
    remove_duplicates,
    sort_by_timestamp,
    validate_columns,
    validate_no_duplicates,
    validate_no_missing_values,
    validate_ohlcv,
    validate_timestamps,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_utc_ts(start: str, periods: int, freq: str = "5min") -> pd.Series:
    """Return a UTC-aware timestamp Series."""
    return pd.Series(pd.date_range(start, periods=periods, freq=freq, tz="UTC"))


def _make_valid_df(
    n: int = 5,
    start: str = "2026-01-05 14:30:00",  # 09:30 ET (EST is UTC-5)
) -> pd.DataFrame:
    """Build a small valid OHLCV DataFrame with UTC-aware timestamps."""
    rng = np.random.default_rng(42)
    ts = _make_utc_ts(start, n)
    close = 200.0 + rng.standard_normal(n).cumsum()
    return pd.DataFrame(
        {
            "symbol": "AMZN",
            "timestamp": ts,
            "open": close + rng.uniform(-0.5, 0.5, n),
            "high": close + rng.uniform(0.5, 1.5, n),
            "low": close - rng.uniform(0.5, 1.5, n),
            "close": close,
            "volume": rng.integers(1000, 50000, n).astype(float),
            "trade_count": rng.integers(10, 500, n).astype(float),
            "vwap": close + rng.uniform(-0.1, 0.1, n),
        }
    )


def _write_csv(df: pd.DataFrame, path: Path, write_tz: bool = True) -> Path:
    """Write DataFrame to CSV."""
    df = df.copy()
    if write_tz and hasattr(df["timestamp"].dtype, "tz") and df["timestamp"].dtype.tz is not None:
        df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
    else:
        df["timestamp"] = df["timestamp"].astype(str)
    df.to_csv(path, index=False)
    return path


# ── Column validation ────────────────────────────────────────────────────────


class TestValidateColumns:
    def test_all_present(self):
        df = _make_valid_df()
        report = validate_columns(df)
        assert report.is_valid

    def test_missing_column(self):
        df = _make_valid_df().drop(columns=["volume"])
        report = validate_columns(df)
        assert not report.is_valid
        assert any("volume" in e for e in report.errors)

    def test_extra_column_is_info_not_error(self):
        df = _make_valid_df()
        df["extra"] = 1
        report = validate_columns(df)
        assert report.is_valid
        assert "extra" in report.info.get("extra_columns", [])

    def test_empty_df_missing_all(self):
        df = pd.DataFrame()
        report = validate_columns(df)
        assert not report.is_valid


# ── Timestamp validation ─────────────────────────────────────────────────────


class TestValidateTimestamps:
    def test_valid_sorted(self):
        df = _make_valid_df()
        report = validate_timestamps(df)
        assert report.is_valid
        assert report.info["is_sorted"] is True
        assert report.info["n_nat"] == 0

    def test_unsorted_warns(self):
        df = _make_valid_df()
        df = df.iloc[::-1].reset_index(drop=True)
        report = validate_timestamps(df)
        assert report.is_valid
        assert report.info["is_sorted"] is False
        assert len(report.warnings) > 0

    def test_missing_column(self):
        df = _make_valid_df().drop(columns=["timestamp"])
        report = validate_timestamps(df)
        assert not report.is_valid
        assert any("timestamp" in e.lower() for e in report.errors)

    def test_naive_timestamps_fail(self):
        df = _make_valid_df()
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        report = validate_timestamps(df)
        assert not report.is_valid
        assert any("timezone" in e.lower() for e in report.errors)

    def test_nat_in_timestamp_fails(self):
        df = _make_valid_df(n=5)
        df.loc[2, "timestamp"] = pd.NaT
        report = validate_timestamps(df)
        assert not report.is_valid
        assert report.info["n_nat"] == 1
        assert any("NaT" in e for e in report.errors)

    def test_empty_df_fails(self):
        df = pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns, UTC]")})
        report = validate_timestamps(df)
        assert not report.is_valid
        assert any("empty" in e.lower() for e in report.errors)


# ── Duplicate detection ──────────────────────────────────────────────────────


class TestValidateNoDuplicates:
    def test_no_duplicates(self):
        df = _make_valid_df()
        report = validate_no_duplicates(df)
        assert report.is_valid
        assert report.info["n_duplicates"] == 0
        assert report.info["n_conflicting_duplicates"] == 0

    def test_identical_duplicates_reported(self):
        df = _make_valid_df()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        report = validate_no_duplicates(df)
        assert not report.is_valid
        assert report.info["n_duplicates"] == 2
        assert report.info["n_conflicting_duplicates"] == 0
        assert len(report.warnings) > 0

    def test_conflicting_duplicates_reported(self):
        df = _make_valid_df()
        conflicting_row = df.iloc[[0]].copy()
        conflicting_row["close"] = 999.0
        conflicting_row["volume"] = 1.0
        df = pd.concat([df, conflicting_row], ignore_index=True)

        report = validate_no_duplicates(df)
        assert not report.is_valid
        assert report.info["n_duplicates"] == 2
        assert report.info["n_conflicting_duplicates"] == 2
        assert any("conflicting" in e.lower() for e in report.errors)


# ── Missing values ───────────────────────────────────────────────────────────


class TestValidateNoMissingValues:
    def test_no_nulls(self):
        df = _make_valid_df()
        report = validate_no_missing_values(df)
        assert report.is_valid

    def test_with_nulls(self):
        df = _make_valid_df()
        df.loc[0, "close"] = np.nan
        report = validate_no_missing_values(df)
        assert not report.is_valid
        assert "close" in report.info["null_counts"]

    def test_empty_df_fails(self):
        df = pd.DataFrame()
        report = validate_no_missing_values(df)
        assert not report.is_valid


# ── OHLCV consistency ────────────────────────────────────────────────────────


class TestValidateOHLCV:
    def test_valid(self):
        df = _make_valid_df()
        report = validate_ohlcv(df)
        assert report.is_valid

    def test_positive_and_negative_infinity_fail(self):
        df = _make_valid_df(n=5)
        df.loc[0, "high"] = np.inf
        df.loc[1, "low"] = -np.inf
        report = validate_ohlcv(df)
        assert not report.is_valid
        assert report.info["high_inf"] == 1
        assert report.info["low_inf"] == 1
        assert any("infinite" in e.lower() for e in report.errors)

    def test_non_numeric_values_fail(self):
        df = _make_valid_df(n=5)
        df["close"] = df["close"].astype(object)
        df.loc[2, "close"] = "invalid_price_str"
        report = validate_ohlcv(df)
        assert not report.is_valid
        assert any("non-numeric" in e.lower() for e in report.errors)

    def test_high_lt_low(self):
        df = _make_valid_df()
        df.loc[0, "high"] = df.loc[0, "low"] - 1.0
        report = validate_ohlcv(df)
        assert not report.is_valid
        assert report.info["high_lt_low"] > 0

    def test_negative_volume(self):
        df = _make_valid_df()
        df.loc[0, "volume"] = -100.0
        report = validate_ohlcv(df)
        assert not report.is_valid
        assert report.info["negative_volume"] > 0

    def test_high_lt_open(self):
        df = _make_valid_df()
        df.loc[0, "high"] = df.loc[0, "open"] - 0.01
        report = validate_ohlcv(df)
        assert not report.is_valid
        assert report.info["high_lt_open"] > 0

    def test_high_lt_close(self):
        df = _make_valid_df()
        df.loc[0, "high"] = df.loc[0, "close"] - 0.01
        report = validate_ohlcv(df)
        assert not report.is_valid
        assert report.info["high_lt_close"] > 0

    def test_low_gt_open(self):
        df = _make_valid_df()
        df.loc[0, "low"] = df.loc[0, "open"] + 0.01
        report = validate_ohlcv(df)
        assert not report.is_valid
        assert report.info["low_gt_open"] > 0

    def test_low_gt_close(self):
        df = _make_valid_df()
        df.loc[0, "low"] = df.loc[0, "close"] + 0.01
        report = validate_ohlcv(df)
        assert not report.is_valid
        assert report.info["low_gt_close"] > 0

    def test_missing_column(self):
        df = _make_valid_df().drop(columns=["high"])
        report = validate_ohlcv(df)
        assert not report.is_valid

    def test_empty_df(self):
        df = pd.DataFrame()
        report = validate_ohlcv(df)
        assert not report.is_valid


# ── Sorting ──────────────────────────────────────────────────────────────────


class TestSortByTimestamp:
    def test_sorts_correctly(self):
        df = _make_valid_df()
        df_rev = df.iloc[::-1].reset_index(drop=True)
        df_sorted = sort_by_timestamp(df_rev)
        assert df_sorted["timestamp"].is_monotonic_increasing

    def test_already_sorted(self):
        df = _make_valid_df()
        df_sorted = sort_by_timestamp(df)
        pd.testing.assert_frame_equal(df, df_sorted)


# ── Duplicate removal ────────────────────────────────────────────────────────


class TestRemoveDuplicates:
    def test_no_dups(self):
        df = _make_valid_df()
        cleaned, n = remove_duplicates(df)
        assert n == 0
        assert len(cleaned) == len(df)

    def test_removes_identical_dups(self):
        df = _make_valid_df()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        cleaned, n = remove_duplicates(df)
        assert n == 1
        assert len(cleaned) == len(df) - 1

    def test_rejects_conflicting_duplicates_without_arbitrary_choice(self):
        df = _make_valid_df(n=3)
        conflicting_row = df.iloc[[0]].copy()
        conflicting_row["close"] = 999.0
        df = pd.concat([df, conflicting_row], ignore_index=True)

        cleaned, n_removed = remove_duplicates(df, reject_conflicts=True)
        assert n_removed == 2
        assert len(cleaned) == 2
        assert df.iloc[0]["timestamp"] not in set(cleaned["timestamp"])


# ── RTH filtering & Boundaries ───────────────────────────────────────────────


class TestFilterRTH:
    def test_exact_boundaries(self):
        """Test explicit boundary timestamps: 09:25, 09:30, 15:55, 16:00 ET."""
        ts = pd.Series(
            pd.to_datetime(
                [
                    "2026-01-05 14:25:00+00:00",  # 09:25 ET
                    "2026-01-05 14:30:00+00:00",  # 09:30 ET
                    "2026-01-05 20:55:00+00:00",  # 15:55 ET
                    "2026-01-05 21:00:00+00:00",  # 16:00 ET
                ]
            )
        )
        df = _make_valid_df(n=4)
        df["timestamp"] = ts

        filtered, n_removed = filter_rth(df)
        assert n_removed == 2
        assert len(filtered) == 2
        ts_et = filtered["timestamp"].dt.tz_convert("America/New_York")
        assert list(ts_et.dt.strftime("%H:%M")) == ["09:30", "15:55"]

    def test_dst_transition(self):
        """Verify candles near the spring DST transition (EDT starts, UTC-4)."""
        df = _make_valid_df(n=1, start="2026-03-09 13:30:00")
        filtered, n_removed = filter_rth(df)
        assert len(filtered) == 1
        assert n_removed == 0

    def test_early_close_schedule_entry_2026_07_03_excluded(self):
        """Ensure 2026-07-03 is NOT in DEFAULT_US_EARLY_CLOSES (it is a full holiday)."""
        assert "2026-07-03" not in DEFAULT_US_EARLY_CLOSES

    def test_early_close_session_filtering(self):
        """On an early-close session (13:00 close), candles >= 13:00 must be removed."""
        ts = pd.Series(
            pd.to_datetime(
                [
                    "2025-11-28 17:55:00+00:00",  # 12:55 ET -> kept
                    "2025-11-28 18:00:00+00:00",  # 13:00 ET -> filtered
                    "2025-11-28 18:05:00+00:00",  # 13:05 ET -> filtered
                ]
            )
        )
        df = _make_valid_df(n=3)
        df["timestamp"] = ts

        filtered, n_removed = filter_rth(df)
        assert n_removed == 2
        assert len(filtered) == 1
        ts_et = filtered["timestamp"].dt.tz_convert("America/New_York")
        assert ts_et.iloc[0].strftime("%H:%M") == "12:55"

    def test_custom_early_close_schedule(self):
        """Test with user-specified early close schedule."""
        custom_schedule = {"2026-02-10": "14:00"}
        ts = pd.Series(
            pd.to_datetime(
                [
                    "2026-02-10 18:55:00+00:00",  # 13:55 ET -> kept
                    "2026-02-10 19:00:00+00:00",  # 14:00 ET -> filtered
                ]
            )
        )
        df = _make_valid_df(n=2)
        df["timestamp"] = ts

        filtered, n_removed = filter_rth(df, early_close_schedule=custom_schedule)
        assert n_removed == 1
        assert len(filtered) == 1


# ── Gap detection & Partial Datasets ─────────────────────────────────────────


class TestDetectGaps:
    def test_no_gaps_in_full_session(self):
        df = _make_valid_df(n=78, start="2026-01-05 14:30:00")
        gaps = detect_gaps(df, rth_only=True)
        assert len(gaps) == 0

    def test_missing_first_candle_detected_in_full_session_mode(self):
        """When partial_dataset=False, day starting at 09:35 flags 09:30 as missing."""
        df = _make_valid_df(n=5, start="2026-01-05 14:35:00")
        gaps = detect_gaps(df, rth_only=True, partial_dataset=False)
        assert len(gaps) >= 1
        first_gap = gaps[0]
        assert first_gap.reason == "missing_session_start"
        assert first_gap.expected_time_utc == pd.Timestamp("2026-01-05 14:30:00+00:00")

    def test_missing_last_candle_detected_in_full_session_mode(self):
        """When partial_dataset=False, day ending at 15:50 flags 15:55 as missing."""
        df = _make_valid_df(n=77, start="2026-01-05 14:30:00")
        gaps = detect_gaps(df, rth_only=True, partial_dataset=False)
        assert len(gaps) == 1
        last_gap = gaps[0]
        assert last_gap.reason == "missing_session_end"
        assert last_gap.expected_time_utc == pd.Timestamp("2026-01-05 20:55:00+00:00")

    def test_partial_dataset_does_not_flag_unrequested_hours(self):
        """A partial slice from 11:00 to 14:00 ET with partial_dataset=True does not flag 09:30 or 15:55."""
        # 11:00 ET = 16:00 UTC, 14:00 ET = 19:00 UTC (37 candles)
        # Drop 12:00 ET (17:00 UTC) to simulate a genuine internal gap
        all_ts = pd.date_range("2026-01-05 16:00:00", "2026-01-05 19:00:00", freq="5min", tz="UTC")
        partial_ts = all_ts[all_ts != pd.Timestamp("2026-01-05 17:00:00+00:00")]

        df = _make_valid_df(n=len(partial_ts))
        df["timestamp"] = partial_ts

        gaps = detect_gaps(df, rth_only=True, partial_dataset=True)
        # Exactly one gap (the 12:00 ET candle); 09:30-10:55 and 14:05-15:55 are not flagged
        assert len(gaps) == 1
        assert gaps[0].expected_time_utc == pd.Timestamp("2026-01-05 17:00:00+00:00")
        assert gaps[0].reason == "missing_candle"

    def test_missing_intermediate_candle_detected(self):
        df = _make_valid_df(n=5, start="2026-01-05 14:30:00")
        df_dropped = df.drop(index=2).reset_index(drop=True)
        gaps = detect_gaps(df_dropped, rth_only=False)
        assert len(gaps) == 1
        assert gaps[0].expected_time_utc == pd.Timestamp("2026-01-05 14:40:00+00:00")

    def test_early_close_does_not_falsely_flag_post_close_candles(self):
        """On an early close day (42 candles, 09:30 to 12:55), no gaps are reported."""
        df = _make_valid_df(n=42, start="2025-11-28 14:30:00")
        gaps = detect_gaps(df, rth_only=True)
        assert len(gaps) == 0

    def test_completely_missing_session_detected(self):
        """If an expected date is provided but missing from df, detect full missing session."""
        day1 = _make_valid_df(n=78, start="2026-01-05 14:30:00")
        gaps = detect_gaps(
            day1,
            rth_only=True,
            expected_dates=["2026-01-05", "2026-01-06"],
        )
        missing_day_gaps = [g for g in gaps if g.trading_date == "2026-01-06"]
        assert len(missing_day_gaps) == 78
        assert all(g.reason == "missing_full_session" for g in missing_day_gaps)

    def test_empty_df_returns_empty_or_calendar_gaps(self):
        df = pd.DataFrame()
        assert detect_gaps(df, rth_only=True) == []

    def test_unsorted_raises(self):
        df = _make_valid_df(n=5)
        df = df.iloc[::-1].reset_index(drop=True)
        with pytest.raises(ValueError, match="sorted"):
            detect_gaps(df)


# ── CSV loading & Timezone policy (including mixed series) ───────────────────


class TestLoadRawCSV:
    def test_loads_valid_csv(self, tmp_path: Path):
        df = _make_valid_df()
        csv_path = _write_csv(df, tmp_path / "test.csv")
        loaded = load_raw_csv(csv_path)
        assert len(loaded) == len(df)
        assert hasattr(loaded["timestamp"].dtype, "tz")
        assert str(loaded["timestamp"].dtype.tz) == "UTC"

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_raw_csv(tmp_path / "nonexistent.csv")

    def test_missing_timestamp_column(self, tmp_path: Path):
        df = _make_valid_df().drop(columns=["timestamp"])
        df.to_csv(tmp_path / "bad.csv", index=False)
        with pytest.raises(ValueError, match="timestamp"):
            load_raw_csv(tmp_path / "bad.csv")

    def test_naive_timestamps_rejected_by_default(self, tmp_path: Path):
        """Default behavior MUST reject timezone-naive timestamps."""
        df = _make_valid_df()
        df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        csv_path = tmp_path / "naive.csv"
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="timezone-naive"):
            load_raw_csv(csv_path)

    def test_mixed_timestamps_rejected_by_default(self, tmp_path: Path):
        """A CSV with both aware and naive timestamps must be rejected by default."""
        df = _make_valid_df(n=2)
        # Row 0 aware, Row 1 naive
        df["timestamp"] = ["2026-01-05 14:30:00+00:00", "2026-01-05 14:35:00"]
        csv_path = tmp_path / "mixed_reject.csv"
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="timezone-naive"):
            load_raw_csv(csv_path)

    def test_mixed_timestamps_assume_utc(self, tmp_path: Path):
        """Mixed timestamps with naive_tz_policy='assume_utc' converts all to UTC."""
        df = _make_valid_df(n=2)
        df["timestamp"] = ["2026-01-05 14:30:00+00:00", "2026-01-05 14:35:00"]
        csv_path = tmp_path / "mixed_assume.csv"
        df.to_csv(csv_path, index=False)

        loaded = load_raw_csv(csv_path, naive_tz_policy="assume_utc")
        assert hasattr(loaded["timestamp"].dtype, "tz")
        assert str(loaded["timestamp"].dtype.tz) == "UTC"
        assert loaded["timestamp"].iloc[0] == pd.Timestamp("2026-01-05 14:30:00+00:00")
        assert loaded["timestamp"].iloc[1] == pd.Timestamp("2026-01-05 14:35:00+00:00")

    def test_mixed_timestamps_explicit_tz_conversion(self, tmp_path: Path):
        """Mixed timestamps with explicit timezone (America/New_York) converts naive from ET to UTC."""
        df = _make_valid_df(n=2)
        # Row 0 is already UTC 14:30 (09:30 ET)
        # Row 1 is naive local NY time 09:35 ET
        df["timestamp"] = ["2026-01-05 14:30:00+00:00", "2026-01-05 09:35:00"]
        csv_path = tmp_path / "mixed_ny.csv"
        df.to_csv(csv_path, index=False)

        loaded = load_raw_csv(csv_path, naive_tz_policy="America/New_York")
        assert loaded["timestamp"].iloc[0] == pd.Timestamp("2026-01-05 14:30:00+00:00")
        assert loaded["timestamp"].iloc[1] == pd.Timestamp("2026-01-05 14:35:00+00:00")

    def test_nonexistent_and_ambiguous_local_times_raise_clear_error(self, tmp_path: Path):
        """Spring forward nonexistent time and fall back ambiguous time must raise clear errors."""
        # Nonexistent local time (2026-03-08 02:30 in America/New_York)
        df_nonexist = _make_valid_df(n=1)
        df_nonexist["timestamp"] = "2026-03-08 02:30:00"
        csv_nonexist = tmp_path / "nonexistent.csv"
        df_nonexist.to_csv(csv_nonexist, index=False)

        with pytest.raises(ValueError, match="ambiguous or nonexistent"):
            load_raw_csv(csv_nonexist, naive_tz_policy="America/New_York")

        # Ambiguous local time (2026-11-01 01:30 in America/New_York)
        df_ambig = _make_valid_df(n=1)
        df_ambig["timestamp"] = "2026-11-01 01:30:00"
        csv_ambig = tmp_path / "ambiguous.csv"
        df_ambig.to_csv(csv_ambig, index=False)

        with pytest.raises(ValueError, match="ambiguous or nonexistent"):
            load_raw_csv(csv_ambig, naive_tz_policy="America/New_York")


# ── Full pipeline & Invalid Row Policy ───────────────────────────────────────


class TestPreprocess:
    def test_full_pipeline_valid_data(self, tmp_path: Path):
        df = _make_valid_df(n=78, start="2026-01-05 14:30:00")
        csv_path = _write_csv(df, tmp_path / "raw.csv")
        out_path = tmp_path / "processed" / "clean.csv"

        result, report = preprocess(csv_path, output_path=out_path)
        assert report.is_valid
        assert out_path.exists()
        assert len(result) == 78
        assert report.info["raw_rows"] == 78
        assert "calendar_limitation_note" in report.info

    def test_empty_input_is_explicitly_invalid_and_not_saved(self, tmp_path: Path):
        csv_path = tmp_path / "empty.csv"
        header = ",".join(REQUIRED_COLUMNS) + "\n"
        csv_path.write_text(header)
        out_path = tmp_path / "processed" / "empty_out.csv"

        result, report = preprocess(csv_path, output_path=out_path)
        assert not report.is_valid
        assert any("empty" in e.lower() for e in report.errors)
        assert not out_path.exists()

    def test_pipeline_fails_on_invalid_data_by_default(self, tmp_path: Path):
        df = _make_valid_df(n=10, start="2026-01-05 14:30:00")
        df.loc[2, "close"] = -10.0
        df.loc[3, "high"] = np.inf
        csv_path = _write_csv(df, tmp_path / "invalid_raw.csv")
        out_path = tmp_path / "processed" / "should_not_exist.csv"

        result, report = preprocess(csv_path, output_path=out_path, drop_invalid=False)
        assert not report.is_valid
        assert not out_path.exists()
        assert len(report.errors) > 0

    def test_pipeline_cleans_and_reports_when_drop_invalid_true(self, tmp_path: Path):
        df = _make_valid_df(n=10, start="2026-01-05 14:30:00")
        df.loc[2, "high"] = df.loc[2, "low"] - 1.0  # OHLC violation
        df.loc[4, "volume"] = -5.0  # Negative volume
        csv_path = _write_csv(df, tmp_path / "bad_rows.csv")
        out_path = tmp_path / "processed" / "cleaned_out.csv"

        result, report = preprocess(csv_path, output_path=out_path, drop_invalid=True)
        assert report.is_valid
        assert out_path.exists()
        assert len(result) == 8
        assert "removed_records" in report.info
        assert report.info["removed_records"]["total_removed"] == 2

    def test_drop_invalid_handles_identical_duplicates(self, tmp_path: Path):
        """drop_invalid=True must remove identical duplicates and pass re-validation."""
        df = _make_valid_df(n=5, start="2026-01-05 14:30:00")
        # Add identical copy of row 0
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        csv_path = _write_csv(df, tmp_path / "identical_dups.csv")
        out_path = tmp_path / "processed" / "deduped.csv"

        result, report = preprocess(csv_path, output_path=out_path, drop_invalid=True)
        assert report.is_valid
        assert out_path.exists()
        assert len(result) == 5
        assert report.info["removed_records"]["n_identical_dups_removed"] == 1

    def test_pipeline_removes_pre_post_market(self, tmp_path: Path):
        pre = _make_valid_df(n=3, start="2026-01-05 14:00:00")
        rth = _make_valid_df(n=5, start="2026-01-05 14:30:00")
        combined = pd.concat([pre, rth], ignore_index=True)
        csv_path = _write_csv(combined, tmp_path / "raw.csv")

        result, report = preprocess(csv_path)
        assert len(result) == 5
        assert report.info["rth_candles_removed"] == 3

    def test_pipeline_without_session_filter(self, tmp_path: Path):
        pre = _make_valid_df(n=3, start="2026-01-05 14:00:00")
        rth = _make_valid_df(n=5, start="2026-01-05 14:30:00")
        combined = pd.concat([pre, rth], ignore_index=True)
        csv_path = _write_csv(combined, tmp_path / "raw.csv")

        result, report = preprocess(csv_path, filter_session=False)
        assert len(result) == 8
        assert report.info["rth_candles_removed"] == 0

    def test_output_does_not_overwrite_input(self, tmp_path: Path):
        df = _make_valid_df(n=5, start="2026-01-05 14:30:00")
        csv_path = _write_csv(df, tmp_path / "raw.csv")
        out_path = tmp_path / "processed" / "clean.csv"

        preprocess(csv_path, output_path=out_path)
        raw_again = pd.read_csv(csv_path)
        assert len(raw_again) == 5

    def test_creates_parent_dirs(self, tmp_path: Path):
        df = _make_valid_df(n=5, start="2026-01-05 14:30:00")
        csv_path = _write_csv(df, tmp_path / "raw.csv")
        deep_path = tmp_path / "nested" / "sub" / "clean.csv"

        preprocess(csv_path, output_path=deep_path)
        assert deep_path.exists()
