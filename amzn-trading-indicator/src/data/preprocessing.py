"""Raw-data cleaning and session handling for AMZN 5-minute OHLCV data.

This module provides functions to:
  - Load and parse raw Alpaca CSV exports with explicit timezone policy.
  - Validate schema, timestamps, OHLCV consistency (types, finite values, logic).
  - Detect duplicates (identical vs. conflicting) and missing values.
  - Detect gaps, including missing session start, intermediate, and session end candles,
    with explicit support for partial intraday dataset boundaries.
  - Filter to regular trading hours (RTH) with early-close session awareness.
  - Report all invalid row counts, removals, and rejections without silent data drops.
  - Save cleaned output safely only when preprocessing validation succeeds.

Design principles:
  - Internal representation strictly uses UTC-aware timestamps.
  - Timezone-naive timestamps are rejected by default unless an explicit policy is specified.
  - Nothing is silently dropped; every rejection or removal is tracked in ValidationReport.
  - No technical indicators or targets are computed in this module.
  - No candles are fabricated for missing intervals.
  - Static early-close schedules are provided for standard US market half-days, but this
    does NOT substitute for a full dynamic exchange calendar.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

REQUIRED_COLUMNS: list[str] = [
    "symbol",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "vwap",
]

NUMERIC_OHLCV_COLUMNS: list[str] = ["open", "high", "low", "close", "volume"]

CANDLE_INTERVAL = pd.Timedelta(minutes=5)

# Regular trading hours in America/New_York.
# Start is inclusive (09:30 candle opens the session).
# End is exclusive (16:00 candle is the closing cross / post-market, excluded from RTH).
# This yields candles from 09:30 through 15:55 — 78 bars on a standard full session day.
RTH_START = "09:30"
RTH_END = "16:00"
RTH_TIMEZONE = "America/New_York"

# Standard scheduled US market early closes (session ends at 13:00 ET).
# Last 5-minute candle on early-close days starts at 12:55 ET.
#
# IMPORTANT LIMITATIONS OF HARDCODED EARLY CLOSES:
# 1. This dictionary contains only standard, recurring scheduled early-close days
#    (e.g., 13:00 ET close on Black Friday and Christmas Eve).
# 2. It is NOT a full exchange calendar:
#    - It does NOT automatically reflect unscheduled early closures, emergency halts,
#      severe weather closures, or national days of mourning (e.g. presidential funerals).
#    - It does not dynamically verify whether an exchange declared an ad-hoc session change.
#    - For example, Friday, 2026-07-03 is the observed Independence Day full market holiday
#      (markets are fully closed), NOT an early close session.
# 3. For production deployment, using an official exchange calendar library (such as
#    `exchange_calendars` or Alpaca's `/v2/calendar` API endpoint) is strongly recommended
#    to maintain an authoritative trading schedule.
DEFAULT_US_EARLY_CLOSES: dict[str, str] = {
    # 2025
    "2025-07-03": "13:00",  # Day before Independence Day (Thursday before holiday Friday)
    "2025-11-28": "13:00",  # Day after Thanksgiving (Black Friday)
    "2025-12-24": "13:00",  # Christmas Eve
    # 2026
    # Note: 2026-07-03 is an observed Independence Day full market holiday (closed all day), NOT an early close.
    "2026-11-27": "13:00",  # Day after Thanksgiving (Black Friday)
    "2026-12-24": "13:00",  # Christmas Eve
}


# ── Data classes for structured reporting ────────────────────────────────────


@dataclass
class ValidationReport:
    """Structured report returned by validation helpers and preprocess()."""

    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict[str, object] = field(default_factory=dict)


@dataclass
class GapInfo:
    """Describes a gap (missing candle interval) in the time series."""

    trading_date: str  # YYYY-MM-DD in ET
    expected_time_utc: pd.Timestamp
    previous_time_utc: pd.Timestamp | None
    gap_size: pd.Timedelta
    reason: str = "missing_candle"  # 'missing_candle', 'missing_session_start', 'missing_session_end', 'missing_full_session'


# ── Loading ──────────────────────────────────────────────────────────────────


def load_raw_csv(
    path: str | Path,
    *,
    naive_tz_policy: str = "reject",
) -> pd.DataFrame:
    """Load a raw Alpaca CSV and parse timestamps as UTC-aware datetime.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.
    naive_tz_policy : str
        Policy for timezone-naive timestamps:
        - "reject" (default): raise ValueError if any timestamp lacks timezone info.
        - "assume_utc": assume naive timestamps are UTC, with a warning.
        - valid tz string (e.g. "America/New_York"): localize to this timezone
          and convert to UTC.

    Returns
    -------
    pd.DataFrame
        DataFrame with a ``timestamp`` column of dtype ``datetime64[ns, UTC]``.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the ``timestamp`` column is missing or naive timestamps are rejected.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {path}")

    df = pd.read_csv(path)

    if "timestamp" not in df.columns:
        raise ValueError("CSV is missing the required 'timestamp' column.")

    if df.empty:
        df["timestamp"] = pd.Series(dtype="datetime64[ns, UTC]")
        return df

    df["timestamp"] = _parse_timestamps(df["timestamp"], naive_tz_policy=naive_tz_policy)
    return df


def _parse_timestamps(
    series: pd.Series,
    *,
    naive_tz_policy: str = "reject",
) -> pd.Series:
    """Parse a Series into UTC-aware Timestamps according to naive_tz_policy.

    Handles mixed timezone-aware and timezone-naive timestamps:
      - Aware timestamps (with UTC or numeric offset) are parsed directly to UTC.
      - Naive timestamps are handled according to `naive_tz_policy`:
        * "reject": raises ValueError.
        * "assume_utc": assumed to be UTC with a logged warning.
        * <timezone_name> (e.g. "America/New_York"): localized to this timezone
          and converted to UTC. Clear ValueErrors are raised for ambiguous
          or nonexistent local times during DST transitions.

    Raises
    ------
    ValueError
        If naive timestamps are rejected or cannot be localized.
    """
    if series.empty:
        return pd.Series(dtype="datetime64[ns, UTC]")

    # Check if series is already a datetime dtype
    if pd.api.types.is_datetime64_any_dtype(series):
        if series.dt.tz is None:
            if naive_tz_policy == "reject":
                raise ValueError(
                    f"Found {len(series)} timezone-naive timestamps. Naive timestamps "
                    "are rejected by default. Set naive_tz_policy='assume_utc' or a specific source timezone."
                )
            elif naive_tz_policy == "assume_utc":
                logger.warning("Localizing %d naive timestamps to UTC.", len(series))
                return series.dt.tz_localize("UTC")
            else:
                logger.warning("Localizing %d naive timestamps to %s and converting to UTC.", len(series), naive_tz_policy)
                try:
                    return series.dt.tz_localize(naive_tz_policy).dt.tz_convert("UTC")
                except Exception as exc:
                    raise ValueError(
                        f"Failed to localize naive timestamps to timezone '{naive_tz_policy}' "
                        f"(possible ambiguous or nonexistent local time during DST transition): {exc}"
                    ) from exc
        return series.dt.tz_convert("UTC")

    # String series: inspect each element for timezone offset (+HH:MM, -HH:MM, or Z)
    raw_strings = series.dropna().astype(str).str.strip()
    has_tz = raw_strings.str.contains(r"[+\-]\d{2}:?\d{2}|Z$", regex=True)
    n_naive = int((~has_tz).sum())

    if n_naive > 0:
        if naive_tz_policy == "reject":
            raise ValueError(
                f"Found {n_naive} timezone-naive timestamps. Naive timestamps are "
                "rejected by default. Set naive_tz_policy='assume_utc' or specify a source timezone."
            )
        elif naive_tz_policy == "assume_utc":
            logger.warning("%d timestamps had no timezone info and were assumed UTC.", n_naive)
            return pd.to_datetime(series, utc=True, format="mixed", errors="coerce")
        else:
            logger.warning("%d timestamps had no timezone info and were localized to %s.", n_naive, naive_tz_policy)
            res = pd.Series(index=series.index, dtype="datetime64[ns, UTC]")
            if has_tz.any():
                res[has_tz] = pd.to_datetime(series[has_tz], utc=True, format="mixed", errors="coerce")
            if (~has_tz).any():
                naive_dt = pd.to_datetime(series[~has_tz], format="mixed", errors="coerce")
                try:
                    localized = naive_dt.dt.tz_localize(naive_tz_policy)
                except Exception as exc:
                    raise ValueError(
                        f"Failed to localize naive timestamps to timezone '{naive_tz_policy}' "
                        f"(possible ambiguous or nonexistent local time during DST transition): {exc}"
                    ) from exc
                res[~has_tz] = localized.dt.tz_convert("UTC")
            return res

    return pd.to_datetime(series, utc=True, format="mixed", errors="coerce")


# ── Validation helpers ───────────────────────────────────────────────────────


def validate_columns(
    df: pd.DataFrame,
    required: Sequence[str] = REQUIRED_COLUMNS,
) -> ValidationReport:
    """Check that all required columns are present.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.
    required : sequence of str
        Column names that must exist.

    Returns
    -------
    ValidationReport
    """
    report = ValidationReport()
    missing = set(required) - set(df.columns)
    if missing:
        report.is_valid = False
        report.errors.append(f"Missing required columns: {sorted(missing)}")
    extra = set(df.columns) - set(required)
    if extra:
        report.info["extra_columns"] = sorted(extra)
    return report


def validate_timestamps(df: pd.DataFrame) -> ValidationReport:
    """Validate that the ``timestamp`` column is well-formed, timezone-aware, and non-null.

    Checks:
      - DataFrame non-empty
      - ``timestamp`` column exists
      - Detects and counts NaT values
      - Verifies timezone-awareness
      - Records monotonic ordering

    Returns
    -------
    ValidationReport
    """
    report = ValidationReport()

    if df.empty:
        report.is_valid = False
        report.errors.append("DataFrame is empty.")
        return report

    ts = df.get("timestamp")
    if ts is None:
        report.is_valid = False
        report.errors.append("No 'timestamp' column found.")
        return report

    # 1. Explicit NaT detection
    n_nat = int(ts.isna().sum())
    report.info["n_nat"] = n_nat
    if n_nat > 0:
        report.is_valid = False
        report.errors.append(f"{n_nat} timestamp(s) are NaT (missing or unparseable).")

    # 2. Timezone-awareness check
    if not hasattr(ts.dtype, "tz") or ts.dtype.tz is None:
        report.is_valid = False
        report.errors.append(
            "Timestamps are not timezone-aware. "
            "Call load_raw_csv() or convert to UTC first."
        )
        return report

    # 3. Monotonic sorting & bounds (computed on valid timestamps)
    valid_ts = ts.dropna()
    if not valid_ts.empty:
        report.info["min_timestamp"] = valid_ts.min()
        report.info["max_timestamp"] = valid_ts.max()
        is_sorted = bool(valid_ts.is_monotonic_increasing)
        report.info["is_sorted"] = is_sorted
        if not is_sorted:
            report.warnings.append("Timestamps are not in ascending order.")

    return report


def validate_no_duplicates(df: pd.DataFrame) -> ValidationReport:
    """Check for duplicate timestamps, distinguishing identical from conflicting duplicates.

    Returns
    -------
    ValidationReport
        ``info["n_duplicates"]``: total rows with non-unique timestamps.
        ``info["n_conflicting_duplicates"]``: rows with duplicate timestamps and conflicting data.
    """
    report = ValidationReport()
    if df.empty or "timestamp" not in df.columns:
        report.info["n_duplicates"] = 0
        report.info["n_conflicting_duplicates"] = 0
        return report

    ts = df["timestamp"]
    dup_mask = ts.duplicated(keep=False)
    n_dups = int(dup_mask.sum())
    report.info["n_duplicates"] = n_dups

    if n_dups > 0:
        report.is_valid = False
        dup_values = ts[dup_mask].unique().tolist()
        report.info["duplicate_timestamps"] = dup_values

        # Check whether duplicate timestamp rows have identical or conflicting data
        dup_subset = df[dup_mask]
        unique_dup_rows = dup_subset.drop_duplicates()
        ts_counts_in_unique = unique_dup_rows["timestamp"].value_counts()
        conflicting_ts = ts_counts_in_unique[ts_counts_in_unique > 1].index.tolist()

        conflicting_rows_mask = ts.isin(conflicting_ts)
        n_conflicts = int(conflicting_rows_mask.sum())
        report.info["n_conflicting_duplicates"] = n_conflicts
        report.info["conflicting_timestamps"] = conflicting_ts

        if n_conflicts > 0:
            report.errors.append(
                f"{n_conflicts} rows have conflicting duplicate timestamps ({len(conflicting_ts)} distinct timestamp(s))."
            )
        else:
            report.warnings.append(
                f"{n_dups} rows are exact duplicate rows with identical data."
            )
            report.errors.append(f"{n_dups} rows have duplicated timestamps.")
    else:
        report.info["n_conflicting_duplicates"] = 0

    return report


def validate_no_missing_values(df: pd.DataFrame) -> ValidationReport:
    """Check for NaN/null values in every column.

    Returns
    -------
    ValidationReport
        ``info["null_counts"]``: dict of column -> count for columns with >=1 null.
    """
    report = ValidationReport()
    if df.empty:
        report.is_valid = False
        report.errors.append("DataFrame is empty.")
        return report

    null_counts = df.isnull().sum()
    bad = null_counts[null_counts > 0]
    if len(bad) > 0:
        report.is_valid = False
        report.info["null_counts"] = bad.to_dict()
        report.errors.append(f"Null values found in columns: {list(bad.index)}")
    return report


def validate_ohlcv(df: pd.DataFrame) -> ValidationReport:
    """Validate OHLCV data types, finiteness, and logical price consistency.

    Checks:
      1. Numeric dtype verification for each column.
      2. Missing / NaN values.
      3. Positive and negative infinity (+inf, -inf).
      4. Price consistency:
         - high >= low
         - high >= open and high >= close
         - low <= open and low <= close
         - volume >= 0

    Returns
    -------
    ValidationReport
    """
    report = ValidationReport()

    if df.empty:
        report.is_valid = False
        report.errors.append("DataFrame is empty.")
        return report

    # 1. Column existence
    for col in NUMERIC_OHLCV_COLUMNS:
        if col not in df.columns:
            report.is_valid = False
            report.errors.append(f"Cannot validate OHLCV: '{col}' column missing.")
            return report

    # 2. Check numeric types, NaNs, and infinities
    has_type_or_finite_error = False
    for col in NUMERIC_OHLCV_COLUMNS:
        s = df[col]

        # Numeric type check
        if not pd.api.types.is_numeric_dtype(s):
            has_type_or_finite_error = True
            report.is_valid = False
            s_num = pd.to_numeric(s, errors="coerce")
            non_num_count = int(s.notna().sum() - s_num.notna().sum())
            report.info[f"{col}_non_numeric"] = non_num_count
            report.errors.append(
                f"Column '{col}' has non-numeric dtype ({s.dtype}) with {non_num_count} invalid value(s)."
            )
            continue

        # NaN check
        n_nan = int(s.isna().sum())
        if n_nan > 0:
            has_type_or_finite_error = True
            report.is_valid = False
            report.info[f"{col}_nan"] = n_nan
            report.errors.append(f"Column '{col}' contains {n_nan} NaN value(s).")

        # Infinity check
        pos_inf = int(np.isposinf(s).sum())
        neg_inf = int(np.isneginf(s).sum())
        n_inf = pos_inf + neg_inf
        if n_inf > 0:
            has_type_or_finite_error = True
            report.is_valid = False
            report.info[f"{col}_inf"] = n_inf
            report.errors.append(
                f"Column '{col}' contains {n_inf} infinite value(s) ({pos_inf} +inf, {neg_inf} -inf)."
            )

    if has_type_or_finite_error:
        return report

    # 3. Logical price consistency checks on clean finite numbers
    checks = {
        "high_lt_low": df["high"] < df["low"],
        "high_lt_open": df["high"] < df["open"],
        "high_lt_close": df["high"] < df["close"],
        "low_gt_open": df["low"] > df["open"],
        "low_gt_close": df["low"] > df["close"],
        "negative_volume": df["volume"] < 0,
    }

    for name, mask in checks.items():
        n = int(mask.sum())
        report.info[name] = n
        if n > 0:
            report.is_valid = False
            report.errors.append(f"OHLCV violation '{name}': {n} rows.")

    return report


# ── Detection helpers (non-destructive) ─────────────────────────────────────


def detect_gaps(
    df: pd.DataFrame,
    interval: pd.Timedelta = CANDLE_INTERVAL,
    rth_only: bool = True,
    start: str = RTH_START,
    end: str = RTH_END,
    tz: str = RTH_TIMEZONE,
    early_close_schedule: dict[str, str] | None = None,
    expected_dates: Sequence[str | datetime.date] | None = None,
    partial_dataset: bool = False,
) -> list[GapInfo]:
    """Detect missing candle intervals in the time series.

    Compares observed candles against expected session timestamps to identify:
      - Missing opening session candle (e.g. 09:30 ET missing)
      - Intermediate missing candles
      - Missing closing session candle (e.g. 15:55 ET missing, or 12:55 on early close)
      - Entirely missing sessions (when expected_dates is provided)

    Handling Partial Dataset Boundaries:
      When `partial_dataset=False` (default for full historical datasets), the function
      expects every trading day present to have full RTH coverage from `start` to `end`.
      When `partial_dataset=True`, the expected window for the dataset's first trading
      date begins at the earliest observed candle on that date, and the expected window
      for the latest trading date ends at the latest observed candle. This prevents
      misinterpreting unrequested pre- or post-query hours as data losses, while still
      rigorously detecting gaps within the covered interval.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a sorted, UTC-aware ``timestamp`` column.
    interval : pd.Timedelta
        Expected interval between consecutive candles (default 5 minutes).
    rth_only : bool
        If True (default), compare observed candles to expected RTH session timestamps.
        If False, detect any consecutive interval > interval across the dataset.
    start : str
        Session open time in ET (default "09:30").
    end : str
        Normal session close time in ET (default "16:00").
    tz : str
        Session timezone name (default "America/New_York").
    early_close_schedule : dict, optional
        Map of "YYYY-MM-DD" -> early close time (e.g. "13:00").
        Defaults to DEFAULT_US_EARLY_CLOSES.
    expected_dates : sequence of str or datetime.date, optional
        Explicit calendar dates expected to be present. If a date is completely
        absent from ``df``, all session candles for that day are flagged as missing.
    partial_dataset : bool
        If True, accommodate partial intraday dataset boundaries without flagging
        unrequested hours outside the dataset's start/end timestamps.

    Returns
    -------
    list[GapInfo]
        One entry per missing expected candle.
    """
    if df.empty and not expected_dates:
        return []

    schedule = dict(DEFAULT_US_EARLY_CLOSES)
    if early_close_schedule:
        schedule.update(early_close_schedule)

    ts = df.get("timestamp", pd.Series(dtype="datetime64[ns, UTC]"))
    if not ts.empty and not ts.is_monotonic_increasing:
        raise ValueError("DataFrame must be sorted by timestamp before gap detection.")

    gaps: list[GapInfo] = []

    if not rth_only:
        if len(ts) < 2:
            return []
        diffs = ts.diff().iloc[1:]
        for idx in diffs[diffs > interval].index:
            prev = ts.iloc[idx - 1]
            curr = ts.iloc[idx]
            date_et = prev.tz_convert(tz).date()
            expected = prev + interval
            while expected < curr:
                gaps.append(
                    GapInfo(
                        trading_date=str(date_et),
                        expected_time_utc=expected,
                        previous_time_utc=prev,
                        gap_size=interval,
                        reason="missing_candle",
                    )
                )
                expected += interval
        return gaps

    # RTH session comparison against expected timetable
    if not ts.empty:
        ts_et = ts.dt.tz_convert(tz)
        observed_by_date = {
            d: set(grp) for d, grp in ts.groupby(ts_et.dt.date)
        }
        dataset_min_utc = ts.min()
        dataset_max_utc = ts.max()
        earliest_date = dataset_min_utc.tz_convert(tz).date()
        latest_date = dataset_max_utc.tz_convert(tz).date()
    else:
        observed_by_date = {}
        dataset_min_utc = None
        dataset_max_utc = None
        earliest_date = None
        latest_date = None

    dates_to_check: list[datetime.date] = []
    if expected_dates:
        for ed in expected_dates:
            d = pd.Timestamp(ed).date()
            if d not in dates_to_check:
                dates_to_check.append(d)

    for d in sorted(observed_by_date.keys()):
        if d not in dates_to_check:
            dates_to_check.append(d)

    for date in sorted(dates_to_check):
        date_str = str(date)
        session_close_time = schedule.get(date_str, end)

        # Determine start boundary for date
        if partial_dataset and dataset_min_utc is not None and date == earliest_date:
            session_start_dt = max(
                pd.Timestamp(f"{date_str} {start}", tz=tz),
                dataset_min_utc.tz_convert(tz),
            )
        else:
            session_start_dt = pd.Timestamp(f"{date_str} {start}", tz=tz)

        # Determine end boundary for date
        if partial_dataset and dataset_max_utc is not None and date == latest_date:
            session_end_dt = min(
                pd.Timestamp(f"{date_str} {session_close_time}", tz=tz),
                dataset_max_utc.tz_convert(tz) + interval,
            )
        else:
            session_end_dt = pd.Timestamp(f"{date_str} {session_close_time}", tz=tz)

        if session_start_dt >= session_end_dt:
            continue

        expected_range_local = pd.date_range(
            start=session_start_dt,
            end=session_end_dt,
            freq=interval,
            inclusive="left",
            tz=tz,
        )
        expected_utc = expected_range_local.tz_convert("UTC")

        observed_utc_set = observed_by_date.get(date, set())

        # Entirely missing session
        if len(observed_utc_set) == 0:
            for exp_ts in expected_utc:
                gaps.append(
                    GapInfo(
                        trading_date=date_str,
                        expected_time_utc=exp_ts,
                        previous_time_utc=None,
                        gap_size=interval,
                        reason="missing_full_session",
                    )
                )
            continue

        # Per-candle check
        sorted_observed = sorted(observed_utc_set)
        for i, exp_ts in enumerate(expected_utc):
            if exp_ts not in observed_utc_set:
                if i == 0:
                    reason = "missing_session_start"
                elif i == len(expected_utc) - 1:
                    reason = "missing_session_end"
                else:
                    reason = "missing_candle"

                earlier_observed = [o for o in sorted_observed if o < exp_ts]
                prev_ts = earlier_observed[-1] if earlier_observed else None

                gaps.append(
                    GapInfo(
                        trading_date=date_str,
                        expected_time_utc=exp_ts,
                        previous_time_utc=prev_ts,
                        gap_size=interval,
                        reason=reason,
                    )
                )

    return gaps


# ── Cleaning / transformation ────────────────────────────────────────────────


def sort_by_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy sorted chronologically by ``timestamp``."""
    return df.sort_values("timestamp").reset_index(drop=True)


def remove_duplicates(
    df: pd.DataFrame,
    *,
    reject_conflicts: bool = True,
) -> tuple[pd.DataFrame, int]:
    """Remove duplicate timestamp rows.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    reject_conflicts : bool
        If True (default), conflicting duplicate rows (same timestamp, differing values)
        are NOT arbitrarily picked; all conflicting rows are removed and logged.
        Identical duplicate rows (exact same data) retain a single copy.

    Returns
    -------
    tuple[pd.DataFrame, int]
        Cleaned DataFrame and the number of removed rows.
    """
    n_before = len(df)
    if n_before == 0 or "timestamp" not in df.columns:
        return df.copy(), 0

    ts = df["timestamp"]
    dup_mask = ts.duplicated(keep=False)
    if not dup_mask.any():
        return df.copy(), 0

    if reject_conflicts:
        dup_subset = df[dup_mask]
        unique_dup_rows = dup_subset.drop_duplicates()
        ts_counts_in_unique = unique_dup_rows["timestamp"].value_counts()
        conflicting_ts = ts_counts_in_unique[ts_counts_in_unique > 1].index

        if len(conflicting_ts) > 0:
            logger.warning(
                "Rejecting %d conflicting duplicate timestamp rows across %d timestamps. "
                "No arbitrary row was chosen.",
                int(ts.isin(conflicting_ts).sum()),
                len(conflicting_ts),
            )
            df_no_conflicts = df[~ts.isin(conflicting_ts)]
            df_clean = df_no_conflicts.drop_duplicates(subset="timestamp", keep="first").reset_index(drop=True)
        else:
            df_clean = df.drop_duplicates(subset="timestamp", keep="first").reset_index(drop=True)
    else:
        df_clean = df.drop_duplicates(subset="timestamp", keep="first").reset_index(drop=True)

    n_removed = n_before - len(df_clean)
    if n_removed > 0:
        logger.info("Removed %d duplicate-timestamp rows in total.", n_removed)
    return df_clean, n_removed


def filter_rth(
    df: pd.DataFrame,
    start: str = RTH_START,
    end: str = RTH_END,
    tz: str = RTH_TIMEZONE,
    early_close_schedule: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Filter to the regular trading session (RTH).

    Keeps candles whose **start** time (the ``timestamp`` value) falls within
    ``[start, session_end)`` in local market time.

    On standard days: keeps candles from 09:30 through 15:55.
    On early close days (e.g. 13:00 close): keeps candles from 09:30 through 12:55.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a UTC-aware ``timestamp`` column.
    start : str
        Session start in HH:MM format (ET, default "09:30").
    end : str
        Standard session close in HH:MM format (ET, default "16:00").
    tz : str
        Market timezone name (default "America/New_York").
    early_close_schedule : dict, optional
        Map of "YYYY-MM-DD" -> early close time (default DEFAULT_US_EARLY_CLOSES).

    Returns
    -------
    tuple[pd.DataFrame, int]
        Filtered DataFrame and the number of rows removed.
    """
    if df.empty:
        return df.copy(), 0

    schedule = dict(DEFAULT_US_EARLY_CLOSES)
    if early_close_schedule:
        schedule.update(early_close_schedule)

    ts_local = df["timestamp"].dt.tz_convert(tz)
    local_dates = ts_local.dt.date.astype(str)
    local_times = ts_local.dt.time

    start_time = pd.Timestamp(start).time()

    unique_dates = local_dates.unique()
    date_end_times = {
        d: pd.Timestamp(schedule.get(d, end)).time() for d in unique_dates
    }
    end_times = local_dates.map(date_end_times)

    mask = (local_times >= start_time) & (local_times < end_times)
    n_removed = int((~mask).sum())
    df_rth = df.loc[mask].reset_index(drop=True)

    if n_removed > 0:
        logger.info("RTH filter removed %d pre/post-market candles.", n_removed)

    return df_rth, n_removed


# ── Full pipeline ────────────────────────────────────────────────────────────


def preprocess(
    path: str | Path,
    *,
    filter_session: bool = True,
    output_path: str | Path | None = None,
    drop_invalid: bool = False,
    naive_tz_policy: str = "reject",
    early_close_schedule: dict[str, str] | None = None,
    partial_dataset: bool = False,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Run the complete preprocessing pipeline on a raw Alpaca CSV.

    Parameters
    ----------
    path : str or Path
        Path to raw CSV.
    filter_session : bool
        Whether to filter to RTH candles (default True).
    output_path : str, Path, or None
        Path to save clean CSV. Saved ONLY if validation passes.
    drop_invalid : bool
        If False (default), invalid rows (NaT, nulls, invalid OHLCV, conflicts)
        cause validation failure without saving.
        If True, invalid rows are removed, documented in the report, and the
        cleaned dataset is saved if valid.
    naive_tz_policy : str
        Policy for naive timestamps: "reject" (default), "assume_utc", or tz name.
    early_close_schedule : dict, optional
        Map of date string -> early close time (ET).
    partial_dataset : bool
        If True, gap detection handles partial intraday session boundaries.

    Returns
    -------
    tuple[pd.DataFrame, ValidationReport]
    """
    combined = ValidationReport()
    combined.info["calendar_limitation_note"] = (
        "Uses static early-close schedule (DEFAULT_US_EARLY_CLOSES). "
        "An exchange calendar package or API is recommended for production."
    )

    # 1. Load
    logger.info("Loading raw data from %s", path)
    try:
        df = load_raw_csv(path, naive_tz_policy=naive_tz_policy)
    except Exception as exc:
        combined.is_valid = False
        combined.errors.append(f"Failed to load CSV: {exc}")
        return pd.DataFrame(), combined

    combined.info["raw_rows"] = len(df)
    if df.empty:
        combined.is_valid = False
        combined.errors.append("Dataset is empty (0 rows).")
        return df, combined

    # 2. Validate columns
    col_report = validate_columns(df)
    _merge_report(combined, col_report)
    if not col_report.is_valid:
        return df, combined

    # 3. Initial validations
    ts_report = validate_timestamps(df)
    _merge_report(combined, ts_report)

    null_report = validate_no_missing_values(df)
    _merge_report(combined, null_report)

    ohlcv_report = validate_ohlcv(df)
    _merge_report(combined, ohlcv_report)

    dup_report = validate_no_duplicates(df)
    _merge_report(combined, dup_report)

    # 4. Handle invalid rows according to policy
    if not combined.is_valid:
        if not drop_invalid:
            combined.info["policy"] = "reject_on_invalid_data"
            logger.error("Preprocessing failed due to validation errors. Dataset rejected without modification.")
            return df, combined
        else:
            logger.warning("Cleaning invalid rows per drop_invalid=True policy.")
            combined.info["policy"] = "drop_invalid_rows"
            df, removal_info = _drop_invalid_rows(df)
            combined.info["removed_records"] = removal_info
            combined.warnings.append(
                f"Removed invalid records: {removal_info['total_removed']} rows dropped."
            )

            # Re-validate cleaned data
            combined.errors.clear()
            combined.is_valid = True

            if df.empty:
                combined.is_valid = False
                combined.errors.append("All rows were removed during cleaning. Result is empty.")
                return df, combined

            re_col = validate_columns(df)
            re_ts = validate_timestamps(df)
            re_null = validate_no_missing_values(df)
            re_ohlcv = validate_ohlcv(df)
            re_dup = validate_no_duplicates(df)

            for rep in (re_col, re_ts, re_null, re_ohlcv, re_dup):
                _merge_report(combined, rep)

            if not combined.is_valid:
                return df, combined

    # 5. Sort chronologically
    df = sort_by_timestamp(df)

    # 6. Deduplicate identical timestamps if any remain
    df, n_dups = remove_duplicates(df, reject_conflicts=True)
    combined.info["duplicates_removed"] = n_dups

    # 7. Session filtering (RTH)
    n_rth_removed = 0
    if filter_session:
        df, n_rth_removed = filter_rth(df, early_close_schedule=early_close_schedule)
    combined.info["rth_candles_removed"] = n_rth_removed
    combined.info["rows_after_cleaning"] = len(df)

    if df.empty:
        combined.is_valid = False
        combined.errors.append("No candles remain after session filtering.")
        return df, combined

    # 8. Gap detection (informational reporting)
    gaps = detect_gaps(
        df,
        rth_only=True,
        early_close_schedule=early_close_schedule,
        partial_dataset=partial_dataset,
    )
    combined.info["rth_gaps_detected"] = len(gaps)
    if gaps:
        combined.warnings.append(
            f"{len(gaps)} missing RTH candle(s) detected. No candles fabricated."
        )

    # 9. Save only if validation succeeded
    if output_path is not None:
        if combined.is_valid and not df.empty:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info("Saved cleaned data (%d rows) to %s", len(df), output_path)
            combined.info["output_path"] = str(output_path)
        else:
            logger.error(
                "Output file %s NOT saved because dataset validation failed (is_valid=%s).",
                output_path,
                combined.is_valid,
            )

    _log_summary(combined)
    return df, combined


def _drop_invalid_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean invalid rows from df and account for each removal reason."""
    n_init = len(df)

    # 1. Invalid timestamps (NaT)
    valid_ts = df["timestamp"].notna()
    n_nat = int((~valid_ts).sum())
    df = df[valid_ts]

    # 2. Conflicting duplicates (reject all conflicting rows)
    dup_ts = df["timestamp"].duplicated(keep=False)
    if dup_ts.any():
        unique_dups = df[dup_ts].drop_duplicates()
        ts_conflict = unique_dups["timestamp"].value_counts()
        conflicting = ts_conflict[ts_conflict > 1].index
        is_conflict = df["timestamp"].isin(conflicting)
        n_conflict = int(is_conflict.sum())
        df = df[~is_conflict]
    else:
        n_conflict = 0

    # 3. Identical duplicate timestamps (keep single first occurrence)
    n_before_dedup = len(df)
    df = df.drop_duplicates(subset=["timestamp"], keep="first")
    n_identical_dups = n_before_dedup - len(df)

    # 4. Missing values in required columns
    has_null = df[REQUIRED_COLUMNS].isna().any(axis=1)
    n_null = int(has_null.sum())
    df = df[~has_null]

    # 5. Non-numeric or non-finite OHLCV
    invalid_ohlcv_mask = pd.Series(False, index=df.index)
    for col in NUMERIC_OHLCV_COLUMNS:
        s_num = pd.to_numeric(df[col], errors="coerce")
        nan_or_bad = s_num.isna()
        inf_bad = np.isposinf(s_num) | np.isneginf(s_num)
        invalid_ohlcv_mask |= nan_or_bad | inf_bad

    n_non_finite = int(invalid_ohlcv_mask.sum())
    df = df[~invalid_ohlcv_mask]

    # Ensure numeric dtypes
    for col in NUMERIC_OHLCV_COLUMNS:
        df[col] = df[col].astype(float)

    # 6. Logical OHLC consistency
    logic_bad = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
        | (df["volume"] < 0)
    )
    n_logic = int(logic_bad.sum())
    df = df[~logic_bad]

    total_removed = n_init - len(df)
    removal_info = {
        "n_nat_removed": n_nat,
        "n_conflict_removed": n_conflict,
        "n_identical_dups_removed": n_identical_dups,
        "n_null_removed": n_null,
        "n_non_finite_removed": n_non_finite,
        "n_logic_removed": n_logic,
        "total_removed": total_removed,
    }
    return df.reset_index(drop=True), removal_info


def _merge_report(target: ValidationReport, source: ValidationReport) -> None:
    """Merge *source* into *target*, accumulating errors/warnings/info."""
    if not source.is_valid:
        target.is_valid = False
    target.errors.extend(source.errors)
    target.warnings.extend(source.warnings)
    target.info.update(source.info)


def _log_summary(report: ValidationReport) -> None:
    """Log a concise summary of the combined report."""
    status = "PASS" if report.is_valid else "FAIL"
    logger.info(
        "Preprocessing %s — %d error(s), %d warning(s).",
        status,
        len(report.errors),
        len(report.warnings),
    )
    for e in report.errors:
        logger.error("  ERROR: %s", e)
    for w in report.warnings:
        logger.warning("  WARN: %s", w)
