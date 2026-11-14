"""Completed 5-minute market data from Alpaca IEX or a local historical CSV."""
from __future__ import annotations

import os
import argparse
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from dotenv import load_dotenv


_ALPACA_IMPORT_LOCK = threading.Lock()
_ALPACA_SDK = None


def _import_alpaca_sdk():
    """Import all SDK modules used by this app in one deterministic sequence."""
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.live import StockDataStream
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    return SimpleNamespace(
        DataFeed=DataFeed,
        StockHistoricalDataClient=StockHistoricalDataClient,
        StockDataStream=StockDataStream,
        StockBarsRequest=StockBarsRequest,
        TimeFrame=TimeFrame,
        TimeFrameUnit=TimeFrameUnit,
    )


def initialize_alpaca_sdk():
    """Serialize Alpaca's first-time imports across stream and REST worker threads.

    The lock protects import initialization only. Client construction and network
    requests happen after it is released.
    """
    global _ALPACA_SDK
    if _ALPACA_SDK is not None:
        return _ALPACA_SDK
    with _ALPACA_IMPORT_LOCK:
        if _ALPACA_SDK is None:
            _ALPACA_SDK = _import_alpaca_sdk()
    return _ALPACA_SDK


def fetch_alpaca_candles(lookback_days: int = 35, start: datetime | None = None,
                         end: datetime | None = None, allow_empty: bool = False) -> pd.DataFrame:
    """Fetch recent IEX bars and exclude the currently forming 5-minute candle."""
    load_dotenv()
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Alpaca credentials are not configured in the backend environment")
    try:
        sdk = initialize_alpaca_sdk()
    except ImportError as exc:
        raise RuntimeError("Install alpaca-py to retrieve Alpaca market data") from exc
    now = end or datetime.now(timezone.utc)
    request = sdk.StockBarsRequest(symbol_or_symbols=["AMZN"], timeframe=sdk.TimeFrame(5, sdk.TimeFrameUnit.Minute),
                               start=start or now - timedelta(days=lookback_days), end=now,
                               limit=10000, feed=sdk.DataFeed.IEX)
    bars = sdk.StockHistoricalDataClient(key, secret).get_stock_bars(request).df
    if bars.empty and not allow_empty:
        raise RuntimeError("Alpaca returned no AMZN IEX candles")
    if bars.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = bars.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    # Alpaca timestamps are candle starts; only use bars whose five minutes have elapsed.
    df = df.loc[df["timestamp"] + pd.Timedelta(minutes=5) <= pd.Timestamp(now)].copy()
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    from src.data.preprocessing import filter_rth
    df, _ = filter_rth(df)
    if df.empty:
        if allow_empty:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        raise RuntimeError("No completed 5-minute AMZN candles returned by Alpaca")
    return df


def latest_processed_path() -> Path:
    """Select the newest processed snapshot without preferring an old canonical file."""
    root = Path(__file__).resolve().parents[2] / "data" / "processed"
    candidates = list(root.glob("AMZN_clean*.csv"))
    if not candidates:
        raise FileNotFoundError("No processed AMZN dataset is available")
    return max(candidates, key=lambda p: (p.stat().st_mtime_ns, p.name))


def dataset_status(chart_limit: int = 1200) -> dict:
    path = latest_processed_path()
    frame = load_local_candles(path)
    if frame.empty:
        raise ValueError(f"Processed dataset {path.name} is empty")
    ts = pd.to_datetime(frame.timestamp, utc=True)
    raw_root = path.parents[1] / "raw"
    raw_candidates = list(raw_root.glob("AMZN_data_1year*.csv"))
    last_sync = None
    try:
        from src.forecasts.operations import list_operations
        last_sync = next((op["result"].get("last_successful_sync") for op in list_operations("data_refresh")
                          if op["status"] == "COMPLETED" and op.get("result")), None)
    except Exception:
        pass
    return {"source": "LOCAL SNAPSHOT", "feed": "Alpaca IEX historical snapshot", "path": str(path),
            "rows": len(frame), "chart_rows": min(chart_limit, len(frame)), "chart_limit": chart_limit,
            "dataset_start": ts.min().isoformat(), "dataset_end": ts.max().isoformat(),
            "latest_completed_rth_candle": ts.max().isoformat(),
            "snapshot_created_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            "last_sync_at": last_sync,
            "raw_path": str(max(raw_candidates, key=lambda p: p.stat().st_mtime_ns)) if raw_candidates else None}


def refresh_latest_market_data(progress=None) -> dict:
    """Incrementally merge completed IEX bars and write new immutable snapshots."""
    project = Path(__file__).resolve().parents[2]
    raw_root, clean_root = project / "data" / "raw", project / "data" / "processed"
    raw_candidates = list(raw_root.glob("AMZN_data_1year*.csv")) if raw_root.exists() else []
    if raw_candidates:
        raw_path = max(raw_candidates, key=lambda p: p.stat().st_mtime_ns)
        existing = pd.read_csv(raw_path)
        existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True, errors="raise")
        since = existing.timestamp.max().to_pydatetime() + timedelta(minutes=5)
    else:
        raw_path = None
        existing = pd.DataFrame()
        since = datetime.now(timezone.utc) - timedelta(days=400)
    if progress: progress("fetching_alpaca_iex", .15)
    fetched = fetch_alpaca_candles(start=since, allow_empty=True)
    fetched_count = len(fetched)
    if fetched.empty and existing.empty:
        raise RuntimeError("No completed regular-session candles were returned and no local history exists")
    if progress: progress("validating_and_merging", .45)
    merged, duplicate_count, added = _merge_candles(existing, fetched)
    # Validate required numeric values and OHLC logic before persisting either snapshot.
    numeric = ["open", "high", "low", "close", "volume"]
    for col in numeric:
        merged[col] = pd.to_numeric(merged[col], errors="raise")
    invalid = (merged[numeric].isna().any(axis=1) | (merged.volume < 0) | (merged.high < merged.low) |
               (merged.high < merged[["open", "close"]].max(axis=1)) |
               (merged.low > merged[["open", "close"]].min(axis=1)))
    if invalid.any():
        raise ValueError(f"Merged market data contains {int(invalid.sum())} invalid OHLCV row(s)")
    stamp = datetime.now(timezone.utc).astimezone(__import__("zoneinfo").ZoneInfo("America/New_York")).strftime("%Y%m%dT%H%M%S")
    new_raw = raw_root / f"AMZN_data_1year_{stamp}.csv"
    new_clean = clean_root / f"AMZN_clean_{stamp}.csv"
    raw_root.mkdir(parents=True, exist_ok=True); clean_root.mkdir(parents=True, exist_ok=True)
    if progress: progress("processing_and_saving_snapshots", .7)
    merged.to_csv(new_raw, index=False)
    from src.data.preprocessing import preprocess
    cleaned, report = preprocess(new_raw, filter_session=True, output_path=new_clean,
                                 naive_tz_policy="reject", partial_dataset=True)
    if not report.is_valid:
        new_raw.unlink(missing_ok=True)
        raise RuntimeError(f"Refreshed data failed validation: {report.errors}")
    if progress: progress("feature_generation", .83)
    from src.features.engineering import generate_features, get_feature_columns
    featured = generate_features(cleaned, reset_daily=False)
    if progress: progress("refresh_complete", .95)
    ts = merged.timestamp
    return {"source": "Alpaca IEX", "fetched": fetched_count, "added": max(0, added),
            "duplicates_removed": duplicate_count, "rows": len(merged),
            "earliest_timestamp": ts.min().isoformat(), "latest_timestamp": ts.max().isoformat(),
            "sync_completed_at": datetime.now(timezone.utc).isoformat(), "raw_path": str(new_raw),
            "processed_path": str(new_clean), "validation": report.info,
            "feature_rows": len(featured), "feature_count": len(get_feature_columns()),
            "latest_feature_timestamp": pd.to_datetime(featured.timestamp, utc=True).max().isoformat(),
            "warnings": report.warnings, "last_successful_sync": datetime.now(timezone.utc).isoformat()}


def load_local_candles(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df:
        raise ValueError("Local candle file has no timestamp column")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def _merge_candles(existing: pd.DataFrame, fetched: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Chronologically merge bars, preferring the newest response for duplicate starts."""
    combined = pd.concat([existing, fetched], ignore_index=True, sort=False)
    if combined.empty:
        return combined, 0, 0
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True, errors="raise")
    before = len(combined)
    merged = combined.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    return merged, before-len(merged), max(0, len(merged)-len(existing))


def refresh_training_history(start: datetime | None = None, end: datetime | None = None) -> tuple[Path, Path]:
    """Download a year of historical IEX bars in bounded chunks and preprocess them."""
    load_dotenv()
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Alpaca credentials are not configured in the backend environment")
    try:
        sdk = initialize_alpaca_sdk()
    except ImportError as exc:
        raise RuntimeError("Install alpaca-py to retrieve Alpaca market data") from exc
    end = end or datetime.now(timezone.utc)
    start = start or end - timedelta(days=400)
    client = sdk.StockHistoricalDataClient(key, secret)
    chunks, cursor = [], start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=30), end)
        bars = client.get_stock_bars(sdk.StockBarsRequest(
            symbol_or_symbols=["AMZN"], timeframe=sdk.TimeFrame(5, sdk.TimeFrameUnit.Minute),
            start=cursor, end=chunk_end, limit=10000, feed=sdk.DataFeed.IEX,
        )).df
        if not bars.empty:
            chunks.append(bars.reset_index())
        cursor = chunk_end
    if not chunks:
        raise RuntimeError("Alpaca returned no historical AMZN bars")
    raw = pd.concat(chunks, ignore_index=True).drop_duplicates(["symbol", "timestamp"]).sort_values("timestamp")
    project = Path(__file__).resolve().parents[2]
    snapshot = pd.Timestamp(end).tz_convert("America/New_York").strftime("%Y%m%dT%H%M%S")
    raw_path = project / "data" / "raw" / f"AMZN_data_1year_{snapshot}.csv"
    clean_path = project / "data" / "processed" / f"AMZN_clean_{snapshot}.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(raw_path, index=False)
    from src.data.preprocessing import preprocess
    _, report = preprocess(raw_path, filter_session=True, output_path=clean_path,
                           naive_tz_policy="reject", partial_dataset=True)
    if not report.is_valid:
        raise RuntimeError(f"Refreshed raw data failed preprocessing: {report.errors}")
    return raw_path, clean_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refresh AMZN IEX historical training data")
    parser.add_argument("--days", type=int, default=400)
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    print(refresh_training_history(now - timedelta(days=args.days), now))
