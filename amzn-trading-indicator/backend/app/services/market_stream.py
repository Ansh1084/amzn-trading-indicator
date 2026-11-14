"""Alpaca IEX one-minute stream with provider-confirmed 5-minute reconciliation."""
from __future__ import annotations

import asyncio
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

from backend.app.services import market_store
from backend.app.services.analysis.config import ANALYSIS_VERSION
from backend.app.services.analysis.persistence import get_signal_for_candle
from src.data.ingestion import (fetch_alpaca_candles, initialize_alpaca_sdk,
                                latest_processed_path, load_local_candles)

logger = logging.getLogger(__name__)
STREAM_RETRY_BASE_SECONDS = 1.0
STREAM_RETRY_MAX_SECONDS = 30.0
AUTH_ERROR_TERMS = ("unauthorized", "authentication", "forbidden", "invalid key",
                    "invalid secret", "401", "403", "insufficient subscription")


def _is_auth_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(term in message for term in AUTH_ERROR_TERMS)


def _is_permanent_sdk_error(exc: Exception) -> bool:
    if type(exc).__name__ == "DeadlockError":
        return False
    return isinstance(exc, ImportError)


def _bar_dict(bar, source: str) -> dict:
    stamp = pd.Timestamp(bar.timestamp).tz_convert("UTC").isoformat()
    return {"timestamp": stamp, "open": float(bar.open), "high": float(bar.high),
            "low": float(bar.low), "close": float(bar.close), "volume": float(bar.volume),
            "symbol": getattr(bar, "symbol", "AMZN"), "vwap": getattr(bar, "vwap", None),
            "data_source": source, "provider_timestamp": stamp}


def _safe_error(exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}"
    for name in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"):
        secret = os.getenv(name)
        if secret:
            message = message.replace(secret, "[redacted]")
    return message[:500]


class MarketStreamService:
    """Owns one Alpaca connection; historical 5-minute bars are authoritative finals."""
    def __init__(self):
        self._thread: threading.Thread | None = None
        self._stream = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._reconcile_state_lock = threading.Lock()
        self._reconcile_task: asyncio.Task | None = None
        self._fatal_stream_error = threading.Event()
        self._retries = 0
        self._project = Path(__file__).resolve().parents[3]

    def start(self) -> bool:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return True
            load_dotenv(self._project / ".env")
            key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
            self._seed_snapshot()
            if not key or not secret:
                market_store.write_status({"connection_state": "DISCONNECTED", "last_error": "Alpaca credentials are not configured"})
                return False
            self._stop.clear()
            self._fatal_stream_error.clear()
            self._retries = 0
            self._thread = threading.Thread(target=self._run_stream, args=(key, secret), daemon=True, name="amzn-alpaca-iex")
            self._thread.start()
            return True

    def stop(self) -> None:
        self._stop.set()
        stream = self._stream
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def _seed_snapshot(self) -> None:
        try:
            frame = load_local_candles(latest_processed_path())
            market_store.seed_completed_candles(frame)
            # Snapshot loading is a seed only; it does not claim a live connection.
        except FileNotFoundError:
            pass

    def _run_stream(self, key: str, secret: str) -> None:
        owner = self
        while not self._stop.is_set() and not self._fatal_stream_error.is_set():
            stream = None
            try:
                market_store.write_status({"connection_state": "CONNECTING"})
                # This shared loader serializes first-time live and historical SDK imports.
                sdk = initialize_alpaca_sdk()
                StockDataStream, DataFeed = sdk.StockDataStream, sdk.DataFeed

                class StatusStream(StockDataStream):
                    async def _auth(self, *args, **kwargs):
                        try:
                            await super()._auth(*args, **kwargs)
                        except Exception as exc:
                            if _is_auth_error(exc):
                                owner._fatal_stream_error.set()
                                market_store.write_status({"connection_state": "ERROR", "last_error": _safe_error(exc),
                                                           "retry_count": owner._retries})
                                self.stop()
                            else:
                                market_store.write_status({"connection_state": "RECONNECTING",
                                                           "last_error": _safe_error(exc),
                                                           "retry_count": owner._retries})
                            raise
                        owner._retries = 0
                        market_store.write_status({"connection_state": "CONNECTED", "last_error": None,
                                                   "retry_count": 0})

                    async def _wait_before_reconnect(self, retries, run):
                        owner._retries = max(owner._retries, int(retries))
                        market_store.write_status({"connection_state": "RECONNECTING",
                                                   "retry_count": owner._retries})
                        await super()._wait_before_reconnect(retries, run)

                stream = StatusStream(key, secret, feed=DataFeed.IEX)
                self._stream = stream

                async def on_minute(bar):
                    stamp = pd.Timestamp(bar.timestamp).tz_convert("UTC").to_pydatetime()
                    local = stamp.astimezone(ZoneInfo("America/New_York"))
                    from src.forecasts.calendar import is_trading_day, session_close
                    if not is_trading_day(local.date()) or (local.hour, local.minute) < (9, 30):
                        return
                    close = session_close(local.date())
                    bucket_minute = (local.minute // 5) * 5
                    if (local.hour, bucket_minute + 5) > (close.hour, close.minute):
                        return
                    now = datetime.now(timezone.utc).isoformat()
                    market_store.write_status({"last_received_event_at": now, "connection_state": "CONNECTED"}, emit=False)
                    market_store.upsert_minute_bar(_bar_dict(bar, "alpaca_iex_minute_aggregation"))

                stream.subscribe_bars(on_minute, "AMZN")
                stream.run()
                if self._stop.is_set() or self._fatal_stream_error.is_set():
                    break
                raise RuntimeError("Alpaca IEX stream stopped unexpectedly")
            except Exception as exc:
                if self._stop.is_set():
                    break
                permanent = _is_auth_error(exc) or _is_permanent_sdk_error(exc)
                error = _safe_error(exc)
                logger.exception("AMZN Alpaca IEX stream %s: %s",
                                 "initialization/connection failed" if not permanent else "cannot start",
                                 error)
                if permanent or self._fatal_stream_error.is_set():
                    market_store.write_status({"connection_state": "ERROR", "last_error": error,
                                               "retry_count": self._retries})
                    self._fatal_stream_error.set()
                    break
                self._retries += 1
                delay = min(STREAM_RETRY_BASE_SECONDS * (2 ** (self._retries - 1)), STREAM_RETRY_MAX_SECONDS)
                market_store.write_status({"connection_state": "RECONNECTING", "last_error": error,
                                           "retry_count": self._retries})
                self._wait_before_retry(delay)
            finally:
                if stream is not None:
                    try:
                        stream.stop()
                    except Exception:
                        logger.debug("Error while stopping failed Alpaca stream", exc_info=True)
                self._stream = None
        if self._stop.is_set():
            market_store.write_status({"connection_state": "DISCONNECTED", "last_error": None,
                                       "retry_count": self._retries})

    def _wait_before_retry(self, delay: float) -> bool:
        """Wait interruptibly; returns True when shutdown was requested."""
        return self._stop.wait(delay)

    async def reconcile(self) -> dict:
        """Coalesce concurrent callers without holding a lock over network I/O."""
        loop = asyncio.get_running_loop()
        with self._reconcile_state_lock:
            task = self._reconcile_task
            if task is not None and not task.done():
                if task.get_loop() is not loop:
                    return {"synchronized": False, "skipped": "reconciliation is running on another event loop"}
            else:
                task = loop.create_task(self._reconcile_once(), name="amzn-market-reconciliation")
                self._reconcile_task = task
        try:
            return await asyncio.shield(task)
        finally:
            if task.done():
                with self._reconcile_state_lock:
                    if self._reconcile_task is task:
                        self._reconcile_task = None

    async def _reconcile_once(self) -> dict:
        """Backfill the recent overlap; provider-native completed 5m bars win corrections."""
        try:
            latest = market_store.get_candles(limit=1)
            latest_start = (pd.Timestamp(latest[-1]["timestamp"]).to_pydatetime() - timedelta(minutes=5)) if latest else datetime.now(timezone.utc) - timedelta(days=5)
            oldest_vwap_gap = market_store.oldest_missing_vwap_timestamp()
            start = min(latest_start, pd.Timestamp(oldest_vwap_gap).to_pydatetime()) if oldest_vwap_gap else latest_start
            frame = await asyncio.to_thread(fetch_alpaca_candles, 5, start, datetime.now(timezone.utc), True)
            bars = []
            if not frame.empty:
                for row in frame.itertuples(index=False):
                    bars.append({"timestamp": row.timestamp, "open": row.open, "high": row.high, "low": row.low,
                                 "close": row.close, "volume": row.volume, "data_source": "alpaca_iex_5m",
                                 "symbol": getattr(row, "symbol", "AMZN"), "vwap": getattr(row, "vwap", None),
                                 "provider_timestamp": row.timestamp})
            events = market_store.upsert_completed_candles(bars)
            now = datetime.now(timezone.utc)
            market_store.write_status({"last_reconciliation_at": now.isoformat(), "reconciliation_error": None}, emit=False)
            market_store.publish_event("synchronization_completed", {"symbol": "AMZN", "bars_fetched": len(bars),
                "event_timestamp": now.isoformat(), "data_source": "alpaca_iex_5m"})
            # Also recover a process interruption after candle persistence but
            # before analysis. The per-candle/version persisted key prevents replay.
            latest = market_store.get_candles(limit=1)
            if latest:
                end = pd.Timestamp(latest[-1]["timestamp"]) + pd.Timedelta(minutes=5)
                if pd.Timestamp(now) - end <= pd.Timedelta(minutes=15):
                    await asyncio.to_thread(self._analyze_latest)
            return {"synchronized": True, "bars_fetched": len(bars), "events": len(events), "at": now.isoformat()}
        except Exception as exc:
            error = _safe_error(exc)
            market_store.write_status({"reconciliation_error": error}, emit=False)
            logger.warning("AMZN market reconciliation failed: %s", error)
            try:
                market_store.publish_event("reconciliation_failed", {"symbol": "AMZN", "error": error,
                    "event_timestamp": datetime.now(timezone.utc).isoformat(), "data_source": "alpaca_iex"})
            except Exception:
                logger.warning("Could not publish AMZN reconciliation failure event")
            return {"synchronized": False, "error": error}

    def _analyze_latest(self) -> None:
        with self._lock:
            rows = market_store.get_candles(limit=10000)
            if not rows:
                return
            latest_timestamp = rows[-1]["timestamp"]
            if get_signal_for_candle(latest_timestamp, ANALYSIS_VERSION):
                return
            try:
                from backend.app.services.analysis.service import analyze_candles
                result = analyze_candles(pd.DataFrame(rows), source="alpaca_iex_5m_reconciled")
            except Exception as exc:
                error = _safe_error(exc)
                logger.warning("AMZN analysis failed for completed candle %s: %s", latest_timestamp, error)
                try:
                    market_store.publish_event("analysis_failed", {"symbol": "AMZN", "candle_timestamp": latest_timestamp,
                        "error": error, "event_timestamp": datetime.now(timezone.utc).isoformat(), "data_source": "alpaca_iex_5m"})
                except Exception:
                    logger.warning("Could not publish AMZN analysis failure for candle %s", latest_timestamp)
                raise
            payload = result["analysis"]
            signal = result.get("signal_record")
            try:
                market_store.publish_event("analysis_updated", {"symbol": "AMZN", "analysis": payload,
                                           "event_timestamp": datetime.now(timezone.utc).isoformat(), "data_source": "alpaca_iex_5m"})
                if signal and signal.get("classification") in {"BUY", "SELL"}:
                    market_store.publish_event("signal_created", {"symbol": "AMZN", "signal": signal,
                                               "event_timestamp": datetime.now(timezone.utc).isoformat(), "data_source": "alpaca_iex_5m"})
            except Exception as exc:
                logger.warning("AMZN analysis persisted for %s but event publication failed: %s",
                               latest_timestamp, _safe_error(exc))


service = MarketStreamService()

