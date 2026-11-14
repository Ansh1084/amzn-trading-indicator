import asyncio
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pandas as pd
import pytest

from backend.app import main
from backend.app.services import market_store, market_stream
from src.data import ingestion


def test_alpaca_sdk_initialization_is_serialized_across_threads(monkeypatch):
    monkeypatch.setattr(ingestion, "_ALPACA_SDK", None)
    active = 0
    peak = 0
    calls = 0
    guard = threading.Lock()
    sdk = object()

    def load_sdk():
        nonlocal active, peak, calls
        with guard:
            calls += 1
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with guard:
            active -= 1
        return sdk

    monkeypatch.setattr(ingestion, "_import_alpaca_sdk", load_sdk)
    with ThreadPoolExecutor(max_workers=2) as pool:
        left = pool.submit(ingestion.initialize_alpaca_sdk)
        right = pool.submit(ingestion.initialize_alpaca_sdk)
        assert left.result(timeout=3) is sdk
        assert right.result(timeout=3) is sdk
    assert calls == 1
    assert peak == 1


def test_historical_fetch_and_stream_initializer_share_the_serialized_loader(monkeypatch):
    monkeypatch.setattr(ingestion, "_ALPACA_SDK", None)
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")
    calls = 0
    guard = threading.Lock()

    class HistoricalClient:
        def __init__(self, *_args):
            pass

        def get_stock_bars(self, _request):
            return SimpleNamespace(df=pd.DataFrame())

    sdk = SimpleNamespace(
        DataFeed=SimpleNamespace(IEX="iex"),
        StockHistoricalDataClient=HistoricalClient,
        StockDataStream=object,
        StockBarsRequest=lambda **kwargs: kwargs,
        TimeFrame=lambda *_args: "5Min",
        TimeFrameUnit=SimpleNamespace(Minute="minute"),
    )

    def load_sdk():
        nonlocal calls
        with guard:
            calls += 1
        time.sleep(0.05)
        return sdk

    monkeypatch.setattr(ingestion, "_import_alpaca_sdk", load_sdk)
    now = pd.Timestamp.now(tz="UTC").to_pydatetime()
    with ThreadPoolExecutor(max_workers=2) as pool:
        stream_init = pool.submit(ingestion.initialize_alpaca_sdk)
        historical_fetch = pool.submit(
            ingestion.fetch_alpaca_candles,
            5,
            now - pd.Timedelta(minutes=10),
            now,
            True,
        )
        assert stream_init.result(timeout=3) is sdk
        assert historical_fetch.result(timeout=3).empty
    assert calls == 1


class _FakeFeed:
    IEX = "iex"


def _fake_sdk(service, auth_error=None):
    created = []

    class FakeStream:
        def __init__(self, *_args, **_kwargs):
            created.append(self)

        async def _auth(self, *_args, **_kwargs):
            if auth_error:
                raise auth_error

        async def _wait_before_reconnect(self, *_args, **_kwargs):
            pass

        def subscribe_bars(self, *_args):
            pass

        def run(self):
            # The fake auth coroutine completes synchronously. Drive it directly
            # to avoid opening a Windows event loop or any network connection.
            auth = self._auth()
            try:
                auth.send(None)
            except StopIteration:
                pass
            service._stop.set()

        def stop(self):
            pass

    return SimpleNamespace(DataFeed=_FakeFeed, StockDataStream=FakeStream), created


def _capture_status(monkeypatch, initial="CONNECTING"):
    current = {"connection_state": initial}
    written = []

    def write(values, **_kwargs):
        current.update(values)
        written.append(dict(current))
        return current

    monkeypatch.setattr(market_store, "write_status", write)
    monkeypatch.setattr(market_store, "get_status", lambda: dict(current))
    return written


def test_transient_sdk_init_failure_retries_then_resets_count_on_authenticated_connect(monkeypatch):
    service = market_stream.MarketStreamService()
    written = _capture_status(monkeypatch)
    sdk, created = _fake_sdk(service)
    attempts = 0

    async def no_op_reconcile():
        return {"synchronized": True}

    service.reconcile = no_op_reconcile

    def initialize():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary initialization failure")
        return sdk

    delays = []
    monkeypatch.setattr(market_stream, "initialize_alpaca_sdk", initialize)
    monkeypatch.setattr(service, "_wait_before_retry", lambda delay: delays.append(delay) or False)
    service._run_stream("key", "secret")
    assert attempts == 2 and len(created) == 1
    assert delays == [market_stream.STREAM_RETRY_BASE_SECONDS]
    assert any(row["connection_state"] == "RECONNECTING" and row["retry_count"] == 1 for row in written)
    assert any(row["connection_state"] == "CONNECTED" and row["retry_count"] == 0 for row in written)
    assert written[-1]["connection_state"] == "DISCONNECTED"


def test_invalid_credentials_stop_without_retrying(monkeypatch):
    service = market_stream.MarketStreamService()
    written = _capture_status(monkeypatch)
    sdk, created = _fake_sdk(service, RuntimeError("401 unauthorized"))
    attempts = []
    delays = []
    monkeypatch.setattr(market_stream, "initialize_alpaca_sdk", lambda: attempts.append(True) or sdk)
    monkeypatch.setattr(service, "_wait_before_retry", lambda delay: delays.append(delay) or False)
    service._run_stream("key", "secret")
    assert attempts == [True] and len(created) == 1
    assert delays == []
    assert written[-1]["connection_state"] == "ERROR"
    assert "401 unauthorized" in written[-1]["last_error"]


def test_shutdown_interrupts_stream_retry_wait(monkeypatch):
    service = market_stream.MarketStreamService()
    written = _capture_status(monkeypatch)
    attempts = []
    def fail_init():
        attempts.append(True)
        raise RuntimeError("temporary import initialization failure")
    monkeypatch.setattr(market_stream, "initialize_alpaca_sdk", fail_init)
    def stop_during_wait(_delay):
        service._stop.set()
        return True
    monkeypatch.setattr(service, "_wait_before_retry", stop_during_wait)
    service._run_stream("key", "secret")
    assert len(attempts) == 1
    assert written[-1]["connection_state"] == "DISCONNECTED"


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Python 3.14 event-loop self-pipe initialization hangs in this Windows runner",
)
def test_overlapping_reconciliation_callers_share_one_operation():
    service = market_stream.MarketStreamService()
    calls = []

    async def fake_once():
        calls.append(True)
        await asyncio.sleep(0.02)
        return {"synchronized": True, "bars_fetched": 0}

    service._reconcile_once = fake_once
    async def run():
        return await asyncio.gather(service.reconcile(), service.reconcile())

    results = asyncio.run(run())
    assert len(calls) == 1
    assert results[0] == results[1] == {"synchronized": True, "bars_fetched": 0}


def test_periodic_reconciliation_loop_survives_unexpected_failure(monkeypatch):
    calls = []

    async def reconcile():
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("one transient failure")

    async def stop_after_second_sleep(_seconds):
        if len(calls) >= 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(main.market_stream, "reconcile", reconcile)
    monkeypatch.setattr(main.asyncio, "sleep", stop_after_second_sleep)
    loop = main._reconcile_loop()
    with pytest.raises(asyncio.CancelledError):
        loop.send(None)
    assert len(calls) == 2
