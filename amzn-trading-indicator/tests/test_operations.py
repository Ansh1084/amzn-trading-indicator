from __future__ import annotations

import threading
import time

import pandas as pd
import pytest

import src.forecasts.operations as operations
from src.data.ingestion import _merge_candles


def test_incremental_merge_deduplicates_and_sorts_chronologically():
    existing = pd.DataFrame({"timestamp": ["2026-11-06T14:35:00Z", "2026-11-06T14:30:00Z"],
                             "close": [2, 1]})
    fetched = pd.DataFrame({"timestamp": ["2026-11-06T14:35:00Z", "2026-11-06T14:40:00Z"],
                            "close": [3, 4]})
    merged, duplicates, added = _merge_candles(existing, fetched)
    assert duplicates == 1
    assert added == 1
    assert merged.timestamp.is_monotonic_increasing
    assert len(merged) == 3
    assert merged.iloc[1].close == 3


def test_data_refresh_rejects_duplicate_concurrent_jobs(monkeypatch, tmp_path):
    monkeypatch.setattr(operations, "DB_PATH", tmp_path / "ops.sqlite3")
    monkeypatch.setattr(operations, "_INITIALIZED", False)
    monkeypatch.setattr(operations, "_LOCKS", {"data_refresh": threading.Lock(), "model_training": threading.Lock()})
    entered, release = threading.Event(), threading.Event()

    def work(progress):
        entered.set()
        release.wait(3)
        progress("saved", .8)
        return {"added": 4}

    started = operations.start_operation("data_refresh", None, work)
    assert entered.wait(1)
    with pytest.raises(RuntimeError, match="already running"):
        operations.start_operation("data_refresh", None, work)
    release.set()
    deadline = time.time() + 3
    result = None
    while time.time() < deadline:
        result = operations.get_operation(started["operation_id"])
        if result and result["status"] == "COMPLETED":
            break
        time.sleep(.02)
    assert result["status"] == "COMPLETED"
    assert result["result"] == {"added": 4}
