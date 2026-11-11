from datetime import date, datetime, timezone

import pandas as pd
import pytest

from src.forecasts.calendar import cutoff_for_evaluation, is_trading_day, previous_trading_day
from src.forecasts.levels import support_hold_probability
from src.forecasts.service import issue_window_forecast
from src.forecasts.store import issue_forecast, list_forecasts, record_actual
from src.forecasts import store as forecast_store
from src.forecasts.service import compare_completed_forecast


def test_previous_trading_day_skips_weekends_and_holidays():
    assert previous_trading_day(date(2026, 11, 9)) == date(2026, 11, 6)
    assert previous_trading_day(date(2026, 11, 27)) == date(2026, 11, 25)
    assert not is_trading_day(date(2026, 11, 26))
    assert cutoff_for_evaluation(date(2026, 11, 10)).isoformat() == "2026-11-09T21:00:00+00:00"
    assert cutoff_for_evaluation(date(2026, 11, 2)).isoformat() == "2026-10-30T20:00:00+00:00"


def test_support_probability_requires_samples_and_is_bounded():
    too_few = support_hold_probability([], 1.0, 2.0, minimum_samples=3)
    assert too_few["probability"] is None
    assert too_few["status"] == "insufficient_calibration_samples"
    samples = [{"break": i < 10, "distance_atr": 0.5} for i in range(30)]
    result = support_hold_probability(samples, 1.0, 2.0, minimum_samples=30)
    assert 0.0 < result["probability"] < 1.0
    assert result["sample_count"] == 30


def test_issued_forecast_and_actual_result_are_write_once(tmp_path):
    db = tmp_path / "forecast.sqlite"
    record = {"forecast_id": "f1", "evaluation_date": "2026-11-10", "window_id": "morning",
              "generated_at": datetime.now(timezone.utc).isoformat()}
    issue_forecast(record, db)
    with pytest.raises(Exception):
        issue_forecast(record, db)
    record_actual("f1", {"direction_correct": True}, db)
    with pytest.raises(ValueError, match="already recorded"):
        record_actual("f1", {"direction_correct": False}, db)
    assert list_forecasts("2026-11-10", db)[0]["actual"]["direction_correct"] is True


def test_forecast_generation_is_blocked_at_window_start():
    candles = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "vwap"])
    with pytest.raises(ValueError, match="strictly before"):
        issue_window_forecast("2026-11-10", "morning", candles,
                              datetime(2026, 11, 10, 14, 30, tzinfo=timezone.utc))


def test_actual_comparison_is_blocked_before_window_end(tmp_path, monkeypatch):
    monkeypatch.setattr(forecast_store, "DB_PATH", tmp_path / "forecast.sqlite")
    payload = {"forecast_id": "f2", "evaluation_date": "2026-10-27", "window_id": "morning",
               "generated_at": "2026-10-27T13:29:00+00:00",
               "window_start": "2026-10-27T09:30:00-04:00", "window_end": "2026-10-27T12:30:00-04:00"}
    issue_forecast(payload)
    with pytest.raises(ValueError, match="before the forecast window has ended"):
        compare_completed_forecast("f2", pd.DataFrame(), datetime(2026, 10, 27, 16, 29, tzinfo=timezone.utc))
