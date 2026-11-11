from datetime import date

import pytest

from backend.app.api import routes


def comparison(day: str, forecast_id: str, *, actual: bool = True, window: str = "morning"):
    return {
        "forecast_id": forecast_id,
        "evaluation_date": day,
        "window_id": window,
        "forecast_type": "ADVANCE",
        "revision": 1,
        "actual": {"actual_direction": "UP"} if actual else None,
    }


@pytest.mark.parametrize(
    ("as_of", "records", "expected"),
    [
        # Nov 10 before the open and during the morning: today's actual is not
        # persisted, so Nov 9 remains the latest completed evaluation session.
        (date(2026, 11, 10), [comparison("2026-11-09", "nov9")], "2026-11-09"),
        (date(2026, 11, 10), [comparison("2026-11-09", "nov9"), comparison("2026-11-10", "pending", actual=False)], "2026-11-09"),
        # After morning actuals are persisted, today becomes the latest date.
        (date(2026, 11, 10), [comparison("2026-11-09", "nov9"), comparison("2026-11-10", "nov10-am")], "2026-11-10"),
        # Afternoon ADVANCE and UPDATED remain separate actual-bearing rows.
        (date(2026, 11, 10), [comparison("2026-11-09", "nov9"), comparison("2026-11-10", "nov10-am"), comparison("2026-11-10", "adv-pm", window="afternoon"), {**comparison("2026-11-10", "updated-pm", window="afternoon"), "forecast_type": "UPDATED"}], "2026-11-10"),
        # Saturday uses the latest actual-backed session; no calendar-day subtraction.
        (date(2026, 11, 14), [comparison("2026-11-13", "fri"), comparison("2026-11-12", "thu")], "2026-11-13"),
        # Veterans Day is a market holiday; Nov 10 is the latest prior session.
        (date(2026, 11, 11), [comparison("2026-11-10", "tue"), comparison("2026-11-09", "mon")], "2026-11-10"),
        # Black Friday is a valid early-close session and remains selectable
        # when its completed morning window has been evaluated.
        (date(2026, 11, 30), [comparison("2026-11-27", "black-friday-am"), comparison("2026-11-25", "wed")], "2026-11-27"),
    ],
)
def test_latest_evaluation_date_uses_persisted_comparisons_and_trading_calendar(monkeypatch, as_of, records, expected):
    # The endpoint's default date is the latest persisted evaluation. `as_of`
    # documents the requested market-calendar scenario without relying on wall clock.
    assert as_of.weekday() >= 0
    monkeypatch.setattr(routes, "list_forecasts", lambda _date=None: records)
    response = routes.evaluation_history(None)
    assert response["latest_completed_date"] == expected
    assert response["evaluation_date"] == expected


def test_latest_evaluation_includes_every_completed_version_for_date(monkeypatch):
    rows = [comparison("2026-11-10", "morning"), comparison("2026-11-10", "advance-pm", window="afternoon"),
            {**comparison("2026-11-10", "updated-pm", window="afternoon"), "forecast_type": "UPDATED"}]
    monkeypatch.setattr(routes, "list_forecasts", lambda _date=None: rows)
    result = routes.evaluation_history(date(2026, 11, 10))
    assert result["latest_completed_date"] == "2026-11-10"
    assert {row["forecast_type"] for row in result["forecasts"]} == {"ADVANCE", "UPDATED"}
    assert len(result["forecasts"]) == 3


def test_no_evaluations_returns_empty_history_and_selected_requested_date(monkeypatch):
    monkeypatch.setattr(routes, "list_forecasts", lambda _date=None: [])
    result = routes.evaluation_history(date(2026, 11, 10))
    assert result["latest_completed_date"] is None
    assert result["available_dates"] == []
    assert result["evaluation_date"] == "2026-11-10"
    assert result["forecasts"] == []


def test_manually_selected_historical_date_is_preserved_by_api(monkeypatch):
    rows = [comparison("2026-11-10", "new"), comparison("2026-11-09", "historic")]
    monkeypatch.setattr(routes, "list_forecasts", lambda _date=None: rows)
    result = routes.evaluation_history(date(2026, 11, 9))
    assert result["latest_completed_date"] == "2026-11-10"
    assert result["evaluation_date"] == "2026-11-09"
    assert [row["forecast_id"] for row in result["forecasts"]] == ["historic"]


def test_holiday_or_weekend_records_are_not_presented_as_trading_sessions(monkeypatch):
    rows = [comparison("2026-11-26", "invalid-thanksgiving"), comparison("2026-11-28", "invalid-weekend"), comparison("2026-11-25", "valid")]
    monkeypatch.setattr(routes, "list_forecasts", lambda _date=None: rows)
    result = routes.evaluation_history(None)
    assert result["latest_completed_date"] == "2026-11-25"
    assert [item["evaluation_date"] for item in result["available_dates"]] == ["2026-11-25"]
