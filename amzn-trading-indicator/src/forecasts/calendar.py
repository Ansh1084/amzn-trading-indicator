"""Small, deterministic NYSE calendar helpers for session and cutoff logic."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def _observed(day: date) -> date:
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    return d + timedelta(days=(weekday - d.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    d = date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _easter(year: int) -> date:
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def market_holidays(year: int) -> set[date]:
    """Regular full-day NYSE holidays (including observed fixed-date holidays)."""
    holidays = {
        _observed(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _easter(year) - timedelta(days=2),
        _last_weekday(year, 5, 0),
        _observed(date(year, 6, 19)) if year >= 2022 else date.min,
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
        _observed(date(year, 12, 25)),
    }
    next_new_year_observed = _observed(date(year + 1, 1, 1))
    if next_new_year_observed.year == year:
        holidays.add(next_new_year_observed)
    if year >= 1998:
        holidays.add(_nth_weekday(year, 2, 0, 3))
    return holidays


def is_trading_day(day: date) -> bool:
    return day.weekday() < 5 and day not in market_holidays(day.year)


def previous_trading_day(day: date) -> date:
    candidate = day - timedelta(days=1)
    while not is_trading_day(candidate):
        candidate -= timedelta(days=1)
    return candidate


def next_trading_day(day: date) -> date:
    candidate = day + timedelta(days=1)
    while not is_trading_day(candidate):
        candidate += timedelta(days=1)
    return candidate


def session_close(day: date) -> time:
    # Standard scheduled 13:00 ET closes. The day before the observed
    # Independence Day closure is an early close (skip a weekend predecessor).
    thanksgiving = _nth_weekday(day.year, 11, 3, 4)
    black_friday = thanksgiving + timedelta(days=1)
    independence_observed = _observed(date(day.year, 7, 4))
    independence_early = independence_observed - timedelta(days=1)
    while independence_early.weekday() >= 5:
        independence_early -= timedelta(days=1)
    if day in {black_friday, independence_early} or (day.month == 12 and day.day == 24 and day.weekday() < 5):
        return time(13, 0)
    return time(16, 0)


def market_timestamp(day: date, value: str) -> datetime:
    hour, minute = (int(x) for x in value.split(":"))
    return datetime.combine(day, time(hour, minute), tzinfo=ET)


def cutoff_for_evaluation(day: date) -> datetime:
    """Exact end of previous regular session, converted to UTC."""
    prev = previous_trading_day(day)
    return datetime.combine(prev, session_close(prev), tzinfo=ET).astimezone(ZoneInfo("UTC"))
