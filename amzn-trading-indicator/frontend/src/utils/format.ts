export const etDate = (d = new Date()) => new Intl.DateTimeFormat('en-CA', {
  timeZone: 'America/New_York', year: 'numeric', month: '2-digit', day: '2-digit',
}).format(d);

export const formatDate = (value?: string) => value
  ? new Intl.DateTimeFormat('en-US', { timeZone: 'UTC', month: 'short', day: '2-digit', year: 'numeric' }).format(new Date(`${value.slice(0, 10)}T12:00:00Z`))
  : '—';

export const formatTime = (value?: string | null, includeDate = false) => {
  if (!value) return '—';
  const options: Intl.DateTimeFormatOptions = includeDate
    ? { timeZone: 'America/New_York', month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: true }
    : { timeZone: 'America/New_York', hour: '2-digit', minute: '2-digit', hour12: true };
  return `${new Intl.DateTimeFormat('en-US', options).format(new Date(value))} ET`;
};

export const price = (value?: number | null) => value == null || !Number.isFinite(value) ? '—' : `$${value.toFixed(2)}`;
export const percent = (value?: number | null, digits = 2) => value == null || !Number.isFinite(value) ? '—' : `${value.toFixed(digits)}%`;

export function marketClockStatus(latest?: string | null) {
  if (!latest) return { label: 'DATA UNAVAILABLE', state: 'neutral' as const };
  const now = new Date();
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/New_York', weekday: 'short', hour: '2-digit', minute: '2-digit', hourCycle: 'h23',
  }).formatToParts(now);
  const day = parts.find((p) => p.type === 'weekday')?.value || '';
  const hour = Number(parts.find((p) => p.type === 'hour')?.value || 0);
  const minute = Number(parts.find((p) => p.type === 'minute')?.value || 0);
  const mins = hour * 60 + minute;
  const age = now.getTime() - new Date(latest).getTime();
  const fresh = age >= -60_000 && age <= 12 * 60_000;
  const clockOpen = !['Sat', 'Sun'].includes(day) && mins >= 570 && mins < 960;
  if (clockOpen && fresh) return { label: 'MARKET DATA FRESH', state: 'good' as const };
  if (clockOpen) return { label: 'RTH · WAITING FOR DATA', state: 'warning' as const };
  return { label: 'MARKET CLOSED', state: 'neutral' as const };
}

export function getPredictionSchedule(
  candleTimestamp?: string | null,
  marketClosed = false,
  isStale = false,
  backendTiming?: import('../types/analysis').PredictionTiming | null,
) {
  if (backendTiming) {
    const isNowStale = Boolean(isStale || backendTiming.is_stale);
    const isOpen = marketClosed ? false : backendTiming.is_market_open;
    let headline = backendTiming.headline_display;
    if (isNowStale) {
      headline = 'WAITING FOR MARKET DATA';
    } else if (marketClosed || !isOpen) {
      headline = backendTiming.headline_display.includes('MARKET CLOSED')
        ? backendTiming.headline_display
        : 'MARKET CLOSED — Next prediction: Next trading session';
    }
    return {
      ...backendTiming,
      is_stale: isNowStale,
      is_market_open: isOpen,
      market_state: marketClosed ? ('CLOSED' as const) : backendTiming.market_state,
      headline_display: headline,
      last_analysis_completed_display: backendTiming.last_analysis_completed_display || backendTiming.last_prediction_display,
    };
  }
  if (!candleTimestamp) {
    return {
      candle_timestamp: '',
      candle_close_timestamp: '',
      next_prediction_timestamp: '',
      next_prediction_display: 'Next trading session',
      last_prediction_display: '—',
      last_candle_display: '—',
      last_candle_close_display: '—',
      last_analysis_completed_display: '—',
      is_market_open: !marketClosed,
      market_state: marketClosed ? ('CLOSED' as const) : ('OPEN' as const),
      headline_display: isStale ? 'WAITING FOR MARKET DATA' : (marketClosed ? 'MARKET CLOSED — Next prediction: Next trading session' : 'NEXT PREDICTION: Scheduled on candle complete'),
      is_stale: isStale,
    };
  }

  const candleStart = new Date(candleTimestamp);
  const candleClose = new Date(candleStart.getTime() + 5 * 60_000);
  const nextPred = new Date(candleClose.getTime() + 5 * 60_000);

  const timeFormatter = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/New_York',
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
  });

  const nextDisplay = `${timeFormatter.format(nextPred)} ET`;
  const lastPredDisplay = `${timeFormatter.format(candleClose)} ET`;
  const lastCandleDisplay = `${timeFormatter.format(candleStart)} ET`;

  let headline = `NEXT PREDICTION: ${nextDisplay}`;
  if (isStale) {
    headline = 'WAITING FOR MARKET DATA';
  } else if (marketClosed) {
    headline = 'MARKET CLOSED — Next prediction: Next trading session';
  }

  return {
    candle_timestamp: candleTimestamp,
    candle_close_timestamp: candleClose.toISOString(),
    next_prediction_timestamp: nextPred.toISOString(),
    next_prediction_display: nextDisplay,
    last_prediction_display: lastPredDisplay,
    last_candle_display: lastCandleDisplay,
    last_candle_close_display: lastPredDisplay,
    last_analysis_completed_display: lastPredDisplay,
    is_market_open: !marketClosed,
    market_state: marketClosed ? ('CLOSED' as const) : ('OPEN' as const),
    headline_display: headline,
    is_stale: isStale,
  };
}
