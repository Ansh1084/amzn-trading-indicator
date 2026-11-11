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
