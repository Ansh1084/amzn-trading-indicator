import type { ActualCandle, DataStatus, ForecastVersion, MarketSnapshot, Operation, SessionId, SessionResult } from '../types';

export const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api').replace(/\/$/, '');

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, init);
  } catch {
    throw new Error(`Cannot reach the backend at ${API_BASE}. Start FastAPI and check VITE_API_BASE_URL.`);
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`);
  return body as T;
}

const query = (date: string, session: SessionId, version?: ForecastVersion) =>
  `evaluation_date=${encodeURIComponent(date)}&session_id=${session}${version ? `&forecast_type=${version}` : ''}`;

export const api = {
  market: () => request<MarketSnapshot>('/market/candles?limit=1&source=local'),
  dataStatus: () => request<DataStatus>('/data/status?chart_limit=1'),
  operations: () => request<{ operations: Operation[] }>('/operations'),
  session: (date: string, session: SessionId, version: ForecastVersion) =>
    request<SessionResult>(`/session-predictions?${query(date, session, version)}`),
  actualCandles: (date: string, session: SessionId) =>
    request<{ candles: ActualCandle[] }>(`/session-predictions/actual-candles?${query(date, session)}`),
  generate: (date: string, session: SessionId, version: ForecastVersion) =>
    request<SessionResult>('/session-predictions/generate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ evaluation_date: date, session_id: session, forecast_type: version }),
    }),
  evaluate: (date: string, session: SessionId, version: ForecastVersion) =>
    request<SessionResult>('/session-predictions/evaluate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ evaluation_date: date, session_id: session, forecast_type: version }),
    }),
  refreshData: () => request<Operation>('/data/refresh', { method: 'POST' }),
  exportUrl: (date: string, session: SessionId, version: ForecastVersion) =>
    `${API_BASE}/session-predictions/export.csv?${query(date, session, version)}`,
};
