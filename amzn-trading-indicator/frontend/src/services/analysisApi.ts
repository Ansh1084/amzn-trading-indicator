import { API_BASE, request } from './api';
import type { AnalysisLatestResponse, DataSnapshotStatus, MarketSnapshot, SignalRecord } from '../types/analysis';

export const analysisApi = {
  market: (limit = 1000) => request<MarketSnapshot>(`/market/candles?limit=${limit}&source=local`),
  status: () => request<DataSnapshotStatus>('/data/status?chart_limit=1000'),
  latest: () => request<AnalysisLatestResponse>('/analysis/latest'),
  signals: (offset = 0, limit = 100, status?: string) => {
    const query = new URLSearchParams({ offset: String(offset), limit: String(limit) });
    if (status) query.set('status', status);
    return request<{ signals: SignalRecord[]; limit: number; offset: number; hypothetical_only: boolean }>(`/analysis/signals?${query}`);
  },
  signal: (id: string) => request<SignalRecord>(`/analysis/signals/${encodeURIComponent(id)}`),
  analyze: () => request<{ status: string; analysis: AnalysisLatestResponse['payload']; signal_record: SignalRecord }>('/analysis/run?source=local', { method: 'POST' }),
  refreshSnapshot: () => request<{ operation_id: string; kind: string; status: string; stage: string; progress: number; updated_at?: string; error?: string | null }>('/data/refresh', { method: 'POST' }),
  operations: () => request<{ operations: Array<{ operation_id: string; kind: string; status: string; stage: string; progress: number; updated_at: string; error?: string | null }> }>('/operations'),
  latestSignalUrl: () => `${API_BASE}/analysis/latest`,
};
