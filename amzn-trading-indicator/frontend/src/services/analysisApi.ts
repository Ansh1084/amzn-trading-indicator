import { API_BASE, request } from './api';
import type { AnalysisLatestResponse, DataSnapshotStatus, MarketSnapshot, MarketStreamStatus, MLPerformanceResponse, SignalRecord } from '../types/analysis';

export const analysisApi = {
  market: (limit = 1500) => request<MarketSnapshot>(`/market/candles?limit=${limit}&source=local`),
  marketAfter: (timestamp: string) => request<MarketSnapshot>(`/market/candles?limit=1500&source=local&after=${encodeURIComponent(timestamp)}`),
  status: () => request<DataSnapshotStatus>('/data/status?chart_limit=1500'),
  streamStatus: () => request<MarketStreamStatus>('/market/stream/status'),
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
  marketIntelligence: () => request<MarketIntelligence>('/analysis/market-intelligence'),
  mlPerformance: (window = '50', resolve = true) =>
    request<MLPerformanceResponse>(`/analysis/ml-performance?window=${encodeURIComponent(window)}&resolve=${resolve}`),
  resolveMLPerformance: () =>
    request<{ status: string; resolved_count: number; excluded_count: number; still_pending_count: number }>(
      '/analysis/ml-performance/resolve',
      { method: 'POST' }
    ),
  latest60m: () =>
    request<{ status: string; signal_id?: string; decision?: string; payload?: import('../types/analysis').Decision60mPayload }>('/analysis/60m/latest'),
  signals60m: (offset = 0, limit = 50, status?: string) => {
    const query = new URLSearchParams({ offset: String(offset), limit: String(limit) });
    if (status) query.set('status', status);
    return request<{ signals: import('../types/analysis').Decision60mRecord[]; limit: number; offset: number; horizon_minutes: number }>(`/analysis/60m/signals?${query}`);
  },
  performance60m: () =>
    request<import('../types/analysis').Decision60mPerformance>('/analysis/60m/performance'),
};

export interface MarketIntelligence {
  as_of: string; latest_completed_candle: string | null; decision_weight_changed: false; lookahead_policy: string;
  trend_alignment: { overall: string; explanation: string; conflict_warning: boolean; timeframes: Array<Record<string, number | string>> };
  opening_range: Record<string, number | string | null>; rvol: Record<string, number | string | null>;
  previous_day_levels: Record<string, unknown>;
}
