import { API_BASE, request } from './api';
import type { AnalysisLatestResponse, DataSnapshotStatus, MarketSnapshot, MarketStreamStatus, SignalRecord } from '../types/analysis';

export const analysisApi = {
  market: (limit = 1000) => request<MarketSnapshot>(`/market/candles?limit=${limit}&source=local`),
  marketAfter: (timestamp: string) => request<MarketSnapshot>(`/market/candles?limit=1000&source=local&after=${encodeURIComponent(timestamp)}`),
  status: () => request<DataSnapshotStatus>('/data/status?chart_limit=1000'),
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
};

export interface MarketIntelligence {
  as_of: string; latest_completed_candle: string | null; decision_weight_changed: false; lookahead_policy: string;
  trend_alignment: { overall: string; explanation: string; conflict_warning: boolean; timeframes: Array<Record<string, number | string>> };
  opening_range: Record<string, number | string | null>; rvol: Record<string, number | string | null>;
  previous_day_levels: Record<string, unknown>;
}
