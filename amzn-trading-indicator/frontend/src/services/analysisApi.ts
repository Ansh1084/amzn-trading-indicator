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
  paperDashboard: () => request<PaperDashboard>('/paper-trading/dashboard'),
  paperSettings: (values: Partial<PaperSettings>) => request<PaperSettings>('/paper-trading/settings', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(values),
  }),
  closePaperTrade: (tradeId: string) => request<{ trade: PaperTrade; fill_timestamp: string; fill_source: string }>(`/paper-trading/trades/${encodeURIComponent(tradeId)}/close`, { method: 'POST' }),
};

export interface PaperSettings {
  enabled: boolean; starting_capital: number; sizing_mode: 'fixed_shares' | 'fixed_capital' | 'equity_percent'; sizing_value: number;
  transaction_cost_bps: number; slippage_bps: number; enabled_at?: string | null;
}
export interface PaperTrade {
  trade_id: string; signal_id: string; symbol: string; setup_type: string; direction: 'LONG' | 'SHORT'; status: string;
  signal_timestamp: string; entry_timestamp?: string | null; entry_price?: number | null; entry_fill_price?: number | null; quantity: number; remaining_quantity: number;
  capital_allocated: number; stop_loss: number; target_1: number; target_2: number; exit_price?: number | null; exit_fill_price?: number | null;
  exit_timestamp?: string | null; exit_reason?: string | null; gross_pnl: number; transaction_costs: number; slippage_cost: number;
  net_pnl?: number | null; r_multiple?: number | null; mark_price?: number | null; mark_timestamp?: string | null;
  unrealized_net_pnl?: number | null;
}
export interface PaperDashboard {
  disclaimer: string; settings: PaperSettings; metrics: Record<string, number | string | null | boolean>;
  open_trades: PaperTrade[]; pending_trades: PaperTrade[]; closed_trades: PaperTrade[];
  equity_curve: Array<{ timestamp: string; equity: number }>;
  journal: Array<{ journal_id: number; trade_id: string; timestamp: string; event_type: string; details: Record<string, unknown> }>;
  mark: { price: number | null; timestamp: string | null; source: string; freshness: string; market_state: string };
}
export interface MarketIntelligence {
  as_of: string; latest_completed_candle: string | null; decision_weight_changed: false; lookahead_policy: string;
  trend_alignment: { overall: string; explanation: string; conflict_warning: boolean; timeframes: Array<Record<string, number | string>> };
  opening_range: Record<string, number | string | null>; rvol: Record<string, number | string | null>;
  previous_day_levels: Record<string, unknown>;
}
