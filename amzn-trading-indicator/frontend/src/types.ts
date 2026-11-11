export type SessionId = 'morning' | 'afternoon';
export type ForecastVersion = 'ADVANCE' | 'UPDATED';
export type Outcome = 'CORRECT' | 'INCORRECT' | 'PENDING';

export interface SessionPrediction {
  prediction_id: string;
  evaluation_date: string;
  session_id: SessionId;
  forecast_type: ForecastVersion;
  revision: number;
  interval_start: string;
  interval_end: string;
  generated_at: string;
  feature_timestamp: string;
  data_cutoff: string;
  reference_price: number;
  predicted_signal: 'UP' | 'DOWN';
  predicted_direction: 'UP' | 'DOWN';
  raw_probability_up: number;
  confidence: null;
  confidence_status: string;
  support: number | null;
  resistance: number | null;
  model_name: string;
  model_version: string;
  model_sha256: string;
  actual: null | {
    actual_signal: 'UP' | 'DOWN' | 'FLAT';
    actual_direction: 'UP' | 'DOWN' | 'FLAT';
    actual_close: number;
    actual_return_pct: number;
    outcome: Exclude<Outcome, 'PENDING'>;
    correct: boolean;
    evaluated_at: string;
    actual_bars: number;
  };
}

export interface SessionResult {
  evaluation_date: string;
  session_id: SessionId;
  forecast_type: ForecastVersion;
  model_version: string | null;
  expected_intervals: number;
  interval_count?: number;
  predictions: SessionPrediction[];
  final_decision: null | {
    decision: 'UP' | 'DOWN' | 'NEUTRAL';
    method: string;
    up_count: number;
    down_count: number;
    neutral_count: number;
    confidence: null;
    confidence_status: string;
  };
  actual_session: null | { decision: 'UP' | 'DOWN' | 'FLAT'; actual_intervals: number; return_pct: number };
  pending_count: number;
  completed_count?: number;
  confidence_note?: string;
  idempotent?: boolean;
}

export interface ActualCandle {
  time: string;
  interval_end: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface MarketSnapshot {
  symbol: string;
  feed: string;
  source: string;
  dataset_rows: number;
  chart_rows: number;
  latest_timestamp: string;
  candles: Array<{ timestamp: string; open: number; high: number; low: number; close: number; volume: number }>;
}

export interface DataStatus {
  source: string;
  feed: string;
  path: string;
  rows: number;
  dataset_start: string;
  dataset_end: string;
  latest_completed_rth_candle: string;
  last_sync_at: string | null;
}

export interface Operation {
  operation_id: string;
  kind: string;
  status: string;
  stage: string;
  progress: number;
  updated_at: string;
  error?: string | null;
  result?: Record<string, unknown> | null;
}
