export interface MarketCandle {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface Zone {
  lower: number;
  upper: number;
  midpoint: number;
  type: 'SUPPORT' | 'RESISTANCE';
  status?: string;
  strength_score?: number;
  distance_from_price?: number;
}

export interface RiskPlan {
  valid: boolean;
  reason?: string;
  direction?: 'BUY' | 'SELL';
  trigger_price?: number;
  entry_zone?: { lower: number; upper: number };
  stop_loss?: number;
  target_1?: number;
  target_2?: number;
  risk_reward_ratio?: number;
  atr_reference?: number;
  setup_invalidation_level?: number;
  stop_method?: string;
  target_method?: string;
}

export interface AnalysisPayload {
  analysis_version: string;
  symbol: string;
  source?: string;
  generated_at: string;
  as_of?: string;
  candle_timestamp: string;
  candle_close_timestamp?: string;
  market_timezone?: string;
  bars_used?: number;
  history_start?: string;
  data_gaps?: Array<{ from: string; to: string; minutes: number }>;
  excluded_calendar_rows?: number;
  indicators?: Record<string, string | number | boolean | null>;
  structure?: {
    status?: string;
    support_zones?: Zone[];
    resistance_zones?: Zone[];
    nearest_support?: Zone | null;
    nearest_resistance?: Zone | null;
    breakout?: { status?: string; volume_confirmed?: boolean };
    breakdown?: { status?: string; volume_confirmed?: boolean };
    sequence?: Record<string, boolean | null>;
  };
  market_regime?: { classification?: string; status?: string; evidence?: string[]; adx_14?: number | null };
  ml_evidence?: {
    available: boolean;
    direction?: 'UP' | 'DOWN' | null;
    raw_probability_up?: number | null;
    raw_directional_score?: number | null;
    score_type?: string;
    model_name?: string;
    model_version?: string;
    input_feature_count?: number;
    generated_at?: string;
    candle_timestamp?: string;
    reason?: string;
  };
  hypothetical_only?: boolean;
  classification: string;
  directional_bias?: string;
  final_decision_score?: number;
  technical_score?: number;
  score_type?: string;
  components?: Record<string, number | null>;
  normalized_component_weights?: Record<string, number>;
  component_contributions?: Record<string, number>;
  dominant_contributors?: Array<{ component: string; contribution: number }>;
  bullish_evidence?: string[];
  bearish_evidence?: string[];
  neutral_observations?: string[];
  conflicting_signals?: string[];
  reasons?: string[];
  required_confirmation?: string;
  setup_invalidation?: string;
  risk_plan?: RiskPlan;
}

export interface SignalRecord {
  signal_id: string;
  candle_timestamp: string;
  generated_at: string;
  analysis_version: string;
  classification: string;
  directional_bias: string;
  final_score?: number | null;
  payload: AnalysisPayload;
  status: string;
  outcome?: Record<string, unknown> | null;
  gross_return_pct?: number | null;
  net_return_pct?: number | null;
  events?: Array<{ timestamp: string; event_type: string; details: string }>;
}

export interface MarketSnapshot {
  symbol: string;
  feed: string;
  source: string;
  dataset_rows: number;
  chart_rows: number;
  latest_timestamp: string;
  candles: MarketCandle[];
}

export interface AnalysisLatestResponse {
  // The latest route returns "empty" only when no row exists. For a stored row,
  // the database status (ACTIVE, CLOSED, etc.) overwrites the route's "ok" value.
  status?: string;
  detail?: string;
  signal_id?: string;
  candle_timestamp?: string;
  generated_at?: string;
  classification?: string;
  directional_bias?: string;
  final_score?: number | null;
  payload?: AnalysisPayload;
  outcome?: Record<string, unknown> | null;
  gross_return_pct?: number | null;
  net_return_pct?: number | null;
  analysis?: null;
}

export interface DataSnapshotStatus {
  source: string;
  feed: string;
  path?: string;
  rows?: number;
  dataset_start?: string;
  dataset_end?: string;
  latest_completed_rth_candle?: string;
  last_sync_at?: string | null;
}
