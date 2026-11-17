export interface MarketCandle {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  observed_minute_count?: number;
}

export interface Zone {
  id?: string;
  lower: number;
  upper: number;
  midpoint: number;
  type: 'SUPPORT' | 'RESISTANCE';
  zone_type?: 'SUPPLY' | 'DEMAND';
  timeframe?: '1H' | '15M' | '5M' | string;
  status?: 'ACTIVE' | 'TESTED' | 'INVALIDATED' | string;
  formation_timestamp?: string;
  formation_price?: number;
  latest_retest_timestamp?: string | null;
  invalidation_timestamp?: string | null;
  touches?: number;
  touch_sessions?: string[];
  strength_score?: number;
  distance_from_price?: number;
  distance_atr?: number;
  age_sessions?: number;
  method?: string;
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

export interface Decision60mComponent {
  score: number | null;
  weight: number;
  weighted_contribution: number;
  summary: string;
  evidence: string[];
  meta: Record<string, unknown>;
}

export interface PredictionTiming {
  candle_timestamp: string;
  candle_close_timestamp: string;
  next_prediction_timestamp: string;
  next_prediction_display: string;
  last_prediction_display: string;
  last_candle_display: string;
  last_candle_close_display?: string;
  last_analysis_completed_display?: string;
  is_market_open: boolean;
  market_state: 'OPEN' | 'CLOSED' | 'PRE_MARKET';
  headline_display: string;
  is_stale: boolean;
}

export interface Active60mSignal {
  signal_id?: string;
  direction: 'BUY' | 'SELL';
  status: string;
  trigger_price: number;
  entry_zone?: { lower: number; upper: number };
  stop_loss: number;
  target_1: number;
  target_2: number;
  risk_reward_ratio?: number;
  candle_timestamp: string;
  is_existing_position: boolean;
  valid?: boolean;
  validation_error?: string | null;
  unrealized_pnl_pct?: number | null;
}

export interface Decision60mPayload {
  intended_horizon_minutes: number;
  intended_horizon_bars: number;
  decision: 'BUY' | 'SELL' | 'HOLD' | 'WAIT FOR CONFIRMATION';
  directional_bias: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  directional_score: number;
  score_type: string;
  gating_status: string;
  reasons: string[];
  invalidation_conditions: string[];
  timing?: PredictionTiming;
  components: {
    trend?: Decision60mComponent;
    structure?: Decision60mComponent;
    momentum?: Decision60mComponent;
    volume?: Decision60mComponent;
    vwap_price_action?: Decision60mComponent;
  };
  risk_plan?: RiskPlan;
  active_signal?: Active60mSignal | null;
  dedicated_60m_ml_model?: {
    status: string;
    message?: string;
  };
  ml_10m_supporting?: {
    model_name?: string;
    model_version?: string;
    prediction_horizon_minutes?: number;
    available?: boolean;
    direction?: string | null;
    raw_probability_up?: number | null;
    raw_directional_score?: number | null;
    score_type?: string;
    role?: string;
  };
  evaluation_candle_timestamp?: string;
  decision_timestamp?: string;
  market_timezone?: string;
}

export interface Decision60mRecord {
  signal_id: string;
  candle_timestamp: string;
  generated_at: string;
  analysis_version: string;
  decision: 'BUY' | 'SELL' | 'HOLD' | 'WAIT FOR CONFIRMATION';
  directional_bias: string;
  directional_score: number;
  horizon_minutes: number;
  horizon_bars: number;
  trigger_price: number;
  entry_zone_lower?: number;
  entry_zone_upper?: number;
  stop_loss?: number;
  target_1?: number;
  target_2?: number;
  risk_reward_ratio?: number;
  payload: Decision60mPayload;
  status: string;
  outcome?: Record<string, unknown> | null;
  gross_return_pct?: number | null;
  net_return_pct?: number | null;
  resolved_at?: string | null;
}

export interface Decision60mPerformance {
  has_data: boolean;
  total_decisions: number;
  counts: { buy: number; sell: number; hold: number; wait: number };
  resolved_count: number;
  win_rate: number | null;
  average_net_return_pct: number | null;
  gross_pnl_pct: number | null;
  net_pnl_pct: number | null;
  profit_factor: number | null;
  cost_assumption_bps?: number;
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
    demand_zones?: Zone[];
    supply_zones?: Zone[];
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
  decision_60m?: Decision60mPayload;
  timing?: PredictionTiming;
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
  latest_timestamp: string | null;
  candles: MarketCandle[];
  forming_candle?: MarketCandle | null;
  stream_status?: MarketStreamStatus;
  event_cursor?: number;
}

export interface MarketStreamStatus {
  provider?: string; symbol?: string; feed?: string;
  connection_state?: 'CONNECTED' | 'CONNECTING' | 'DISCONNECTED' | 'RECONNECTING' | 'STALE' | 'ERROR' | string;
  market_state?: 'OPEN' | 'CLOSED' | string; freshness?: 'FRESH' | 'STALE' | 'UNAVAILABLE' | string;
  last_received_event_at?: string | null; last_completed_candle?: string | null;
  last_persisted_candle?: string | null; last_reconciliation_at?: string | null;
  reconciliation_error?: string | null; last_error?: string | null; retry_count?: number;
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

export interface MLPerformanceResponse {
  has_data: boolean;
  empty_reason?: string;
  window: string;
  requested_window_size: number | null;
  sample_count: number;
  is_partial: boolean;
  total_predictions_tracked: number;
  pending_count: number;
  excluded_count: number;
  total_resolved_count: number;
  exclusion_breakdown: Record<string, number>;
  start_date: string | null;
  end_date: string | null;
  last_updated: string;
  model_identifier: {
    model_name: string;
    model_version: string;
    artifact_sha256?: string;
    horizon_minutes: number;
    input_feature_count: number;
  };
  summary: {
    accuracy: number;
    macro_f1: number;
    weighted_f1: number;
    balanced_accuracy: number;
    precision_overall: number;
    recall_overall: number;
    total_resolved: number;
    correct: number;
    incorrect: number;
  } | null;
  metrics: {
    accuracy: number;
    up: {
      precision: number;
      recall: number;
      f1: number;
      actual_count: number;
      predicted_count: number;
    };
    down: {
      precision: number;
      recall: number;
      f1: number;
      actual_count: number;
      predicted_count: number;
    };
    macro_f1: number;
    weighted_f1: number;
    balanced_accuracy: number;
  } | null;
  confusion_matrix: {
    true_up_pred_up: number;
    true_up_pred_down: number;
    true_down_pred_up: number;
    true_down_pred_down: number;
    total_actual_up: number;
    total_actual_down: number;
    total_pred_up: number;
    total_pred_down: number;
    total_resolved: number;
  } | null;
  probability_evaluation: {
    log_loss: number;
    brier_score: number;
    expected_calibration_error: number;
    is_uncalibrated: boolean;
    raw_score_notice: string;
    calibration_bins: Array<{
      bin_label: string;
      lower: number;
      upper: number;
      sample_count: number;
      mean_predicted_prob: number | null;
      observed_positive_rate: number | null;
    }>;
  } | null;
  detailed_table: Array<{
    metric: string;
    up: string;
    down: string;
    overall: string;
    tooltip?: string;
  }>;
}
