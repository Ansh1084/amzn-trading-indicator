import { useMemo } from 'react';
import type { AnalysisPayload, Decision60mPayload } from '../../types/analysis';
import { formatTime, getPredictionSchedule, price } from '../../utils/format';

interface Props {
  decision60m?: Decision60mPayload | null;
  candleTimestamp?: string;
  generatedAt?: string;
  marketClosed?: boolean;
  isStale?: boolean;
  activeSignalStatus?: string | null;
  ml10Evidence?: AnalysisPayload['ml_evidence'] | null;
}

function validateTradePlan(
  plan?: {
    valid?: boolean;
    direction?: string;
    stop_loss?: number;
    target_1?: number;
    target_2?: number;
    trigger_price?: number;
    risk_reward_ratio?: number;
  } | null,
  expectedDirection?: string,
): boolean {
  if (!plan) return false;
  const dir = (plan.direction || expectedDirection || '').toUpperCase();
  if (dir !== 'BUY' && dir !== 'SELL') return false;
  if (expectedDirection && dir !== expectedDirection.toUpperCase()) return false;
  const stop = plan.stop_loss;
  const t1 = plan.target_1;
  const t2 = plan.target_2;
  const entry = plan.trigger_price;
  if (stop == null || t1 == null || t2 == null || entry == null) return false;
  if (dir === 'BUY') {
    if (!(stop < entry && entry < t1 && t1 < t2)) return false;
    const risk = entry - stop;
    const reward = t2 - entry;
    if (risk <= 0 || reward <= 0) return false;
    if (reward / risk < 1.49) return false;
  } else if (dir === 'SELL') {
    if (!(stop > entry && entry > t1 && t1 > t2)) return false;
    const risk = stop - entry;
    const reward = entry - t2;
    if (risk <= 0 || reward <= 0) return false;
    if (reward / risk < 1.49) return false;
  }
  return true;
}

export function Decision60mPanel({
  decision60m,
  candleTimestamp,
  generatedAt,
  marketClosed = false,
  isStale = false,
  activeSignalStatus,
  ml10Evidence,
}: Props) {
  const d = decision60m;
  const decision = d?.decision || 'HOLD';
  const score = d?.directional_score ?? 0.0;
  const bias = d?.directional_bias || 'NEUTRAL';
  const risk = d?.risk_plan;
  const components = d?.components || {};
  const activeSignal = d?.active_signal;

  const hasActivePosition = Boolean(
    activeSignal &&
    activeSignal.status &&
    ['ACTIVE', 'NEW', 'TARGET_1_HIT'].includes(activeSignal.status),
  );

  const isNewPlanValid = useMemo(() => {
    return Boolean(risk?.valid && validateTradePlan(risk, decision));
  }, [risk, decision]);

  const isActivePlanValid = useMemo(() => {
    if (!activeSignal) return false;
    return Boolean(validateTradePlan(activeSignal, activeSignal.direction));
  }, [activeSignal]);

  // Timing & Prediction Schedule
  const schedule = useMemo(() => {
    return getPredictionSchedule(
      candleTimestamp || d?.evaluation_candle_timestamp,
      marketClosed,
      isStale,
      d?.timing,
    );
  }, [candleTimestamp, d?.evaluation_candle_timestamp, marketClosed, isStale, d?.timing]);

  // Main decision styling (BUY=green, SELL=red, WAIT=amber, HOLD=neutral grey)
  const decisionTheme = useMemo(() => {
    switch (decision) {
      case 'BUY':
        return { cls: 'd60-dec-buy', label: 'BUY', tone: 'pos' };
      case 'SELL':
        return { cls: 'd60-dec-sell', label: 'SELL', tone: 'neg' };
      case 'WAIT FOR CONFIRMATION':
        return { cls: 'd60-dec-wait', label: 'WAIT', tone: 'warn' };
      default:
        return { cls: 'd60-dec-hold', label: 'HOLD', tone: 'neutral' };
    }
  }, [decision]);

  // 1. Trend: Bullish / Bearish / Mixed
  const trend = useMemo(() => {
    const s = components.trend?.summary || '';
    const sc = components.trend?.score ?? 0;
    if (s.includes('BULLISH') || sc > 0.15) return { text: 'Bullish', tone: 'pos', icon: '▲' };
    if (s.includes('BEARISH') || sc < -0.15) return { text: 'Bearish', tone: 'neg', icon: '▼' };
    return { text: 'Mixed', tone: 'warn', icon: '◆' };
  }, [components.trend]);

  // 2. Structure: Bullish / Bearish / Mixed
  const structure = useMemo(() => {
    const s = components.structure?.summary || '';
    const sc = components.structure?.score ?? 0;
    if (s.includes('BULLISH') || sc > 0.15) return { text: 'Bullish', tone: 'pos', icon: '▲' };
    if (s.includes('BEARISH') || sc < -0.15) return { text: 'Bearish', tone: 'neg', icon: '▼' };
    return { text: 'Mixed', tone: 'warn', icon: '◆' };
  }, [components.structure]);

  // 3. Momentum: Positive / Negative / Neutral
  const momentum = useMemo(() => {
    const s = components.momentum?.summary || '';
    const sc = components.momentum?.score ?? 0;
    if (s.includes('BULLISH') || sc > 0.10) return { text: 'Positive', tone: 'pos', icon: '▲' };
    if (s.includes('BEARISH') || sc < -0.10) return { text: 'Negative', tone: 'neg', icon: '▼' };
    return { text: 'Neutral', tone: 'neutral', icon: '●' };
  }, [components.momentum]);

  // 4. Volume: Confirming / Weak / Unavailable
  const volume = useMemo(() => {
    const s = components.volume?.summary || '';
    const sc = components.volume?.score ?? 0;
    if (s.includes('UNAVAILABLE') || s.includes('NO_DATA')) return { text: 'Unavailable', tone: 'neutral', icon: '○' };
    if (s.includes('CONFIRM') || sc > 0.10) return { text: 'Confirming', tone: 'pos', icon: '✓' };
    return { text: 'Weak', tone: 'warn', icon: '—' };
  }, [components.volume]);

  // One-line explanation (max 2 concise reasons)
  const explanation = useMemo(() => {
    const raw = d?.reasons || [];
    if (!raw.length) return 'Market conditions within balanced baseline. Awaiting directional breakout.';
    const sentences = raw.slice(0, 2);
    return sentences.join(' ');
  }, [d?.reasons]);

  // Supporting 10M ML context
  const ml10 = ml10Evidence || d?.ml_10m_supporting;
  const ml10Dir = ml10?.direction || (ml10?.raw_probability_up != null ? (ml10.raw_probability_up >= 0.5 ? 'UP' : 'DOWN') : null);
  const ml10Up = ml10?.raw_probability_up;

  return (
    <div className="d60-card" aria-label="60-Minute Trading Decision">
      {/* ── A. HEADER & NEXT PREDICTION BANNER ────────────────────────────── */}
      <div className="d60-header-block">
        <div className="d60-title-row">
          <span className="d60-card-kicker">60-MINUTE TRADING DECISION</span>
          <div className="d60-meta-pills">
            <span className="d60-meta-pill">Last candle: <b>{schedule.last_candle_display}</b></span>
            <span className="d60-meta-pill">Analysis completed: <b>{schedule.last_analysis_completed_display || schedule.last_prediction_display}</b></span>
            <span className="d60-meta-pill">Horizon: <b>60 min</b></span>
          </div>
        </div>

        {/* PROMINENT NEXT PREDICTION BANNER */}
        <div className={`d60-next-banner ${schedule.is_stale ? 'banner-stale' : !schedule.is_market_open ? 'banner-closed' : 'banner-active'}`}>
          <span className="d60-next-dot" />
          <b className="d60-next-text">{schedule.headline_display}</b>
        </div>
      </div>

      {/* ── B. MAIN DECISION & EXISTING SIGNAL STATUS ─────────────────────── */}
      <div className="d60-decision-row">
        <div className={`d60-badge ${decisionTheme.cls}`}>
          <span className="d60-badge-lbl">RECOMMENDATION</span>
          <b className="d60-badge-val">{decisionTheme.label}</b>
        </div>

        <div className="d60-decision-meta">
          {/* Explicit Separation: Existing Position / Signal vs Recommendation */}
          <div className="d60-pos-line">
            <span className="d60-k">Existing Position/Signal:</span>
            {hasActivePosition && activeSignal ? (
              <span className="d60-active-pos-block">
                <b className={`d60-pos-badge pos-${activeSignal.direction.toLowerCase()}`}>
                  ACTIVE {activeSignal.direction}
                </b>
                <span className="d60-pos-meta">
                  Entry {price(activeSignal.trigger_price)} · Stop {price(activeSignal.stop_loss)} · T2 {price(activeSignal.target_2)}
                  {activeSignal.unrealized_pnl_pct != null && (
                    <span className={`pos-pnl ${activeSignal.unrealized_pnl_pct >= 0 ? 'pos-pnl-gain' : 'pos-pnl-loss'}`}>
                      {' '}({activeSignal.unrealized_pnl_pct >= 0 ? '+' : ''}{activeSignal.unrealized_pnl_pct.toFixed(2)}%)
                    </span>
                  )}
                </span>
              </span>
            ) : (
              <span className="d60-no-pos-text">None (No open position)</span>
            )}
          </div>

          <div className="d60-bias-line">
            <span className="d60-k">Market Bias:</span>
            <b className={`d60-v bias-${bias.toLowerCase()}`}>{bias}</b>
            <span className="d60-score-sep">|</span>
            <span className="d60-k">Directional Score:</span>
            <b className={`d60-v score-${decisionTheme.tone}`}>
              {score > 0 ? `+${score.toFixed(3)}` : score.toFixed(3)}
            </b>
            <span className="d60-score-tag">HEURISTIC [-1.0, +1.0]</span>
          </div>
        </div>
      </div>

      {/* ── C. COMPACT EVIDENCE ROW (EXACTLY FOUR INDICATORS) ─────────────── */}
      <div className="d60-indicators-grid">
        <div className={`d60-ind-tile ind-${trend.tone}`}>
          <span className="d60-ind-k">Trend</span>
          <div className="d60-ind-v-row">
            <span className="d60-ind-icon">{trend.icon}</span>
            <b className="d60-ind-v">{trend.text}</b>
          </div>
        </div>

        <div className={`d60-ind-tile ind-${structure.tone}`}>
          <span className="d60-ind-k">Structure</span>
          <div className="d60-ind-v-row">
            <span className="d60-ind-icon">{structure.icon}</span>
            <b className="d60-ind-v">{structure.text}</b>
          </div>
        </div>

        <div className={`d60-ind-tile ind-${momentum.tone}`}>
          <span className="d60-ind-k">Momentum</span>
          <div className="d60-ind-v-row">
            <span className="d60-ind-icon">{momentum.icon}</span>
            <b className="d60-ind-v">{momentum.text}</b>
          </div>
        </div>

        <div className={`d60-ind-tile ind-${volume.tone}`}>
          <span className="d60-ind-k">Volume</span>
          <div className="d60-ind-v-row">
            <span className="d60-ind-icon">{volume.icon}</span>
            <b className="d60-ind-v">{volume.text}</b>
          </div>
        </div>
      </div>

      {/* ── D. TRADE PLAN (CONDITIONAL RENDERING BY DECISION & ACTIVE POSITION) ──── */}
      {(decision === 'BUY' || decision === 'SELL') && isNewPlanValid ? (
        /* Case 1: Actionable New BUY or SELL Setup */
        <div className="d60-trade-plan">
          <div className="d60-plan-heading">
            <span>60-MINUTE EXECUTION PLAN</span>
            <span className={`d60-plan-pill ${decision === 'BUY' ? 'pill-pos' : 'pill-neg'}`}>
              New {decision} Setup Approved
            </span>
          </div>
          <div className="d60-plan-grid">
            <div className="d60-plan-box">
              <span className="box-k">Entry Zone</span>
              <b className="box-v">
                {risk?.entry_zone ? `${price(risk.entry_zone.lower)}–${price(risk.entry_zone.upper)}` : price(risk?.trigger_price)}
              </b>
            </div>
            <div className="d60-plan-box">
              <span className="box-k">Stop Loss</span>
              <b className="box-v tone-neg">{price(risk?.stop_loss)}</b>
            </div>
            <div className="d60-plan-box">
              <span className="box-k">Target 1</span>
              <b className="box-v tone-pos">{price(risk?.target_1)}</b>
            </div>
            <div className="d60-plan-box">
              <span className="box-k">Target 2</span>
              <b className="box-v tone-pos">{price(risk?.target_2)}</b>
            </div>
            <div className="d60-plan-box">
              <span className="box-k">Risk/Reward</span>
              <b className="box-v tone-pos">
                {risk?.risk_reward_ratio != null ? `${risk.risk_reward_ratio.toFixed(2)} : 1` : '—'}
              </b>
            </div>
          </div>
        </div>
      ) : hasActivePosition && isActivePlanValid && activeSignal ? (
        /* Case 2: HOLD or WAIT with an Existing Active Signal */
        <div className="d60-trade-plan d60-existing-signal-plan">
          <div className="d60-plan-heading">
            <span className="d60-existing-signal-title">
              ACTIVE {activeSignal.direction} — EXISTING SIGNAL
            </span>
            <span className="d60-plan-pill pill-warn">
              Managing Active Position
            </span>
          </div>
          <div className="d60-existing-notice">
            Active position from candle {formatTime(activeSignal.candle_timestamp, true)}. Managing risk to target/stop; this is not a new entry recommendation.
          </div>
          <div className="d60-plan-grid">
            <div className="d60-plan-box">
              <span className="box-k">Entry Price</span>
              <b className="box-v">{price(activeSignal.trigger_price)}</b>
            </div>
            <div className="d60-plan-box">
              <span className="box-k">Stop Loss</span>
              <b className="box-v tone-neg">{price(activeSignal.stop_loss)}</b>
            </div>
            <div className="d60-plan-box">
              <span className="box-k">Target 1</span>
              <b className="box-v tone-pos">{price(activeSignal.target_1)}</b>
            </div>
            <div className="d60-plan-box">
              <span className="box-k">Target 2</span>
              <b className="box-v tone-pos">{price(activeSignal.target_2)}</b>
            </div>
            <div className="d60-plan-box">
              <span className="box-k">Risk/Reward</span>
              <b className="box-v tone-pos">
                {activeSignal.risk_reward_ratio != null ? `${activeSignal.risk_reward_ratio.toFixed(2)} : 1` : '—'}
              </b>
            </div>
          </div>
        </div>
      ) : (
        /* Case 3: HOLD or WAIT without active position -> Clean No-Setup strip */
        <div className="d60-no-setup-strip">
          <span className="no-setup-icon">ℹ</span>
          <span className="no-setup-text">
            {decision === 'HOLD'
              ? (risk?.reason || 'No new trade setup approved. Market conditions within neutral holding balance.')
              : (risk?.reason || 'Awaiting confirmation: directional momentum detected but structural confirmation gates are active.')}
          </span>
        </div>
      )}

      {/* ── E. ONE-LINE EXPLANATION ────────────────────────────────────────── */}
      <div className="d60-explanation-line">
        <span className="exp-label">Rationale:</span>
        <span className="exp-text">{explanation}</span>
      </div>

      {/* ── SEPARATE 10M ML SUPPORTING ITEM ────────────────────────────────── */}
      <div className="d60-ml10-strip">
        <span className="ml10-chip-tag">10M ML Supporting:</span>
        {ml10?.available && ml10Dir ? (
          <span className="ml10-chip-data">
            <b>10M ML: {ml10Dir}</b> | Raw P(UP): {ml10Up != null ? ml10Up.toFixed(3) : '—'} | <span className="ml10-muted">Uncalibrated micro-momentum</span>
          </span>
        ) : (
          <span className="ml10-chip-data ml10-muted">10M ML: Not available for current candle</span>
        )}
      </div>
    </div>
  );
}
