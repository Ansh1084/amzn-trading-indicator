import { useState, useMemo } from 'react';
import type { AnalysisPayload, DataSnapshotStatus, MarketSnapshot, SignalRecord } from '../../types/analysis';
import type { MarketIntelligence } from '../../services/analysisApi';
import { formatTime, price } from '../../utils/format';
import { LiveMLPerformancePanel } from './LiveMLPerformancePanel';
import { Decision60mPanel } from './Decision60mPanel';

const n = (value: unknown, digits = 2) =>
  typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';
const text = (value: unknown) => (value == null || value === '' ? '—' : String(value));
const number = (value: unknown) => (typeof value === 'number' && Number.isFinite(value) ? value : null);

const componentLabels: Record<string, string> = {
  trend: 'Moving average structure evidence',
  momentum: 'RSI / MACD momentum',
  volume: 'Signed session rolling volume evidence',
  structure: 'Market structure',
  price_action: 'Candle price action',
  ml: 'Production ML evidence',
};

const etDateFormatter = new Intl.DateTimeFormat('en-CA', {
  timeZone: 'America/New_York',
  year: 'numeric',
  month: '2-digit',
  day: '2-digit',
});

function formatStatus(val: unknown) {
  if (val == null || val === '') return '—';
  return String(val).replaceAll('_', ' ');
}

function formatEvidenceItem(item: unknown, indicators?: AnalysisPayload['indicators']) {
  if (typeof item !== 'string') return String(item ?? '');
  if (item.startsWith('Trend alignment is ')) {
    return `Moving average structure is ${String(indicators?.trend_classification || 'unavailable')} (EMA 9/20/50 ordering and slopes); this is separate from 1H/15M/5M trend alignment.`;
  }
  return item.replace(/^Relative volume=/, 'Session rolling volume ratio=');
}

export function AnalysisPanel({
  record,
  intelligence,
  intelligenceError,
  market,
  dataStatus,
  fresh,
  marketClosed = false,
  latestAligned,
}: {
  record: SignalRecord | null;
  intelligence?: MarketIntelligence | null;
  intelligenceError?: string;
  market: MarketSnapshot | null;
  dataStatus: DataSnapshotStatus | null;
  fresh: boolean;
  marketClosed?: boolean;
  latestAligned: boolean;
}) {
  const payload = record?.payload;
  const ml = payload?.ml_evidence;
  const risk = payload?.risk_plan;

  // Formula notes visibility drawer
  const [showFormulas, setShowFormulas] = useState(false);

  // Market Intelligence extracted data
  const orb = intelligence?.opening_range || {};
  const rvol = intelligence?.rvol || {};
  const levels = (intelligence?.previous_day_levels || {}) as Record<string, unknown>;
  const distance =
    levels.distance && typeof levels.distance === 'object'
      ? (levels.distance as Record<string, unknown>)
      : {};

  // Timeframe trend breakdown (1H, 15M, 5M)
  const timeframes = intelligence?.trend_alignment?.timeframes || [
    { timeframe: '1H', trend: 'LOADING', bars: null },
    { timeframe: '15M', trend: 'LOADING', bars: null },
    { timeframe: '5M', trend: 'LOADING', bars: null },
  ];
  const trend1H = String(timeframes.find((t) => t.timeframe === '1H')?.trend || 'NEUTRAL').toUpperCase();
  const bars1H = timeframes.find((t) => t.timeframe === '1H')?.bars;
  const trend15M = String(timeframes.find((t) => t.timeframe === '15M')?.trend || 'NEUTRAL').toUpperCase();
  const bars15M = timeframes.find((t) => t.timeframe === '15M')?.bars;
  const trend5M = String(timeframes.find((t) => t.timeframe === '5M')?.trend || 'NEUTRAL').toUpperCase();
  const bars5M = timeframes.find((t) => t.timeframe === '5M')?.bars;

  const alignmentOverall = intelligence?.trend_alignment?.overall || 'LOADING';
  const isAligned = alignmentOverall.includes('ALIGNED');

  // Final Decision Classification
  const classification = record?.classification || payload?.classification || 'UNAVAILABLE';
  const decisionSlug =
    classification === 'BUY'
      ? 'buy'
      : classification === 'SELL'
      ? 'sell'
      : classification === 'HOLD'
      ? 'hold'
      : 'wait';

  // Directional Evidence helpers
  const maClassification = String(payload?.indicators?.trend_classification || 'Not reported');
  const maTone = maClassification.includes('BULLISH')
    ? 'pos'
    : maClassification.includes('BEARISH')
    ? 'neg'
    : 'neutral';

  const shortTermBias = payload?.directional_bias || 'Neutral';
  const shortTermTone =
    shortTermBias.toUpperCase().includes('BULL')
      ? 'pos'
      : shortTermBias.toUpperCase().includes('BEAR')
      ? 'neg'
      : 'neutral';

  // Heuristic Component Contributions
  const components = useMemo(() => {
    const raw = Object.entries(payload?.component_contributions || {});
    return raw
      .filter((entry): entry is [string, number] => typeof entry[1] === 'number' && Number.isFinite(entry[1]))
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  }, [payload?.component_contributions]);

  const scale = useMemo(() => {
    if (!components.length) return 1;
    return Math.max(0.0001, ...components.map(([, val]) => Math.abs(val)));
  }, [components]);

  // Evidence counts
  const bullishCount = payload?.bullish_evidence?.length || 0;
  const bearishCount = payload?.bearish_evidence?.length || 0;
  const neutralCount = payload?.neutral_observations?.length || 0;

  // Market Statistics computations
  const candles = market?.candles || [];
  const latestCandle = candles.at(-1);
  const sessionDate = latestCandle?.timestamp
    ? (() => {
        try {
          const d = new Date(latestCandle.timestamp);
          return Number.isNaN(d.getTime()) ? '' : etDateFormatter.format(d);
        } catch {
          return '';
        }
      })()
    : '';
  const sessionBars = sessionDate
    ? candles.filter((c) => {
        try {
          const d = new Date(c.timestamp);
          return !Number.isNaN(d.getTime()) && etDateFormatter.format(d) === sessionDate;
        } catch {
          return false;
        }
      })
    : [];
  const high = sessionBars.length ? Math.max(...sessionBars.map((c) => c.high)) : null;
  const low = sessionBars.length ? Math.min(...sessionBars.map((c) => c.low)) : null;
  let pv = 0;
  let volume = 0;
  for (const c of sessionBars) {
    pv += ((c.high + c.low + c.close) / 3) * c.volume;
    volume += c.volume;
  }
  const sessionVwap = volume ? pv / volume : null;
  const indicators = payload?.indicators || {};
  const analysisTime = payload?.candle_timestamp ? new Date(payload.candle_timestamp).getTime() : NaN;
  const analysisDate = payload?.candle_timestamp
    ? (() => {
        try {
          const d = new Date(payload.candle_timestamp);
          return Number.isNaN(d.getTime()) ? '' : etDateFormatter.format(d);
        } catch {
          return '';
        }
      })()
    : '';
  const analysisSessionBars =
    Number.isFinite(analysisTime) && analysisDate
      ? candles.filter((c) => {
          try {
            const d = new Date(c.timestamp);
            return !Number.isNaN(d.getTime()) && etDateFormatter.format(d) === analysisDate && d.getTime() <= analysisTime;
          } catch {
            return false;
          }
        })
      : [];
  const volumeSampleCount =
    typeof indicators.volume_observation_count_20 === 'number'
      ? indicators.volume_observation_count_20
      : analysisSessionBars.some((c) => new Date(c.timestamp).getTime() === analysisTime)
      ? Math.min(20, analysisSessionBars.length)
      : null;

  // 15 Verified Market Statistics fields
  const cvdRaw = indicators.cvd ?? indicators.cumulative_volume_delta;
  const statItems: Array<{ label: string; value: string; tooltip?: string }> = [
    { label: 'Last completed close', value: price(latestCandle?.close) },
    { label: 'Session high / low', value: high == null || low == null ? '—' : `${price(high)} / ${price(low)}` },
    { label: 'Session range', value: high == null || low == null ? '—' : price(high - low) },
    { label: 'Session VWAP*', value: price(sessionVwap), tooltip: 'Volume-Weighted Average Price computed from actual completed 5m OHLCV bars' },
    { label: 'VWAP distance', value: latestCandle && sessionVwap != null ? `${n(((latestCandle.close / sessionVwap) - 1) * 100)}%` : '—' },
    { label: 'CVD', value: cvdRaw != null ? n(cvdRaw, 2) : '—', tooltip: 'Cumulative Volume Delta' },
    { label: 'RSI 14', value: n(indicators.rsi_14, 1), tooltip: '14-period Relative Strength Index momentum oscillator' },
    { label: 'MACD', value: n(indicators.macd, 3), tooltip: 'Moving Average Convergence Divergence' },
    { label: 'ATR percentage', value: indicators.atr_pct == null ? '—' : `${n(indicators.atr_pct, 2)}%`, tooltip: 'Average True Range expressed as percentage of price' },
    {
      label: 'Session rolling volume ratio',
      value: indicators.relative_volume == null ? '—' : `${n(indicators.relative_volume, 2)}×`,
      tooltip: 'Current 5m candle volume ÷ mean of up to 20 completed bars in the SAME session. Distinct from historical same-slot RVOL.',
    },
    { label: 'Volatility regime', value: formatStatus(indicators.volatility_classification) },
    { label: 'Market regime', value: formatStatus(payload?.market_regime?.classification) },
    { label: 'Feed', value: String(market?.feed || dataStatus?.feed || 'Unavailable') },
    { label: 'Latest market candle', value: formatTime(market?.latest_timestamp || dataStatus?.latest_completed_rth_candle, true) },
    { label: 'Analysis input candle', value: formatTime(payload?.candle_timestamp, true) },
  ];

  return (
    <div className="tp-container" aria-label="AMZN AI Trading Decision Terminal">
      {/* =========================================================================
          1. MARKET OVERVIEW (Clean, spacious and compact)
          ========================================================================= */}
      <section className="tp-card tp-overview-card" aria-label="Section 1: Market Overview">
        <div className="tp-card-header">
          <h2 className="tp-heading">Market Overview</h2>
          <div className="tp-badge-group">
            <span className={`tp-status-pill ${marketClosed ? 'pill-muted' : fresh ? 'pill-green' : 'pill-amber'}`}>
              {marketClosed ? 'Market Closed' : fresh ? 'Data Fresh' : 'Data Stale'}
            </span>
            <span className={`tp-status-pill ${isAligned ? 'pill-blue' : 'pill-amber'}`}>
              {alignmentOverall}
            </span>
          </div>
        </div>

        {intelligenceError && <div className="tp-alert-box error">{intelligenceError}</div>}

        {/* 3 Prominent Timeframe Trend Cards */}
        <div className="tp-trend-grid">
          <div className={`tp-trend-tile trend-${trend1H.toLowerCase()}`}>
            <span className="tp-tile-label">1H Trend</span>
            <b className="tp-tile-val">{trend1H}</b>
            <span className="tp-tile-sub">{bars1H != null ? `${Number(bars1H).toLocaleString()} bars` : '—'}</span>
          </div>
          <div className={`tp-trend-tile trend-${trend15M.toLowerCase()}`}>
            <span className="tp-tile-label">15M Trend</span>
            <b className="tp-tile-val">{trend15M}</b>
            <span className="tp-tile-sub">{bars15M != null ? `${Number(bars15M).toLocaleString()} bars` : '—'}</span>
          </div>
          <div className={`tp-trend-tile trend-${trend5M.toLowerCase()}`}>
            <span className="tp-tile-label">5M Trend</span>
            <b className="tp-tile-val">{trend5M}</b>
            <span className="tp-tile-sub">{bars5M != null ? `${Number(bars5M).toLocaleString()} bars` : '—'}</span>
          </div>
        </div>

        {!latestAligned && record && (
          <div className="tp-alert-box warning">
            Analysis is based on completed candle at {formatTime(record.candle_timestamp, true)}. Does not match latest market candle.
          </div>
        )}
      </section>

      {/* =========================================================================
          2. ML MODEL PREDICTION (Dedicated Prominent Component)
          ========================================================================= */}
      <section className="tp-card tp-ml-card" aria-label="Section 2: ML Model Prediction">
        <div className="tp-card-header">
          <h3 className="tp-heading">ML Model Prediction</h3>
          <span className="tp-horizon-pill">Horizon: Next 10 Minutes</span>
        </div>

        {ml?.available && ml.raw_probability_up != null ? (
          (() => {
            const rawUp = ml.raw_probability_up;
            const rawDown = Math.max(0, Math.min(1, 1.0 - rawUp));
            const upPct = (rawUp * 100).toFixed(1);
            const downPct = (rawDown * 100).toFixed(1);
            const isUp = ml.direction === 'UP';

            return (
              <div className="tp-ml-content">
                {/* Big Direction Callout */}
                <div className="tp-ml-verdict-row">
                  <span className="tp-ml-verdict-label">Model Direction</span>
                  <b className={`tp-ml-direction ${isUp ? 'tone-up' : 'tone-down'}`}>
                    {ml.direction || (rawUp >= 0.5 ? 'UP' : 'DOWN')}
                  </b>
                </div>

                {/* Score Labels Row */}
                <div className="tp-ml-score-row">
                  <div className="tp-ml-score-col score-up">
                    <span className="score-label">Raw P(UP)</span>
                    <b className="score-val">{upPct}%</b>
                  </div>
                  <div className="tp-ml-score-col score-down">
                    <span className="score-label">Raw P(DOWN)</span>
                    <b className="score-val">{downPct}%</b>
                  </div>
                </div>

                {/* Horizontal Two-Sided Split Bar */}
                <div
                  className="tp-split-bar-track"
                  title={`Raw P(UP): ${upPct}%, Raw P(DOWN): ${downPct}%`}
                  role="progressbar"
                  aria-valuenow={Number(upPct)}
                  aria-valuemin={0}
                  aria-valuemax={100}
                >
                  <div
                    className="tp-split-segment seg-up"
                    style={{ width: `${upPct}%` }}
                  />
                  <div
                    className="tp-split-segment seg-down"
                    style={{ width: `${downPct}%` }}
                  />
                </div>

                {/* Uncalibrated Warning & Secondary Metadata */}
                <div className="tp-ml-footer">
                  <span className="tp-uncalibrated-tag">
                    Raw model scores — uncalibrated (not confidence or profit probability)
                  </span>
                  <span className="tp-ml-meta">
                    {ml.model_name || 'LGBMClassifier'} ({ml.model_version || 'legacy-10m'}) · {ml.input_feature_count ?? 40} inputs
                  </span>
                </div>
              </div>
            );
          })()
        ) : (
          <div className="tp-ml-unavailable">
            <span className="tp-ml-unavail-title">ML Prediction Unavailable</span>
            <p className="tp-ml-unavail-msg">
              {ml?.reason || 'Insufficient market history or feature warm-up for 40-feature model inference.'}
            </p>
          </div>
        )}
      </section>

      {/* =========================================================================
          3. 60-MINUTE PRIMARY TRADING DECISION SYSTEM
          ========================================================================= */}
      <Decision60mPanel
        decision60m={payload?.decision_60m}
        candleTimestamp={payload?.candle_timestamp}
        generatedAt={payload?.generated_at}
        marketClosed={marketClosed}
        isStale={!fresh}
        activeSignalStatus={payload?.decision_60m?.active_signal ? `ACTIVE ${payload.decision_60m.active_signal.direction}` : null}
        ml10Evidence={ml}
      />

      {!record || !payload ? (
        <div className="tp-card tp-empty-card">
          <b>Analysis unavailable</b>
          <p>No stored analysis record was returned. Click “Analyze latest candle” above to compute a fresh decision.</p>
        </div>
      ) : (
        <>
          {/* =====================================================================
              5. IMPORTANT MARKET INTELLIGENCE
              ===================================================================== */}
          <section className="tp-card" aria-label="Section 5: Market Intelligence">
            <div className="tp-card-header">
              <h3 className="tp-heading">Market Intelligence</h3>
            </div>

            <div className="tp-intel-list">
              {/* 15-Minute Opening Range */}
              <div className="tp-intel-row">
                <div className="tp-intel-header-row">
                  <span className="tp-intel-name">15M Opening Range</span>
                  <span className="tp-intel-pill">{formatStatus(orb.classification || orb.status)}</span>
                </div>
                <div className="tp-intel-values">
                  <span>High <b>{price(number(orb.high))}</b></span>
                  <span className="tp-sep">·</span>
                  <span>Low <b>{price(number(orb.low))}</b></span>
                  <span className="tp-sep">·</span>
                  <span>Mid <b>{price(number(orb.midpoint))}</b></span>
                </div>
              </div>

              {/* Historical Same-Slot RVOL */}
              <div
                className="tp-intel-row"
                title="Historical Same-Slot RVOL: Volume compared to historical average for this exact time slot across prior sessions."
              >
                <div className="tp-intel-header-row">
                  <span className="tp-intel-name">Historical Same-Slot RVOL</span>
                  <span className="tp-intel-pill">
                    {number(rvol.rvol) != null ? `${number(rvol.rvol)!.toFixed(2)}× · ${formatStatus(rvol.classification)}` : formatStatus(rvol.classification)}
                  </span>
                </div>
                <div className="tp-intel-values">
                  <span><b>{text(rvol.current_volume)}</b> current ÷ <b>{number(rvol.average_same_slot_volume) != null ? number(rvol.average_same_slot_volume)!.toFixed(2) : '—'}</b> avg</span>
                  <span className="tp-sep">·</span>
                  <span className="tp-muted-meta">{text(rvol.sample_count)} sessions · {text(rvol.slot_et)} ET</span>
                </div>
              </div>

              {/* Previous Session Levels */}
              <div className="tp-intel-row">
                <div className="tp-intel-header-row">
                  <span className="tp-intel-name">Previous Session ({text(levels.session_date)})</span>
                  {number(distance.close) != null && (
                    <span className={`tp-intel-pill ${number(distance.close)! >= 0 ? 'pos' : 'neg'}`}>
                      Δ close {price(number(distance.close))}
                    </span>
                  )}
                </div>
                <div className="tp-intel-values">
                  <span>High <b>{price(number(levels.high))}</b></span>
                  <span className="tp-sep">·</span>
                  <span>Low <b>{price(number(levels.low))}</b></span>
                  <span className="tp-sep">·</span>
                  <span>Close <b>{price(number(levels.close))}</b></span>
                </div>
              </div>
            </div>
          </section>

          {/* =====================================================================
              8. LIVE ML MODEL PERFORMANCE
              ===================================================================== */}
          <LiveMLPerformancePanel lastUpdated={record?.generated_at} />

          {/* =====================================================================
              9. VERIFIED INPUTS / MARKET STATISTICS
              ===================================================================== */}
          <section className="tp-card" aria-label="Section 9: Market Statistics">
            <div className="tp-card-header">
              <h3 className="tp-heading">Verified Inputs / Market Statistics</h3>
              <span className="tp-pill-counter">15 Verified Fields</span>
            </div>

            <div className="tp-stats-table">
              {statItems.map((item, idx) => (
                <div className="tp-stat-item" key={idx} title={item.tooltip}>
                  <span className="tp-stat-label">{item.label}</span>
                  <b className="tp-stat-value">{item.value}</b>
                </div>
              ))}
            </div>

            {/* Calculation definitions drawer */}
            <div className="tp-notes-drawer">
              <button
                type="button"
                className="tp-notes-toggle"
                onClick={() => setShowFormulas((prev) => !prev)}
              >
                <span>{showFormulas ? '▾ Hide Volume & Formula Notes' : '▸ Volume & Formula Notes'}</span>
              </button>

              {showFormulas && (
                <div className="tp-notes-body">
                  <p>
                    <strong>1. Historical Same-Slot RVOL (Section 5):</strong> Current candle volume compared to the historical average for this exact time slot across prior completed trading sessions. Formula: current candle volume ÷ average volume for the same time slot across previous sessions.
                  </p>
                  <p>
                    <strong>2. Session Rolling Volume Ratio (Stat 10):</strong> Current candle volume compared against the mean of up to 20 completed 5-minute bars in the <em>same</em> active ET session.
                  </p>
                  {indicators.relative_volume != null && (
                    <div className="tp-notes-calc">
                      Session volume breakdown: <b>{n(indicators.volume, 0)}</b> current shares ÷ <b>{n(indicators.volume_sma_20, 2)}</b> same-session mean across <b>{n(volumeSampleCount, 0)}</b> observed bars.
                    </div>
                  )}
                  <span className="tp-notes-footnote">
                    Session VWAP is calculated from actual 5m OHLCV bars. Indicator values as of {formatTime(payload?.candle_timestamp, true)}.
                  </span>
                </div>
              )}
            </div>
          </section>
        </>
      )}
    </div>
  );
}
