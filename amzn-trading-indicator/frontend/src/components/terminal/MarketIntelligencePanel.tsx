import type { MarketIntelligence } from '../../services/analysisApi';
import { formatTime, price } from '../../utils/format';

const text = (value: unknown) => (value == null || value === '' ? '—' : String(value));
const number = (value: unknown) => (typeof value === 'number' && Number.isFinite(value) ? value : null);

export function MarketIntelligencePanel({
  data,
  error = '',
  analysisTimestamp,
}: {
  data: MarketIntelligence | null;
  error?: string;
  analysisTimestamp?: string | null;
  initiallyExpanded?: boolean;
}) {
  const orb = data?.opening_range || {};
  const rvol = data?.rvol || {};
  const levels = (data?.previous_day_levels || {}) as Record<string, unknown>;
  const distance =
    levels.distance && typeof levels.distance === 'object'
      ? (levels.distance as Record<string, unknown>)
      : {};

  const timeframes = data?.trend_alignment?.timeframes || [
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

  const overall = data?.trend_alignment?.overall || 'LOADING';
  const isAligned = overall.includes('ALIGNED');

  return (
    <div className="tp-card">
      <div className="tp-card-header">
        <h2 className="tp-heading">Market Overview</h2>
        <span className={`tp-status-pill ${isAligned ? 'pill-blue' : 'pill-amber'}`}>
          {overall}
        </span>
      </div>

      {error && <div className="tp-alert-box error">{error}</div>}

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

      <div className="tp-alignment-box">
        <p className="tp-alignment-copy">
          {data?.trend_alignment?.explanation || 'Timeframe trend summary is based on completed bars.'}
        </p>
      </div>

      <div className="tp-intel-list" style={{ marginTop: '12px' }}>
        <div className="tp-intel-row">
          <div className="tp-intel-header-row">
            <span className="tp-intel-name">15M Opening Range</span>
            <span className="tp-intel-pill">{text(orb.classification || orb.status)}</span>
          </div>
          <div className="tp-intel-values">
            <span>High <b>{price(number(orb.high))}</b></span>
            <span className="tp-sep">·</span>
            <span>Low <b>{price(number(orb.low))}</b></span>
            <span className="tp-sep">·</span>
            <span>Mid <b>{price(number(orb.midpoint))}</b></span>
          </div>
        </div>

        <div className="tp-intel-row">
          <div className="tp-intel-header-row">
            <span className="tp-intel-name">Historical Same-Slot RVOL</span>
            <span className="tp-intel-pill">
              {number(rvol.rvol) != null ? `${number(rvol.rvol)!.toFixed(2)}× · ${text(rvol.classification)}` : text(rvol.classification)}
            </span>
          </div>
          <div className="tp-intel-values">
            <span><b>{text(rvol.current_volume)}</b> current ÷ <b>{number(rvol.average_same_slot_volume) != null ? number(rvol.average_same_slot_volume)!.toFixed(2) : '—'}</b> avg</span>
            <span className="tp-sep">·</span>
            <span className="tp-muted-meta">{text(rvol.sample_count)} sessions · {text(rvol.slot_et)} ET</span>
          </div>
        </div>

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

      {analysisTimestamp && (
        <div style={{ marginTop: '8px', fontSize: '11px', color: '#64748b', textAlign: 'right' }}>
          As of {formatTime(analysisTimestamp, true)}
        </div>
      )}
    </div>
  );
}
