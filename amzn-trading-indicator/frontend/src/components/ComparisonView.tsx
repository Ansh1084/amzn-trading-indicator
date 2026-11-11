import type { ActualCandle, SessionPrediction } from '../types';
import { ActualChart } from './ActualChart';
import { formatTime, price } from '../utils/format';

export function ComparisonView({ candles, predictions }: { candles: ActualCandle[]; predictions: SessionPrediction[] }) {
  return <div className="comparison-grid">
    <section className="panel compare-card">
      <div className="section-heading"><div><span className="eyebrow">ACTUAL · GENUINE MARKET DATA</span><h2>Actual AMZN candles</h2><p>10-minute OHLC aggregated from complete local 5-minute bars. Markers show each stored prediction’s outcome.</p></div><span className="panel-count">{candles.length} BARS</span></div>
      {candles.length ? <ActualChart candles={candles} predictions={predictions} /> : <div className="empty-state">No complete actual candles are available for this session yet.</div>}
      <div className="chart-footer"><span>OHLC · 10 MIN</span><span className="legend-inline"><i className="legend-dot up" />Correct <i className="legend-dot down" />Incorrect <i className="legend-dot pending" />Pending</span></div>
    </section>
    <section className="panel compare-card predicted-compare">
      <div className="section-heading"><div><span className="eyebrow">PREDICTED · DIRECTIONAL MODEL</span><h2>Forecast signals</h2><p>Same session intervals and timestamps; no invented forecast prices or candles.</p></div><span className="panel-count">{predictions.length} SIGNALS</span></div>
      <div className="compare-signal-list">{predictions.map((p) => {
        const actual = candles.find((c) => c.time === p.interval_start);
        return <article className="compare-signal-row" key={p.prediction_id}>
          <time>{formatTime(p.interval_start)}<small>{formatTime(p.interval_end)}</small></time>
          <span className={`signal-badge small ${p.predicted_signal.toLowerCase()}`}>{p.predicted_signal}</span>
          <span className="raw-score">{(p.raw_probability_up * 100).toFixed(1)}% <small>raw P(UP)</small></span>
          <span className={`outcome ${p.actual?.outcome.toLowerCase() || 'pending'}`}>{p.actual?.outcome || 'PENDING'}</span>
          <span className="compare-close">{actual ? price(actual.close) : '—'}</span>
        </article>;
      })}{!predictions.length && <div className="empty-state">No predictions are stored for this date/session/version.</div>}</div>
      <div className="honesty-note"><b>Predicted OHLC model unavailable</b><span>The existing LightGBM artifact classifies direction only. This panel shows its stored directional output and raw score; predicted open, high, low, and close are intentionally omitted.</span></div>
    </section>
  </div>;
}
