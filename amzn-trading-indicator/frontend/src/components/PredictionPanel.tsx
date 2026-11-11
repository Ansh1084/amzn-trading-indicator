import type { SessionPrediction } from '../types';
import { formatTime, price } from '../utils/format';

export function PredictionPanel({ prediction, modelVersion, source, latestCandle }: {
  prediction?: SessionPrediction; modelVersion?: string | null; source?: string; latestCandle?: string;
}) {
  const signal = prediction?.predicted_signal;
  return <div className="panel prediction-panel">
    <div className="section-heading compact-heading"><div><span className="eyebrow">SELECTED 10-MIN INTERVAL</span><h2>Prediction signal</h2></div><span className={`signal-badge ${signal?.toLowerCase() || 'pending'}`}>{signal || 'PENDING'}</span></div>
    <div className={`big-signal ${signal?.toLowerCase() || 'pending'}`}>{signal || '—'}</div>
    <p className="signal-time">{prediction ? `${formatTime(prediction.interval_start)} → ${formatTime(prediction.interval_end)}` : 'Load or generate a stored session schedule'}</p>
    <div className="score-box"><span>RAW MODEL SCORE · NOT CALIBRATED CONFIDENCE</span><strong>{prediction ? `${(prediction.raw_probability_up * 100).toFixed(1)}% P(UP)` : '—'}</strong></div>
    <div className="metric-list">
      <div><span>Reference price</span><b>{price(prediction?.reference_price)}</b></div>
      <div><span>Support</span><b>{price(prediction?.support)}</b></div>
      <div><span>Resistance</span><b>{price(prediction?.resistance)}</b></div>
      <div><span>Evaluation</span><b className={prediction?.actual ? 'good-copy' : 'muted-copy'}>{prediction?.actual?.outcome || 'PENDING'}</b></div>
    </div>
    <div className="instrument-grid">
      <div><small>SYMBOL</small><b>AMZN</b></div><div><small>HORIZON</small><b>10 minutes</b></div>
      <div><small>MODEL</small><b title={modelVersion || prediction?.model_version}>{modelVersion || prediction?.model_version || '—'}</b></div>
      <div><small>LAST ACTUAL BAR</small><b>{latestCandle ? formatTime(latestCandle, true) : '—'}</b></div>
      <div className="wide-tile"><small>DATA SOURCE</small><b>{source || '—'}</b></div>
    </div>
  </div>;
}
