import { useMemo, useState } from 'react';
import type { SessionPrediction } from '../types';
import { formatTime, percent, price } from '../utils/format';

export function PredictionHistory({ predictions }: { predictions: SessionPrediction[] }) {
  const [descending, setDescending] = useState(true);
  const rows = useMemo(() => [...predictions].sort((a, b) => descending
    ? b.interval_start.localeCompare(a.interval_start) : a.interval_start.localeCompare(b.interval_start)), [predictions, descending]);
  return <section className="panel history-panel">
    <div className="section-heading"><div><span className="eyebrow">FROZEN INTERVAL RECORDS</span><h2>Prediction history</h2><p>Stored forecasts are unchanged after issue; actual values appear only after evaluation.</p></div>
      <button className="sort-button" onClick={() => setDescending((v) => !v)}>TIME {descending ? '↓' : '↑'}</button></div>
    <div className="table-scroll"><table className="history-table"><thead><tr><th>Time · ET</th><th>Reference</th><th>Signal</th><th>Raw score</th><th>Predicted close</th><th>Support</th><th>Resistance</th><th>Actual direction</th><th>Outcome</th><th>Return</th></tr></thead>
      <tbody>{rows.map((item) => <tr key={item.prediction_id}>
        <td>{formatTime(item.interval_start)}–{formatTime(item.interval_end)}</td><td>{price(item.reference_price)}</td>
        <td><span className={`signal-badge small ${item.predicted_signal.toLowerCase()}`}>{item.predicted_signal}</span></td>
        <td>{percent(item.raw_probability_up * 100, 1)} <em>raw</em></td><td className="not-available">— <em>no OHLC model</em></td>
        <td>{price(item.support)}</td><td>{price(item.resistance)}</td><td>{item.actual?.actual_direction || '—'}</td>
        <td><span className={`outcome ${item.actual?.outcome.toLowerCase() || 'pending'}`}>{item.actual?.outcome || 'PENDING'}</span></td>
        <td>{item.actual ? percent(item.actual.actual_return_pct, 3) : '—'}</td>
      </tr>)}</tbody></table>
      {!rows.length && <div className="empty-state table-empty">No prediction records are stored for this date and session.</div>}
    </div>
  </section>;
}
