import type { ForecastVersion, SessionId } from '../types';
import { formatDate } from '../utils/format';

export function SessionControls(props: {
  date: string; session: SessionId; version: ForecastVersion; updatedAvailable: boolean; busy: boolean;
  onDate: (date: string) => void; onSession: (session: SessionId) => void; onVersion: (version: ForecastVersion) => void;
  onGenerate: () => void; onLoad: () => void; onEvaluate: () => void; exportUrl?: string;
}) {
  const { date, session, version, updatedAvailable, busy } = props;
  return <section className="controls-panel panel">
    <div className="section-heading compact-heading"><div><span className="eyebrow">SESSION WORKSPACE</span><h2>Forecast controls</h2></div>
      <span className="version-tag">{version} · FROZEN</span></div>
    <div className="control-grid">
      <label>Trading date<input type="date" value={date} onChange={(e) => props.onDate(e.target.value)} /></label>
      <label>Session<select value={session} onChange={(e) => props.onSession(e.target.value as SessionId)}>
        <option value="morning">Morning · 09:30–12:30 ET</option><option value="afternoon">Afternoon · 12:30–15:30 ET</option>
      </select></label>
      <label>Prediction version<select value={version} onChange={(e) => props.onVersion(e.target.value as ForecastVersion)}>
        <option value="ADVANCE">ADVANCE</option>{session === 'afternoon' && updatedAvailable && <option value="UPDATED">UPDATED</option>}
      </select></label>
    </div>
    <div className="button-row">
      <button className="button primary" disabled={busy || !date} onClick={props.onGenerate}>{busy ? 'Working…' : version === 'UPDATED' ? 'Generate UPDATED Afternoon' : `Predict ${session === 'morning' ? 'Morning' : 'Afternoon'} Session`}</button>
      {session === 'afternoon' && updatedAvailable && <button className="button secondary" disabled={busy} onClick={() => { props.onVersion('UPDATED'); }}>Load UPDATED</button>}
      <button className="button secondary" disabled={busy} onClick={props.onLoad}>Load predictions</button>
      <button className="button secondary" disabled={busy} onClick={props.onEvaluate}>Refresh outcomes</button>
      {props.exportUrl && <a className="button ghost" href={props.exportUrl}>Export CSV</a>}
    </div>
    <p className="control-footnote">Selected: {formatDate(date)} · {session} · {version}. Outcomes update only after actual interval data is complete.</p>
  </section>;
}
