import { useState } from 'react';
import type { AnalysisPayload, DataSnapshotStatus, MarketSnapshot, SignalRecord } from '../../types/analysis';
import { formatDate, formatTime, price } from '../../utils/format';

const n = (value: unknown, digits = 2) => typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';
const componentLabels: Record<string, string> = { trend: 'Moving average structure evidence', momentum: 'RSI / MACD momentum', volume: 'Signed session rolling volume evidence', structure: 'Market structure', price_action: 'Candle price action', ml: 'Production ML evidence' };
const etDateFormatter = new Intl.DateTimeFormat('en-CA', { timeZone: 'America/New_York', year: 'numeric', month: '2-digit', day: '2-digit' });

function MarketStatistics({ snapshot, status, payload }: { snapshot: MarketSnapshot | null; status: DataSnapshotStatus | null; payload?: AnalysisPayload }) {
  const candles = snapshot?.candles || [];
  const latest = candles.at(-1);
  const sessionDate = latest ? etDateFormatter.format(new Date(latest.timestamp)) : '';
  const sessionBars = candles.filter((c) => etDateFormatter.format(new Date(c.timestamp)) === sessionDate);
  const high = sessionBars.length ? Math.max(...sessionBars.map((c) => c.high)) : null;
  const low = sessionBars.length ? Math.min(...sessionBars.map((c) => c.low)) : null;
  let pv = 0; let volume = 0;
  for (const c of sessionBars) { pv += ((c.high + c.low + c.close) / 3) * c.volume; volume += c.volume; }
  const sessionVwap = volume ? pv / volume : null;
  const indicators = payload?.indicators || {};
  const analysisTime = payload?.candle_timestamp ? new Date(payload.candle_timestamp).getTime() : NaN;
  const analysisDate = payload?.candle_timestamp ? etDateFormatter.format(new Date(payload.candle_timestamp)) : '';
  const analysisSessionBars = Number.isFinite(analysisTime) && analysisDate
    ? candles.filter((c) => etDateFormatter.format(new Date(c.timestamp)) === analysisDate && new Date(c.timestamp).getTime() <= analysisTime)
    : [];
  const volumeSampleCount = typeof indicators.volume_observation_count_20 === 'number'
    ? indicators.volume_observation_count_20
    : analysisSessionBars.some((c) => new Date(c.timestamp).getTime() === analysisTime)
      ? Math.min(20, analysisSessionBars.length) : null;
  const items: Array<[string, string]> = [
    ['Last completed close', price(latest?.close)], ['Session high / low', high == null || low == null ? '—' : `${price(high)} / ${price(low)}`],
    ['Session range', high == null || low == null ? '—' : price(high - low)], ['Session VWAP*', price(sessionVwap)],
    ['VWAP distance', latest && sessionVwap != null ? `${n((latest.close / sessionVwap - 1) * 100)}%` : '—'],
    ['RSI 14', n(indicators.rsi_14, 1)], ['MACD', n(indicators.macd, 3)],
    ['ATR %', indicators.atr_pct == null ? '—' : `${n(indicators.atr_pct, 2)}%`],
    ['Session rolling volume ratio', indicators.relative_volume == null ? '—' : `${n(indicators.relative_volume, 2)}×`],
    ['Volatility regime', String(indicators.volatility_classification || '—')],
    ['Market regime', String(payload?.market_regime?.classification || '—')],
    ['Feed', `${snapshot?.feed || status?.feed || 'Unavailable'}`],
    ['Latest market candle', formatTime(snapshot?.latest_timestamp || status?.latest_completed_rth_candle, true)],
    ['Analysis input candle', formatTime(payload?.candle_timestamp, true)],
  ];
  return <section className="terminal-panel market-stats-panel">
    <div className="terminal-section-head"><div><span className="terminal-kicker">VERIFIED INPUTS</span><h3>Market statistics</h3></div><span className="terminal-badge">{sessionDate ? formatDate(sessionDate) : 'NO SESSION'}</span></div>
    <div className="stat-rows">{items.map(([label, value]) => <div className="stat-row" key={label}><span>{label}</span><b>{value}</b></div>)}</div>
    <p className="tiny-disclaimer">Session rolling volume ratio = current 5-minute candle volume ÷ mean of up to 20 observed completed bars in the same ET session (includes current bar). It is not same-time-slot historical RVOL. Analysis indicator values are as of {formatTime(payload?.candle_timestamp, true)}. Session VWAP uses actual OHLCV bars.</p>
    {indicators.relative_volume != null && <p className="tiny-disclaimer">Volume breakdown: {n(indicators.volume, 0)} current shares ÷ {n(indicators.volume_sma_20, 2)} same-session mean across {n(volumeSampleCount, 0)} observed bars.</p>}
  </section>;
}

export function AnalysisPanel({ record, market, dataStatus, fresh, marketClosed = false, latestAligned }: {
  record: SignalRecord | null; market: MarketSnapshot | null; dataStatus: DataSnapshotStatus | null; fresh: boolean; marketClosed?: boolean; latestAligned: boolean;
}) {
  const payload = record?.payload;
  const ml = payload?.ml_evidence;
  const risk = payload?.risk_plan;
  const components = Object.entries(payload?.component_contributions || {}).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const scale = Math.max(.0001, ...components.map(([, value]) => Math.abs(value)));
  const classification = record?.classification || payload?.classification || 'UNAVAILABLE';
  const decisionClass = classification === 'BUY' ? 'decision-buy' : classification === 'SELL' ? 'decision-sell' : classification === 'HOLD' ? 'decision-hold' : 'decision-wait';
  return <div className="analysis-column">
    <section className="terminal-panel analysis-main-panel">
      <div className="terminal-section-head"><div><span className="terminal-kicker">AI MARKET ANALYSIS</span><h2>Decision engine</h2></div><span className={`terminal-badge ${fresh ? 'badge-good' : marketClosed ? 'badge-muted' : 'badge-warn'}`}>{fresh ? 'INPUT FRESH' : marketClosed ? 'MARKET CLOSED' : 'INPUT STALE'}</span></div>
      {!record || !payload ? <div className="terminal-empty"><b>Analysis unavailable</b><span>No stored analysis record was returned. Use “Analyze latest candle” to request one against the local completed-bar snapshot.</span></div> : <>
        {!latestAligned && <div className="inline-warning">Stored analysis is for {formatTime(record.candle_timestamp, true)}. It does not match the latest available market candle.</div>}
        <div className={`decision-banner ${decisionClass}`}>
          <span>FINAL DECISION</span><strong>{classification}</strong>
          <div className="decision-meta"><span>Final decision-engine score <b>{n(payload.final_decision_score, 3)}</b></span></div>
          <small>Weighted heuristic score · not a probability or confidence</small>
        </div>
        <div className="direction-breakdown" aria-label="Decision evidence layers">
          <div><small>MOVING AVERAGE STRUCTURE</small><b>{String(payload.indicators?.trend_classification || 'Not reported')}</b><span>EMA 9/20/50 ordering and slopes · as of {formatTime(record.candle_timestamp, true)}</span></div>
          <div><small>SHORT-TERM DIRECTIONAL EVIDENCE</small><b>{payload.directional_bias || 'Not reported'}</b><span>Technical score {n(payload.technical_score, 3)}</span></div>
          <div><small>ML DIRECTIONAL SIGNAL</small><b>{ml?.available ? ml.direction || 'Not reported' : 'Unavailable'}</b><span>Raw score shown separately below</span></div>
          <div><small>FINAL DECISION</small><b>{classification}</b><span>Decision-engine result</span></div>
        </div>
        <div className="analysis-meta-grid">
          <div><small>MARKET REGIME</small><b>{payload.market_regime?.classification || '—'}</b></div>
          <div><small>ANALYSIS GENERATED</small><b>{formatTime(record.generated_at, true)}</b></div>
          <div><small>ANALYSIS INPUT CANDLE</small><b>{formatTime(record.candle_timestamp, true)}</b><span className="meta-subvalue">Close {price(typeof payload.indicators?.close === 'number' ? payload.indicators.close : null)}</span></div>
          <div><small>RECORD STATUS</small><b className="capitalize">{record.status.replaceAll('_', ' ')}</b></div>
        </div>
        <div className="ml-signal-card">
          <div className="ml-heading"><div><span className="terminal-kicker">FROZEN PRODUCTION MODEL</span><b>{ml?.available ? ml.direction : 'ML UNAVAILABLE'}</b></div><span className="terminal-badge">{ml?.model_version || 'MODEL UNKNOWN'}</span></div>
          <div className="ml-score-row"><span>RAW ML SCORE — NOT CALIBRATED CONFIDENCE</span><b>{ml?.available && ml.raw_probability_up != null ? `${(ml.raw_probability_up * 100).toFixed(1)}% P(UP)` : '—'}</b></div>
          <div className="ml-score-track"><i style={{ width: `${ml?.available && ml.raw_probability_up != null ? Math.max(0, Math.min(100, ml.raw_probability_up * 100)) : 0}%` }} /></div>
          <small>{ml?.model_name || '—'} · {ml?.input_feature_count ?? '—'} ordered inputs · model output is directional evidence only</small>
          {!ml?.available && ml?.reason && <small className="negative-copy">{ml.reason}</small>}
        </div>
        {payload.conflicting_signals?.length ? <div className="inline-warning"><b>Trend and immediate evidence may disagree</b><ul>{payload.conflicting_signals.map((item) => <li key={item}>{item}</li>)}</ul></div> : null}
        <div className="evidence-grid">
          <Evidence title="Bullish evidence" entries={payload.bullish_evidence || []} kind="positive" indicators={payload.indicators} />
          <Evidence title="Bearish evidence" entries={payload.bearish_evidence || []} kind="negative" indicators={payload.indicators} />
          <Evidence title="Neutral observations" entries={payload.neutral_observations || []} kind="neutral" indicators={payload.indicators} />
        </div>
        <details className="reason-block"><summary><span className="terminal-kicker">DECISION EXPLANATION · BACKEND REASONS</span></summary>{(payload.reasons || []).map((reason) => <p key={reason}>{reason}</p>)}
          {payload.required_confirmation && <p><b>Required confirmation:</b> {payload.required_confirmation}</p>}
        </details>
        <div className="contribution-block"><div className="terminal-section-head"><div><span className="terminal-kicker">ENGINE COMPONENTS</span><h3>Decision contributions</h3></div><span className="subtle-label">not ML feature importance</span></div>
          {components.length ? components.map(([name, value]) => <div className="contribution-row" key={name}>
            <span>{componentLabels[name] || name.replaceAll('_', ' ')}</span><div className="contribution-track"><i className={value >= 0 ? 'positive-fill' : 'negative-fill'} style={{ width: `${Math.abs(value) / scale * 100}%` }} /></div><b className={value >= 0 ? 'positive-copy' : 'negative-copy'}>{value >= 0 ? '+' : ''}{value.toFixed(3)}</b>
          </div>) : <span className="muted-copy">Contributions are unavailable.</span>}
        </div>
        <div className="risk-block"><div className="terminal-section-head"><div><span className="terminal-kicker">HYPOTHETICAL TRADE SETUP</span><h3>Risk management</h3></div><span className={`terminal-badge ${risk?.valid ? 'badge-good' : 'badge-muted'}`}>{risk?.valid ? 'VALID PLAN' : 'NOT ACTIONABLE'}</span></div>
          {risk?.valid ? <>
            <div className="risk-level-grid">
              <Level title="TRIGGER PRICE" value={price(risk.trigger_price)} />
              <Level title="ENTRY ZONE" value={risk.entry_zone ? `${price(risk.entry_zone.lower)} – ${price(risk.entry_zone.upper)}` : '—'} />
              <Level title="STOP LOSS" value={price(risk.stop_loss)} tone="negative-copy" />
              <Level title="TARGET 1" value={price(risk.target_1)} tone="positive-copy" />
              <Level title="TARGET 2" value={price(risk.target_2)} tone="positive-copy" />
              <Level title="TARGET 2 R/R" value={risk.risk_reward_ratio == null ? '—' : `${risk.risk_reward_ratio.toFixed(2)} : 1`} />
            </div>
            <small>ATR {price(risk.atr_reference)} · stop: {risk.stop_method || '—'} · target: {risk.target_method || '—'}</small>
          </> : <div className="terminal-empty compact"><span>{risk?.reason || 'No approved hypothetical trade plan for this decision.'}</span>{payload.setup_invalidation && <small>Invalidation: {payload.setup_invalidation}</small>}</div>}
        </div>
      </>}
    </section>
    <MarketStatistics snapshot={market} status={dataStatus} payload={payload} />
  </div>;
}

function Evidence({ title, entries, kind, indicators }: { title: string; entries: string[]; kind: string; indicators?: AnalysisPayload['indicators'] }) {
  const [expanded, setExpanded] = useState(false);
  const visible = expanded ? entries : entries.slice(0, 3);
  const present = (item: string) => {
    if (item.startsWith('Trend alignment is ')) {
      return `Moving average structure is ${String(indicators?.trend_classification || 'unavailable')} (EMA 9/20/50 ordering and slopes); this is separate from 1H/15M/5M trend alignment.`;
    }
    return item.replace(/^Relative volume=/, 'Session rolling volume ratio=');
  };
  return <section className={`evidence-box ${kind}`}><h4>{title}</h4>{entries.length ? <><ul>{visible.map((item) => <li key={item}>{present(item)}</li>)}</ul>{entries.length > 3 && <button className="evidence-expand" onClick={() => setExpanded((value) => !value)}>{expanded ? 'Show less' : `Show all ${entries.length}`}</button>}</> : <p>No items returned.</p>}</section>;
}

function Level({ title, value, tone = '' }: { title: string; value: string; tone?: string }) {
  return <div className="risk-level"><small>{title}</small><b className={tone}>{value}</b></div>;
}
