import { useState } from 'react';
import type { MarketIntelligence } from '../../services/analysisApi';
import { formatTime, price } from '../../utils/format';

const text = (value: unknown) => value == null || value === '' ? '—' : String(value);
const number = (value: unknown) => typeof value === 'number' && Number.isFinite(value) ? value : null;

export function MarketIntelligencePanel({ data, error = '', analysisTimestamp }: { data: MarketIntelligence | null; error?: string; analysisTimestamp?: string | null }) {
  const orb = data?.opening_range || {};
  const rvol = data?.rvol || {};
  const levels = data?.previous_day_levels || {};
  const distance = levels.distance && typeof levels.distance === 'object' ? levels.distance as Record<string, unknown> : {};
  const timestampsMatch = !analysisTimestamp || !data?.latest_completed_candle ||
    new Date(analysisTimestamp).getTime() === new Date(data.latest_completed_candle).getTime();
  return <section className="terminal-panel market-intel-panel">
    <div className="terminal-section-head"><div><span className="terminal-kicker">COMPLETED CANDLE EVIDENCE</span><h3>Market structure</h3></div><span className="terminal-badge">{data?.trend_alignment.overall || 'LOADING'}</span></div>
    {error && <p className="market-intel-error">{error}</p>}
    <div className="market-trend-row">{(data?.trend_alignment.timeframes || []).map((item) => <div key={String(item.timeframe)}><small>{String(item.timeframe)} TREND</small><b className={`trend-${String(item.trend).toLowerCase()}`}>{String(item.trend)}</b><span>{typeof item.bars === 'number' ? `${item.bars} complete bars` : 'Need more history'}</span></div>)}</div>
    <p className="market-intel-explanation">{data?.trend_alignment.explanation || 'Waiting for completed market data.'} Alignment is context only and does not change the decision-engine score.</p>
    {!timestampsMatch && <p className="market-intel-error">Evidence timestamps differ: market structure is as of {formatTime(data?.latest_completed_candle, true)}; decision analysis is as of {formatTime(analysisTimestamp, true)}.</p>}
    <div className="market-intel-grid">
      <div><small>15M OPENING RANGE</small><b>{text(orb.classification || orb.status)}</b><span>High {price(number(orb.high))} · Low {price(number(orb.low))}</span><span>Mid {price(number(orb.midpoint))}</span></div>
      <div title={text(rvol.formula)}><small>HISTORICAL SAME-SLOT RVOL</small><b>{number(rvol.rvol) == null ? text(rvol.classification) : `${number(rvol.rvol)!.toFixed(2)}× · ${text(rvol.classification)}`}</b><span>{text(rvol.current_volume)} current ÷ {number(rvol.average_same_slot_volume) == null ? '—' : number(rvol.average_same_slot_volume)!.toFixed(2)} avg · {text(rvol.sample_count)} prior sessions · {text(rvol.slot_et)} ET</span><span>As of {formatTime(typeof rvol.candle_timestamp === 'string' ? rvol.candle_timestamp : null, true)}</span></div>
      <div><small>PREVIOUS SESSION · {text(levels.session_date)}</small><b>H {price(number(levels.high))} · L {price(number(levels.low))}</b><span>C {price(number(levels.close))} · Δ close {number(distance.close) == null ? '—' : price(number(distance.close))}</span></div>
    </div>
    <div className="market-intel-foot">As of completed candle {formatTime(data?.latest_completed_candle, true)}. No decision weights changed. {data?.trend_alignment.conflict_warning ? 'Timeframe conflict is present.' : ''}</div>
  </section>;
}
