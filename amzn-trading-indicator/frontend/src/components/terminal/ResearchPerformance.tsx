import { researchPerformance as research } from '../../data/researchPerformance';
import { formatDate, percent } from '../../utils/format';

export function ResearchPerformance() {
  const combined = research.strategies[0];
  return <main className="research-page">
    <div className="research-title"><div><span className="terminal-kicker">CHRONOLOGICAL LOCAL REPLAY · RESEARCH ONLY</span><h1>Historical performance</h1><p>{formatDate(research.period.start)} – {formatDate(research.period.end)} ET · {research.source}</p></div><span className="research-warning">NOT A PROFITABILITY CLAIM</span></div>
    <div className="research-disclaimer"><b>Historical hypothetical results — not live trading performance.</b><span>Same-session next 5-minute open fills only when the open is inside the planned entry zone. One share per setup, additive returns, no compounding. Net scenario assumes 10 bps round-trip cost. Bar-level stop-first ordering is conservative; actual intrabar order and fills are unknown.</span></div>
    <section className="research-kpi-grid">
      <Metric label="Valid analyses" value={research.validAnalyses.toLocaleString()} detail="after indicator warm-up" />
      <Metric label="Actionable BUY / SELL" value={(research.signalCounts.BUY + research.signalCounts.SELL).toLocaleString()} detail={`${research.signalCounts.BUY.toLocaleString()} BUY · ${research.signalCounts.SELL.toLocaleString()} SELL`} />
      <Metric label="Resolved combined trades" value={combined.resolved.toLocaleString()} detail={`${combined.wins} net-positive · ${((combined.wins! / combined.resolved) * 100).toFixed(1)}% after assumed costs`} tone="neutral" />
      <Metric label="Gross / net profit factor" value={`${combined.grossProfitFactor.toFixed(3)} / ${combined.netProfitFactor.toFixed(3)}`} detail="gross / assumed-cost net" tone="negative" />
      <Metric label="Net additive return" value={percent(combined.netReturn, 2)} detail={`gross ${percent(combined.grossReturn, 2)} · not compounded`} tone="negative" />
      <Metric label="Maximum drawdown" value={`${combined.maxDrawdown.toFixed(2)} pp`} detail="peak-to-trough on additive net returns" tone="negative" />
    </section>
    <section className="terminal-panel performance-panel">
      <div className="terminal-section-head"><div><span className="terminal-kicker">COMMON ENTRY / EXIT RULES</span><h2>Strategy comparison</h2></div><span className="terminal-badge">{research.assumptions.transactionCostBpsRoundTrip} BPS ROUND TRIP ASSUMED</span></div>
      <div className="performance-table-wrap"><table className="performance-table"><thead><tr><th>Strategy</th><th>Setups</th><th>Resolved</th><th>Gross return sum</th><th>Net return sum</th><th>Max DD</th><th>Gross PF</th><th>Net PF</th><th>Avg net / trade</th></tr></thead><tbody>
        {research.strategies.map((row) => <tr key={row.name} className={row.name === combined.name ? 'highlight-row' : ''}><td>{row.name}{row.name === combined.name && <small>Decision engine</small>}</td><td>{row.setups.toLocaleString()}</td><td>{row.resolved.toLocaleString()}</td><td className={row.grossReturn >= 0 ? 'positive-copy' : 'negative-copy'}>{percent(row.grossReturn, 2)}</td><td className={row.netReturn >= 0 ? 'positive-copy' : 'negative-copy'}>{percent(row.netReturn, 2)}</td><td>{row.maxDrawdown.toFixed(2)} pp</td><td>{row.grossProfitFactor.toFixed(3)}</td><td className="negative-copy">{row.netProfitFactor.toFixed(3)}</td><td>{percent(row.avgNetReturn, 3)}</td></tr>)}
      </tbody></table></div>
    </section>
    <section className="terminal-panel performance-panel">
      <div className="terminal-section-head"><div><span className="terminal-kicker">COMBINED ENGINE ONLY</span><h2>Market-regime breakdown</h2></div></div>
      <div className="performance-table-wrap"><table className="performance-table"><thead><tr><th>Regime</th><th>Valid candles</th><th>Setups</th><th>Resolved</th><th>Net additive return</th><th>Max DD</th></tr></thead><tbody>
        {research.regimes.map((row) => <tr key={row.name}><td>{row.name}</td><td>{row.candles.toLocaleString()}</td><td>{row.setups.toLocaleString()}</td><td>{row.resolved.toLocaleString()}</td><td className={row.netReturn >= 0 ? 'positive-copy' : 'negative-copy'}>{percent(row.netReturn, 2)}</td><td>{row.maxDrawdown.toFixed(2)} pp</td></tr>)}
      </tbody></table></div>
    </section>
    <section className="terminal-panel methodology-panel"><div className="terminal-section-head"><div><span className="terminal-kicker">REPLAY METHODOLOGY</span><h2>Assumptions and limits</h2></div></div>
      <ul><li>{research.assumptions.entry}.</li><li>{research.assumptions.exit}.</li><li>{research.assumptions.positionSize}.</li><li>{research.assumptions.target1}.</li><li>All strategies use separate one-position-at-a-time ledgers. Gross and net totals are not a combined portfolio.</li><li>Costs are a fixed assumed scenario, not measured spread, commissions, slippage or market impact.</li><li>Results cover one local historical snapshot. They are not statistically validated, selected for future profitability, or indicative of actual broker outcomes.</li><li>The 10-minute session prediction workflow and this 5-minute analysis-engine replay are separate systems.</li></ul>
      <p>Source: <code>reports/experiments/trading_analysis_engine_audit/replay_metadata.json</code>, <code>trade_outcomes.csv</code>, and <code>regime_performance.csv</code>. The screen uses the saved research artifact; it is not a live performance endpoint.</p>
    </section>
  </main>;
}

function Metric({ label, value, detail, tone = '' }: { label: string; value: string; detail: string; tone?: string }) {
  return <div className={`research-metric ${tone}`}><small>{label}</small><b>{value}</b><span>{detail}</span></div>;
}
