import { useCallback, useEffect, useMemo, useState } from 'react';
import { analysisApi } from '../../services/analysisApi';
import type { PaperDashboard, PaperSettings, PaperTrade } from '../../services/analysisApi';
import { formatTime, price } from '../../utils/format';

const dollars = (value: unknown) => typeof value === 'number' && Number.isFinite(value)
  ? `${value < 0 ? '−' : ''}$${Math.abs(value).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : '—';
const metric = (value: unknown, digits = 2) => typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';

export function PaperTrading() {
  const [data, setData] = useState<PaperDashboard | null>(null);
  const [draft, setDraft] = useState<PaperSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');

  const reload = useCallback(async () => {
    try {
      const value = await analysisApi.paperDashboard();
      setData(value);
      setDraft((current) => current || value.settings);
      setError('');
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Unable to load the paper ledger.'); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => {
    void reload();
    const timer = window.setInterval(() => void reload(), 15_000);
    return () => window.clearInterval(timer);
  }, [reload]);

  const save = async (values: Partial<PaperSettings>, message: string) => {
    setBusy(true); setError(''); setNotice('');
    try {
      const saved = await analysisApi.paperSettings(values);
      setDraft(saved); setNotice(message); await reload();
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Paper settings could not be saved.'); }
    finally { setBusy(false); }
  };

  const close = async (trade: PaperTrade) => {
    setBusy(true); setError(''); setNotice('');
    try {
      const result = await analysisApi.closePaperTrade(trade.trade_id);
      setNotice(`Simulated close recorded at ${price(result.trade.exit_price)} using the last completed candle at ${formatTime(result.fill_timestamp, true)}.`);
      await reload();
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Trade could not be closed.'); }
    finally { setBusy(false); }
  };

  const curve = useMemo(() => {
    const points = data?.equity_curve || [];
    if (points.length < 2) return '';
    const values = points.map((point) => point.equity);
    const lo = Math.min(...values); const hi = Math.max(...values); const span = Math.max(1, hi - lo);
    return values.map((value, i) => `${(i / (values.length - 1)) * 1000},${160 - ((value - lo) / span) * 145}`).join(' ');
  }, [data?.equity_curve]);

  if (loading && !data) return <main className="paper-page"><div className="terminal-empty">Loading persistent simulated account…</div></main>;
  const settings = draft || data?.settings;
  const stats = data?.metrics || {};
  const trades = data?.closed_trades || [];
  return <main className="paper-page">
    <section className="paper-page-heading">
      <div><span className="terminal-kicker">SIMULATION LEDGER · PERSISTENT SQLITE</span><h1>Paper Trading</h1><p>Simulated positions from actionable analysis signals. No orders are sent to a broker.</p></div>
      <b className="paper-disclaimer">{data?.disclaimer || 'PAPER TRADING — SIMULATED, NOT REAL BROKER EXECUTION'}</b>
    </section>
    {error && <div className="terminal-alert" role="alert">{error}</div>}{notice && <div className="terminal-notice" role="status">{notice}</div>}
    <section className="paper-settings terminal-panel">
      <div className="terminal-section-head"><div><span className="terminal-kicker">ACCOUNT CONTROLS</span><h2>{settings?.enabled ? 'Paper trading enabled' : 'Paper trading paused'}</h2></div>
        <span className={`terminal-badge ${settings?.enabled ? 'badge-good' : 'badge-muted'}`}>{settings?.enabled ? 'ENABLED' : 'DISABLED'}</span></div>
      <div className="paper-settings-grid">
        <label>Starting capital<input type="number" min="1" step="1000" value={settings?.starting_capital ?? ''} onChange={(e) => setDraft((s) => s ? { ...s, starting_capital: Number(e.target.value) } : s)} /></label>
        <label>Position sizing<select value={settings?.sizing_mode || 'fixed_shares'} onChange={(e) => setDraft((s) => s ? { ...s, sizing_mode: e.target.value as PaperSettings['sizing_mode'] } : s)}>
          <option value="fixed_shares">Fixed shares</option><option value="fixed_capital">Fixed capital</option><option value="equity_percent">Percent of equity</option></select></label>
        <label>{settings?.sizing_mode === 'fixed_shares' ? 'Shares per trade' : settings?.sizing_mode === 'equity_percent' ? 'Equity allocation (%)' : 'Capital per trade'}<input type="number" min="0.000001" step="any" value={settings?.sizing_value ?? ''} onChange={(e) => setDraft((s) => s ? { ...s, sizing_value: Number(e.target.value) } : s)} /></label>
        <label>Transaction costs (bps)<input type="number" min="0" max="500" step="1" value={settings?.transaction_cost_bps ?? ''} onChange={(e) => setDraft((s) => s ? { ...s, transaction_cost_bps: Number(e.target.value) } : s)} /></label>
        <label>Slippage (bps)<input type="number" min="0" max="500" step="1" value={settings?.slippage_bps ?? ''} onChange={(e) => setDraft((s) => s ? { ...s, slippage_bps: Number(e.target.value) } : s)} /></label>
      </div>
      <div className="paper-control-actions">
        <button className="terminal-primary-button" disabled={busy || !settings} onClick={() => void save({ ...settings }, 'Paper account settings saved.')}>Save settings</button>
        <button className={settings?.enabled ? 'terminal-icon-button' : 'terminal-primary-button'} disabled={busy || !settings} onClick={() => void save({ enabled: !settings?.enabled }, settings?.enabled ? 'Paper trading paused. Existing open trades continue to be marked, but no new trades will be queued.' : 'Paper trading enabled for future fresh actionable signals.')}>{settings?.enabled ? 'Pause paper trading' : 'Enable paper trading'}</button>
      </div>
      <p className="paper-note">Entry simulation waits for the next same-session completed 5-minute candle and fills only when its open is inside the planned entry zone. Existing open trades can still be exited while paused.</p>
    </section>
    <section className="paper-kpi-grid">
      {[["ACCOUNT EQUITY", dollars(stats.equity)], ["STARTING CAPITAL", dollars(settings?.starting_capital)], ["REALIZED P&L", dollars(stats.realized_pnl)], ["UNREALIZED P&L", dollars(stats.unrealized_pnl)], ["TOTAL NET P&L", dollars(stats.net_pnl)], ["WIN RATE", typeof stats.win_rate === 'number' ? `${(stats.win_rate * 100).toFixed(1)}%` : '—'], ["PROFIT FACTOR", metric(stats.profit_factor)], ["MAX DRAWDOWN", typeof stats.maximum_drawdown === 'number' ? `${(stats.maximum_drawdown * 100).toFixed(2)}%` : '—'], ["AVG R-MULTIPLE", metric(stats.average_r_multiple)]].map(([label, value]) =>
        <article key={label} className="paper-metric"><small>{label}</small><b>{value}</b></article>)}
    </section>
    <section className="terminal-panel paper-section"><div className="terminal-section-head"><div><span className="terminal-kicker">ACCOUNT VALUE HISTORY</span><h2>Equity curve</h2></div><span className="terminal-badge">CLOSED TRADE NET P&L</span></div>
      {data?.equity_curve && data.equity_curve.length > 1 ? <svg className="paper-equity-chart" viewBox="0 0 1000 180" role="img" aria-label="Paper account realized equity curve"><line x1="0" y1="165" x2="1000" y2="165" /><polyline points={curve} /></svg> : <div className="paper-empty">Equity curve will appear after simulated trades close. No sample or projected values are shown.</div>}
      <div className="paper-mark">Mark: <b>{price(data?.mark.price)}</b> · {data?.mark.source} · candle {formatTime(data?.mark.timestamp, true)} · {data?.mark.freshness === 'FRESH' ? 'FRESH' : 'LAST KNOWN / STALE'} · market {data?.mark.market_state || 'unknown'}</div>
    </section>
    <TradeTable title="Open positions" trades={data?.open_trades || []} close={close} busy={busy} open />
    <TradeTable title="Queued entries" trades={data?.pending_trades || []} busy={busy} />
    <TradeTable title="Closed trade history" trades={trades} busy={busy} />
    <section className="terminal-panel paper-section"><div className="terminal-section-head"><div><span className="terminal-kicker">PERSISTENT EVENT HISTORY</span><h2>Trade journal</h2></div><span className="terminal-badge">{data?.journal.length || 0} recent events</span></div>
      {data?.journal.length ? <div className="paper-journal">{data.journal.map((event) => <article key={event.journal_id}><time>{formatTime(event.timestamp, true)}</time><b>{event.event_type.replaceAll('_', ' ')}</b><span>{String(event.details.fill_price ?? event.details.mark_price ?? event.details.assumption ?? event.details.reason ?? 'Paper ledger event')}</span><code>{event.trade_id.slice(0, 8)}</code></article>)}</div> : <div className="paper-empty">No paper-trade events have been recorded.</div>}
    </section>
    <p className="paper-note">Historical research P&L is stored separately. Profit factor and drawdown are calculated from completed paper trades; unrealized P&L uses the last stored completed candle and is not a live quote when data is stale or the market is closed.</p>
  </main>;
}

function TradeTable({ title, trades, close, busy, open = false }: { title: string; trades: PaperTrade[]; close?: (trade: PaperTrade) => void; busy: boolean; open?: boolean }) {
  return <section className="terminal-panel paper-section"><div className="terminal-section-head"><div><span className="terminal-kicker">PERSISTED TRADE RECORDS</span><h2>{title}</h2></div><span className="terminal-badge">{trades.length}</span></div>
    <div className="paper-table-scroll"><table className="paper-table"><thead><tr><th>Direction / Setup</th><th>Status</th><th>Signal / Entry</th><th>Fill / Mark</th><th>Qty</th><th>Stop</th><th>Targets</th><th>{open ? 'Unrealized' : 'Net P&L'}</th><th>Exit</th>{open && <th />}</tr></thead>
      <tbody>{trades.length ? trades.map((trade) => <tr key={trade.trade_id}><td><b className={trade.direction === 'LONG' ? 'paper-long' : 'paper-short'}>{trade.direction}</b><small>{trade.setup_type}</small></td><td>{trade.status.replaceAll('_', ' ')}</td><td>{formatTime(trade.signal_timestamp, true)}<small>{trade.entry_timestamp ? `Entry ${formatTime(trade.entry_timestamp, true)}` : 'No fill yet'}</small></td><td>{price(trade.entry_price)}<small>{trade.entry_fill_price ? `Simulated fill ${price(trade.entry_fill_price)}` : 'Fill pending'}</small><small>{trade.mark_price ? `Mark ${price(trade.mark_price)}` : '—'}</small></td><td>{metric(trade.remaining_quantity, 4)} / {metric(trade.quantity, 4)}</td><td>{price(trade.stop_loss)}</td><td>{price(trade.target_1)} / {price(trade.target_2)}</td><td>{dollars(open ? trade.unrealized_net_pnl : trade.net_pnl)}</td><td>{trade.exit_reason ? <>{trade.exit_reason}<small>Reference {price(trade.exit_price)} · fill {price(trade.exit_fill_price)}</small><small>{formatTime(trade.exit_timestamp, true)}</small></> : '—'}</td>{open && <td><button className="terminal-icon-button" disabled={busy} onClick={() => close?.(trade)}>Close at last completed candle</button></td>}</tr>) : <tr><td colSpan={open ? 10 : 9} className="paper-empty">No records yet. Trades appear only for fresh actionable signals while paper trading is enabled.</td></tr>}</tbody></table></div>
  </section>;
}
