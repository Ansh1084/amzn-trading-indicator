import { useEffect, useMemo, useRef, useState } from 'react';
import { analysisApi } from '../../services/analysisApi';
import type { SignalRecord } from '../../types/analysis';
import { formatTime, percent, price } from '../../utils/format';

const PAGE_SIZE = 25;
const toEtDate = (value: string) => new Intl.DateTimeFormat('en-CA', { timeZone: 'America/New_York', year: 'numeric', month: '2-digit', day: '2-digit' }).format(new Date(value));
const statusLabel = (value: string) => value.replaceAll('_', ' ');
const signalType = (value: string) => ['BUY', 'SELL', 'HOLD'].includes(value) ? value : value === 'WAIT FOR CONFIRMATION' ? 'WAIT' : value;

export function SignalStream({ signals, onSignals, onSelect }: {
  signals: SignalRecord[]; onSignals: (rows: SignalRecord[], append: boolean) => void; onSelect: (record: SignalRecord) => void;
}) {
  const [classification, setClassification] = useState('ALL');
  const [status, setStatus] = useState('ALL');
  const [date, setDate] = useState('');
  const [search, setSearch] = useState('');
  const [ascending, setAscending] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);
  const [page, setPage] = useState(0);
  const [hasMore, setHasMore] = useState(signals.length >= 100);
  const [loadingMore, setLoadingMore] = useState(false);
  const [expandedId, setExpandedId] = useState('');
  const [detail, setDetail] = useState<SignalRecord | null>(null);
  const [detailError, setDetailError] = useState('');
  const scroller = useRef<HTMLDivElement>(null);
  const olderRequested = useRef(false);
  const statuses = useMemo(() => Array.from(new Set(signals.map((item) => item.status))).sort(), [signals]);

  const filtered = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return signals.filter((record) => {
      const type = signalType(record.classification);
      const matchesType = classification === 'ALL' || type === classification;
      const matchesStatus = status === 'ALL' || record.status === status;
      const matchesDate = !date || toEtDate(record.candle_timestamp) === date;
      const text = [record.signal_id, record.classification, record.directional_bias, record.status, ...(record.payload.reasons || [])].join(' ').toLowerCase();
      return matchesType && matchesStatus && matchesDate && (!needle || text.includes(needle));
    }).sort((a, b) => ascending
      ? a.candle_timestamp.localeCompare(b.candle_timestamp)
      : b.candle_timestamp.localeCompare(a.candle_timestamp));
  }, [signals, classification, status, date, search, ascending]);
  const pages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const rows = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  useEffect(() => { setPage(0); }, [classification, status, date, search]);
  useEffect(() => { if (!olderRequested.current && signals.length) setHasMore(signals.length >= 100); }, [signals.length]);
  useEffect(() => { if (autoScroll && scroller.current) scroller.current.scrollTop = 0; }, [signals.length, autoScroll]);

  const loadOlder = async () => {
    setLoadingMore(true);
    try {
      olderRequested.current = true;
      const response = await analysisApi.signals(signals.length, 100);
      const unique = response.signals.filter((item) => !signals.some((current) => current.signal_id === item.signal_id));
      onSignals(unique, true);
      setHasMore(response.signals.length === 100);
    } catch (error) {
      setDetailError(error instanceof Error ? error.message : 'Unable to load older signals.');
    } finally { setLoadingMore(false); }
  };

  const toggleDetails = async (record: SignalRecord) => {
    onSelect(record);
    setDetailError('');
    if (expandedId === record.signal_id) { setExpandedId(''); setDetail(null); return; }
    setExpandedId(record.signal_id);
    setDetail(record);
    try { setDetail(await analysisApi.signal(record.signal_id)); }
    catch (error) { setDetailError(error instanceof Error ? error.message : 'Full signal details are unavailable.'); }
  };

  return <section className="terminal-panel signal-stream-panel">
    <div className="terminal-section-head signal-stream-heading">
      <div><span className="terminal-kicker">PERSISTED DECISIONS · HYPOTHETICAL ONLY</span><h2>Algorithmic Signal Stream</h2><p>Stored records from the analysis API. Select a row to inspect its explanation and locate its actual candle.</p></div>
      <div className="stream-count">{filtered.length.toLocaleString()} <span>shown of {signals.length.toLocaleString()} loaded</span></div>
    </div>
    <div className="signal-filter-bar">
      <label className="stream-search"><span>⌕</span><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search reason, signal ID, status…" aria-label="Search signals" /></label>
      <div className="classification-filter" role="group" aria-label="Filter by classification">{['ALL', 'BUY', 'SELL', 'HOLD', 'WAIT'].map((item) => <button key={item} className={classification === item ? 'active' : ''} onClick={() => setClassification(item)}>{item}</button>)}</div>
      <label className="filter-select"><span>STATUS</span><select value={status} onChange={(event) => setStatus(event.target.value)}><option value="ALL">All statuses</option>{statuses.map((item) => <option value={item} key={item}>{statusLabel(item)}</option>)}</select></label>
      <label className="filter-select"><span>ET DATE</span><input type="date" value={date} onChange={(event) => setDate(event.target.value)} /></label>
      <label className="compact-check"><input type="checkbox" checked={autoScroll} onChange={(event) => setAutoScroll(event.target.checked)} />Auto-scroll</label>
      <button className="terminal-icon-button" onClick={() => setAscending((old) => !old)} title="Sort by candle timestamp">{ascending ? 'Oldest first ↑' : 'Newest first ↓'}</button>
    </div>
    {detailError && <div className="inline-warning stream-warning">{detailError}</div>}
    <div className="signal-table-scroll" ref={scroller}>
      <table className="signal-table"><thead><tr>
        <th>Timestamp</th><th>Classification</th><th>Bias</th><th>ML raw / decision</th><th>Trigger</th><th>Entry zone</th><th>Target / stop</th><th>Dominant drivers</th><th>Status</th><th>Hypothetical P&amp;L</th>
      </tr></thead><tbody>
        {rows.map((record) => {
          const risk = record.payload.risk_plan;
          const drivers = Object.entries(record.payload.component_contributions || {}).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 2);
          const expanded = expandedId === record.signal_id;
          const raw = record.payload.ml_evidence?.raw_probability_up;
          const actionable = ['BUY', 'SELL'].includes(record.classification);
          return <FragmentRow key={record.signal_id} record={record} risk={risk} drivers={drivers} expanded={expanded} detail={detail} raw={raw} actionable={actionable} onToggle={() => void toggleDetails(record)} />;
        })}
        {!rows.length && <tr><td colSpan={10} className="empty-table-cell">{signals.length ? 'No persisted signals match these filters.' : 'No analysis signals have been persisted yet. Load existing local analysis records or run an explicit analysis.'}</td></tr>}
      </tbody></table>
    </div>
    <div className="stream-pagination">
      <button className="terminal-icon-button" disabled={page === 0} onClick={() => setPage((p) => Math.max(0, p - 1))}>← Previous</button><span>Page {page + 1} of {pages}</span><button className="terminal-icon-button" disabled={page + 1 >= pages} onClick={() => setPage((p) => Math.min(pages - 1, p + 1))}>Next →</button>
      <button className="terminal-icon-button load-older" disabled={!hasMore || loadingMore} onClick={() => void loadOlder()}>{loadingMore ? 'Loading…' : hasMore ? 'Load older records' : 'No older records loaded'}</button>
    </div>
    <div className="signal-table-footnote">Source: persisted SQLite analysis signals · 5-minute completed candles · price levels and returns are hypothetical. No broker fills or realized P&amp;L are represented.</div>
  </section>;
}

function FragmentRow({ record, risk, drivers, expanded, detail, raw, actionable, onToggle }: {
  record: SignalRecord; risk?: SignalRecord['payload']['risk_plan']; drivers: Array<[string, number]>; expanded: boolean; detail: SignalRecord | null; raw?: number | null; actionable: boolean; onToggle: () => void;
}) {
  const score = record.final_score ?? record.payload.final_decision_score;
  const gross = record.gross_return_pct;
  const net = record.net_return_pct;
  const outcomePending = gross == null && net == null;
  return <>
    <tr className={`signal-row ${expanded ? 'selected-row' : ''}`} onClick={onToggle} tabIndex={0} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') onToggle(); }}>
      <td><button className="table-link" onClick={(event) => { event.stopPropagation(); onToggle(); }}>{formatTime(record.candle_timestamp, true)}<small>generated {formatTime(record.generated_at)}</small></button></td>
      <td><span className={`classification-badge ${record.classification === 'BUY' ? 'buy' : record.classification === 'SELL' ? 'sell' : record.classification === 'HOLD' ? 'hold' : 'wait'}`}>{record.classification === 'BUY' ? '↑ BUY / LONG' : record.classification === 'SELL' ? '↓ SELL / SHORT' : signalType(record.classification)}</span></td>
      <td className={record.directional_bias === 'BULLISH' ? 'positive-copy' : record.directional_bias === 'BEARISH' ? 'negative-copy' : ''}>{record.directional_bias || '—'}</td>
      <td><span title="Model score is uncalibrated">{raw == null ? 'ML —' : `${(raw * 100).toFixed(1)}% P(UP)`}</span><small className="cell-secondary">Decision {score == null ? '—' : score.toFixed(3)}</small></td>
      <td>{price(risk?.trigger_price ?? record.payload.indicators?.close as number | undefined)}</td>
      <td>{risk?.valid && risk.entry_zone ? <>{price(risk.entry_zone.lower)}<small className="cell-secondary">to {price(risk.entry_zone.upper)}</small></> : '—'}</td>
      <td>{risk?.valid ? <>{price(risk.target_2)}<small className="cell-secondary">/ SL {price(risk.stop_loss)}</small></> : '—'}</td>
      <td className="driver-cell">{drivers.length ? drivers.map(([name, value]) => <span key={name} className={value >= 0 ? 'positive-copy' : 'negative-copy'}>{name.replaceAll('_', ' ')} {value >= 0 ? '+' : ''}{value.toFixed(2)}</span>) : '—'}</td>
      <td><span className={`status-badge ${record.status.toLowerCase()}`}>{statusLabel(record.status)}</span></td>
      <td>{outcomePending ? <span className="pending-copy">PENDING</span> : <span className={Number(net) >= 0 ? 'positive-copy' : 'negative-copy'}>Gross {percent(gross)}<small className="cell-secondary">Net {percent(net)}</small></span>}{outcomePending && actionable && <small className="cell-secondary">no resolved return</small>}</td>
    </tr>
    {expanded && <tr className="signal-detail-row"><td colSpan={10}><div className="signal-detail-content">
      <div className="signal-detail-top"><b>{record.signal_id}</b><span>{record.payload.analysis_version}</span><span>{formatTime(record.payload.candle_timestamp, true)} candle · {formatTime(record.payload.candle_close_timestamp, true)} complete</span></div>
      <div className="signal-detail-grid"><div><h4>Decision reasons</h4>{(record.payload.reasons || []).map((reason) => <p key={reason}>{reason}</p>)}<p><b>Confirmation:</b> {record.payload.required_confirmation || '—'}</p><p><b>Invalidation:</b> {record.payload.setup_invalidation || '—'}</p></div>
        <div><h4>Evidence and state</h4><p><b>Bullish:</b> {(record.payload.bullish_evidence || []).join(' · ') || '—'}</p><p><b>Bearish:</b> {(record.payload.bearish_evidence || []).join(' · ') || '—'}</p><p><b>Regime:</b> {record.payload.market_regime?.classification || '—'} · <b>ML:</b> {record.payload.ml_evidence?.direction || 'unavailable'} (raw score, uncalibrated)</p><p><b>Setup:</b> {risk?.valid ? `${price(risk.entry_zone?.lower)}–${price(risk.entry_zone?.upper)}, stop ${price(risk.stop_loss)}, T1 ${price(risk.target_1)}, T2 ${price(risk.target_2)}` : risk?.reason || 'No actionable setup'}</p></div>
      </div>
      {detail?.events?.length ? <div className="event-strip"><h4>Lifecycle events</h4>{detail.events.map((event, index) => <span key={`${event.event_type}-${index}`}>{event.event_type} · {formatTime(event.timestamp, true)}</span>)}</div> : null}
      <small className="signal-detail-note">{record.payload.hypothetical_only ? 'Hypothetical analysis only. No broker execution occurred.' : 'Persisted analysis record.'} {detail && detail.signal_id !== record.signal_id ? 'Detail loading…' : ''}</small>
    </div></td></tr>}
  </>;
}
