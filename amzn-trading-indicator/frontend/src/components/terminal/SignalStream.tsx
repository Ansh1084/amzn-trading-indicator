import { useEffect, useMemo, useRef, useState } from 'react';
import { analysisApi } from '../../services/analysisApi';
import type { SignalRecord } from '../../types/analysis';
import { formatTime, percent, price } from '../../utils/format';

const PAGE_SIZE = 25;

const toEtDate = (value: string) => {
  try {
    return new Intl.DateTimeFormat('en-CA', {
      timeZone: 'America/New_York',
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
    }).format(new Date(value));
  } catch {
    return '';
  }
};

function formatStatus(val: string | null | undefined): string {
  if (!val) return '—';
  return val
    .toLowerCase()
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

function normalizeSignal(val: string | null | undefined): 'BUY' | 'SELL' | 'HOLD' | 'WAIT' {
  const upper = String(val || '').toUpperCase();
  if (upper.includes('BUY')) return 'BUY';
  if (upper.includes('SELL')) return 'SELL';
  if (upper.includes('HOLD')) return 'HOLD';
  return 'WAIT';
}

const componentLabels: Record<string, string> = {
  trend: 'Moving average structure evidence',
  momentum: 'RSI / MACD momentum',
  volume: 'Signed session rolling volume evidence',
  structure: 'Market structure',
  price_action: 'Candle price action',
  ml: 'Production ML evidence',
};

function formatTraderEvidence(item: string): string {
  if (typeof item !== 'string') return String(item ?? '');

  let text = item;

  // 1. Rename Relative volume to Session Rolling Volume Ratio (Volume terminology fix)
  text = text.replace(/^Relative volume=([0-9.]+x?);?\s*/i, 'Session Rolling Volume Ratio=$1; ');
  text = text.replace(/Relative volume=/gi, 'Session Rolling Volume Ratio=');

  // 2. Trend alignment clarification
  if (text.startsWith('Trend alignment is ')) {
    text = text.replace(
      /^Trend alignment is ([a-z_]+)\s*\((.*?)\)\.?/i,
      (_, trend, details) =>
        `Moving average structure is ${formatStatus(trend)} (${details}); separate from 1H/15M/5M trend alignment.`
    );
    return text;
  }

  // 3. Structure sequence dictionary parser:
  // e.g.: Structure sequence={'higher_high': False, 'higher_low': True, 'lower_high': True, 'lower_low': False}; breakout=BREAKOUT_RETEST; breakdown=CONFIRMED_BREAKDOWN.
  if (text.includes('Structure sequence={')) {
    const seqMatch = text.match(/Structure sequence=\{([^}]+)\}/);
    const breakoutMatch = text.match(/breakout=([A-Z_]+)/i);
    const breakdownMatch = text.match(/breakdown=([A-Z_]+)/i);

    if (seqMatch) {
      const parts: string[] = [];
      const dictStr = seqMatch[1];

      const hh = dictStr.match(/'higher_high':\s*(True|False)/);
      const hl = dictStr.match(/'higher_low':\s*(True|False)/);
      const lh = dictStr.match(/'lower_high':\s*(True|False)/);
      const ll = dictStr.match(/'lower_low':\s*(True|False)/);

      if (hh) parts.push(`Higher High: ${hh[1] === 'True' ? 'Yes' : 'No'}`);
      if (hl) parts.push(`Higher Low: ${hl[1] === 'True' ? 'Yes' : 'No'}`);
      if (lh) parts.push(`Lower High: ${lh[1] === 'True' ? 'Yes' : 'No'}`);
      if (ll) parts.push(`Lower Low: ${ll[1] === 'True' ? 'Yes' : 'No'}`);

      let result = `Market Structure Sequence: ${parts.join(' · ')}`;

      if (breakoutMatch && breakoutMatch[1]) {
        result += ` · Breakout: ${breakoutMatch[1] === 'NONE' ? 'None' : formatStatus(breakoutMatch[1])}`;
      }

      if (breakdownMatch && breakdownMatch[1]) {
        result += ` · Breakdown: ${breakdownMatch[1] === 'NONE' ? 'None' : formatStatus(breakdownMatch[1])}`;
      }

      return result;
    }
  }

  // 4. Divergences formatting:
  text = text.replace(/bullish divergence=(True|False)/gi, (_, val) => `Bullish divergence: ${val === 'True' ? 'Yes' : 'No'}`);
  text = text.replace(/bearish divergence=(True|False)/gi, (_, val) => `Bearish divergence: ${val === 'True' ? 'Yes' : 'No'}`);

  return text;
}

export function SignalStream({
  signals,
  onSignals,
  onSelect,
}: {
  signals: SignalRecord[];
  onSignals: (rows: SignalRecord[], append: boolean) => void;
  onSelect: (record: SignalRecord) => void;
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
    return signals
      .filter((record) => {
        const type = normalizeSignal(record.classification);
        const matchesType = classification === 'ALL' || type === classification;
        const matchesStatus = status === 'ALL' || record.status === status;
        const matchesDate = !date || toEtDate(record.candle_timestamp) === date;
        const searchPool = [
          record.signal_id,
          record.classification,
          record.directional_bias,
          record.status,
          ...(record.payload?.reasons || []),
        ]
          .join(' ')
          .toLowerCase();
        return matchesType && matchesStatus && matchesDate && (!needle || searchPool.includes(needle));
      })
      .sort((a, b) =>
        ascending
          ? a.candle_timestamp.localeCompare(b.candle_timestamp)
          : b.candle_timestamp.localeCompare(a.candle_timestamp)
      );
  }, [signals, classification, status, date, search, ascending]);

  const pages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const rows = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  useEffect(() => {
    setPage(0);
  }, [classification, status, date, search]);

  useEffect(() => {
    if (!olderRequested.current && signals.length) setHasMore(signals.length >= 100);
  }, [signals.length]);

  useEffect(() => {
    if (autoScroll && scroller.current) scroller.current.scrollTop = 0;
  }, [signals.length, autoScroll]);

  const loadOlder = async () => {
    setLoadingMore(true);
    try {
      olderRequested.current = true;
      const response = await analysisApi.signals(signals.length, 100);
      const unique = response.signals.filter(
        (item) => !signals.some((current) => current.signal_id === item.signal_id)
      );
      onSignals(unique, true);
      setHasMore(response.signals.length === 100);
    } catch (error) {
      setDetailError(error instanceof Error ? error.message : 'Unable to load older signals.');
    } finally {
      setLoadingMore(false);
    }
  };

  const toggleDetails = async (record: SignalRecord) => {
    onSelect(record);
    setDetailError('');
    if (expandedId === record.signal_id) {
      setExpandedId('');
      setDetail(null);
      return;
    }
    setExpandedId(record.signal_id);
    setDetail(record);
    try {
      const full = await analysisApi.signal(record.signal_id);
      setDetail(full);
    } catch (error) {
      setDetailError(error instanceof Error ? error.message : 'Full signal details are unavailable.');
    }
  };

  const hasActiveFilters = classification !== 'ALL' || status !== 'ALL' || date !== '' || search !== '';

  return (
    <section className="terminal-panel signal-stream-panel" aria-label="Algorithmic Signal Stream">
      {/* Header */}
      <div className="terminal-section-head signal-stream-heading">
        <div>
          <span className="terminal-kicker">PERSISTED DECISIONS · HYPOTHETICAL ONLY</span>
          <h2>Algorithmic Signal Stream</h2>
          <p>Stored records from the analysis API. Select any row to inspect full reasoning, risk setup, and model breakdown.</p>
        </div>
        <div className="stream-count">
          {filtered.length.toLocaleString()}{' '}
          <span>shown of {signals.length.toLocaleString()} loaded</span>
        </div>
      </div>

      {/* Simplified, Clean Toolbar */}
      <div className="stream-toolbar">
        {/* Search Input */}
        <div className="stream-search-wrap">
          <span className="stream-search-icon" aria-hidden="true">⌕</span>
          <input
            className="stream-search-input"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search signals, reasons, IDs..."
            aria-label="Search signals"
          />
          {search && (
            <button
              type="button"
              className="stream-clear-search"
              onClick={() => setSearch('')}
              title="Clear search"
            >
              ×
            </button>
          )}
        </div>

        {/* Compact Dropdown Filters */}
        <div className="stream-filters-group">
          {/* 1. Signal Filter */}
          <label className="stream-filter-control">
            <span className="stream-filter-label">Signal</span>
            <select
              value={classification}
              onChange={(e) => setClassification(e.target.value)}
              aria-label="Filter by signal"
            >
              <option value="ALL">All Signals</option>
              <option value="BUY">BUY</option>
              <option value="SELL">SELL</option>
              <option value="HOLD">HOLD</option>
              <option value="WAIT">WAIT</option>
            </select>
          </label>

          {/* 2. Status Filter */}
          <label className="stream-filter-control">
            <span className="stream-filter-label">Status</span>
            <select
              value={status}
              onChange={(e) => setStatus(e.target.value)}
              aria-label="Filter by status"
            >
              <option value="ALL">All Statuses</option>
              {statuses.map((item) => (
                <option value={item} key={item}>
                  {formatStatus(item)}
                </option>
              ))}
            </select>
          </label>

          {/* 3. ET Date Filter */}
          <label className="stream-filter-control">
            <span className="stream-filter-label">ET Date</span>
            <input
              type="date"
              value={date}
              onChange={(e) => setDate(e.target.value)}
              aria-label="Filter by ET date"
            />
          </label>

          {/* 4. Sort Order */}
          <button
            type="button"
            className="stream-btn-secondary"
            onClick={() => setAscending((old) => !old)}
            title="Toggle sort order"
          >
            {ascending ? 'Oldest first ↑' : 'Newest first ↓'}
          </button>

          {/* 5. Auto-Scroll Toggle */}
          <label className="stream-filter-check">
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={(e) => setAutoScroll(e.target.checked)}
            />
            <span>Auto-scroll</span>
          </label>

          {/* Optional Reset Link */}
          {hasActiveFilters && (
            <button
              type="button"
              className="stream-reset-link"
              onClick={() => {
                setClassification('ALL');
                setStatus('ALL');
                setDate('');
                setSearch('');
              }}
            >
              Reset filters
            </button>
          )}
        </div>
      </div>

      {detailError && <div className="inline-warning stream-warning">{detailError}</div>}

      {/* Main Signal Table (Reduced to 6 visible columns) */}
      <div className="signal-table-scroll" ref={scroller}>
        <table className="stream-table">
          <thead>
            <tr>
              <th className="col-time">Time</th>
              <th className="col-signal">Signal</th>
              <th className="col-ml">ML Direction</th>
              <th className="col-trigger text-right">Trigger Price</th>
              <th className="col-status">Status</th>
              <th className="col-pnl text-right">Hypothetical P&amp;L</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((record) => {
              const expanded = expandedId === record.signal_id;
              return (
                <SignalRow
                  key={record.signal_id}
                  record={record}
                  expanded={expanded}
                  detail={expanded ? detail : null}
                  onToggle={() => void toggleDetails(record)}
                />
              );
            })}
            {!rows.length && (
              <tr>
                <td colSpan={6} className="stream-empty-cell">
                  {signals.length
                    ? 'No persisted signals match the active filters.'
                    : 'No analysis signals have been persisted yet. Run an analysis on the latest candle above.'}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      <div className="stream-pagination">
        <button
          type="button"
          className="terminal-icon-button"
          disabled={page === 0}
          onClick={() => setPage((p) => Math.max(0, p - 1))}
        >
          ← Previous
        </button>
        <span>
          Page {page + 1} of {pages}
        </span>
        <button
          type="button"
          className="terminal-icon-button"
          disabled={page + 1 >= pages}
          onClick={() => setPage((p) => Math.min(pages - 1, p + 1))}
        >
          Next →
        </button>
        <button
          type="button"
          className="terminal-icon-button load-older"
          disabled={!hasMore || loadingMore}
          onClick={() => void loadOlder()}
        >
          {loadingMore ? 'Loading…' : hasMore ? 'Load older records' : 'No older records loaded'}
        </button>
      </div>

      {/* Footnote */}
      <div className="signal-table-footnote">
        Source: persisted SQLite analysis signals · 5-minute completed candles · Price levels and returns are hypothetical. No broker fills or realized P&amp;L are represented.
      </div>
    </section>
  );
}

function SignalRow({
  record,
  expanded,
  detail,
  onToggle,
}: {
  record: SignalRecord;
  expanded: boolean;
  detail: SignalRecord | null;
  onToggle: () => void;
}) {
  const payload = record.payload || {};
  const risk = payload.risk_plan;
  const ml = payload.ml_evidence;
  const normSignal = normalizeSignal(record.classification);
  const isActionable = ['BUY', 'SELL'].includes(normSignal);

  // Trigger price: actual recorded trigger price if valid setup, otherwise reference candle close
  const triggerPrice =
    risk?.trigger_price != null
      ? risk.trigger_price
      : typeof payload.indicators?.close === 'number'
      ? payload.indicators.close
      : null;

  // ML Direction & Raw Probability:
  const mlDir =
    ml?.direction ||
    (ml?.raw_probability_up != null ? (ml.raw_probability_up >= 0.5 ? 'UP' : 'DOWN') : null);
  const rawUp = ml?.raw_probability_up;
  const rawUpPct = rawUp != null ? (rawUp * 100).toFixed(1) : null;

  // Hypothetical P&L:
  const gross = record.gross_return_pct;
  const net = record.net_return_pct;
  const hasOutcome = gross != null || net != null;

  return (
    <>
      <tr
        className={`stream-row ${expanded ? 'is-expanded' : ''} row-${normSignal.toLowerCase()}`}
        onClick={onToggle}
        tabIndex={0}
        role="button"
        aria-expanded={expanded}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            onToggle();
          }
        }}
      >
        {/* 1. TIME */}
        <td className="stream-cell-time">
          <div className="stream-time-wrap">
            <span className="stream-primary-time">{formatTime(record.candle_timestamp, true)}</span>
            <span className="stream-secondary-time">
              Gen: {formatTime(record.generated_at)}
            </span>
          </div>
        </td>

        {/* 2. SIGNAL */}
        <td className="stream-cell-signal">
          <span className={`stream-signal-badge badge-${normSignal.toLowerCase()}`}>
            {normSignal === 'BUY' ? '↑ BUY' : normSignal === 'SELL' ? '↓ SELL' : normSignal}
          </span>
        </td>

        {/* 3. ML DIRECTION */}
        <td className="stream-cell-ml">
          <div className="stream-ml-wrap">
            <span
              className={`stream-ml-dir ${
                mlDir === 'UP' ? 'tone-up' : mlDir === 'DOWN' ? 'tone-down' : 'tone-muted'
              }`}
            >
              {mlDir || '—'}
            </span>
            {rawUpPct != null ? (
              <span className="stream-ml-sub" title="Raw model probability (uncalibrated score)">
                {rawUpPct}% P(UP) · raw
              </span>
            ) : (
              <span className="stream-ml-sub">Unavailable</span>
            )}
          </div>
        </td>

        {/* 4. TRIGGER PRICE */}
        <td className="stream-cell-price text-right">
          <span className="mono-num">
            {triggerPrice != null ? price(triggerPrice) : '—'}
          </span>
        </td>

        {/* 5. STATUS */}
        <td className="stream-cell-status">
          <span className={`stream-status-pill status-${record.status.toLowerCase()}`}>
            {formatStatus(record.status)}
          </span>
        </td>

        {/* 6. HYPOTHETICAL P&L */}
        <td className="stream-cell-pnl text-right">
          {hasOutcome ? (
            <div className="stream-pnl-wrap">
              <span
                className={`stream-pnl-val mono-num ${
                  Number(net) > 0 ? 'tone-up' : Number(net) < 0 ? 'tone-down' : ''
                }`}
              >
                {percent(net)}
              </span>
              {gross != null && gross !== net && (
                <span className="stream-pnl-gross mono-num">
                  Gross {percent(gross)}
                </span>
              )}
            </div>
          ) : isActionable ? (
            <span className="stream-pnl-pending">Pending</span>
          ) : (
            <span className="stream-pnl-none">—</span>
          )}
        </td>
      </tr>

      {/* Expandable Detail View */}
      {expanded && (
        <ExpandableDetailRow
          record={record}
          detail={detail}
          onClose={onToggle}
        />
      )}
    </>
  );
}

export function ExpandableDetailRow({
  record,
  detail,
  onClose,
}: {
  record: SignalRecord;
  detail: SignalRecord | null;
  onClose: () => void;
}) {
  const payload = record.payload || {};
  const risk = payload.risk_plan;
  const ml = payload.ml_evidence;
  const score = record.final_score ?? payload.final_decision_score;
  const techScore = payload.technical_score;

  const rawUp = ml?.raw_probability_up;
  const rawDown = rawUp != null ? Math.max(0, Math.min(1, 1.0 - rawUp)) : null;

  const bias = record.directional_bias || payload.directional_bias || 'NEUTRAL';
  const biasTone = bias.toUpperCase().includes('BULL')
    ? 'tone-up'
    : bias.toUpperCase().includes('BEAR')
    ? 'tone-down'
    : '';

  // Component contributions:
  const contributions = useMemo(() => {
    const raw = Object.entries(payload.component_contributions || {});
    return raw
      .filter((entry): entry is [string, number] => typeof entry[1] === 'number' && Number.isFinite(entry[1]))
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  }, [payload.component_contributions]);

  const scale = useMemo(() => {
    if (!contributions.length) return 1;
    return Math.max(0.0001, ...contributions.map(([, val]) => Math.abs(val)));
  }, [contributions]);

  return (
    <tr className="stream-detail-row">
      <td colSpan={6}>
        <div className="stream-detail-panel">
          {/* Header Bar */}
          <div className="stream-detail-header">
            <div className="stream-detail-header-info">
              <span className="stream-detail-id">
                Signal ID: <code>{record.signal_id}</code>
              </span>
              <span className="stream-detail-chip">
                {payload.analysis_version || record.analysis_version || 'v0.2.0'}
              </span>
              <span className="stream-detail-chip">
                Candle: {formatTime(record.candle_timestamp, true)}
              </span>
              <span className="stream-detail-chip">
                Generated: {formatTime(record.generated_at || payload.generated_at, true)}
              </span>
            </div>
            <button
              type="button"
              className="stream-detail-close-btn"
              onClick={(e) => {
                e.stopPropagation();
                onClose();
              }}
              title="Close details"
            >
              Close ✕
            </button>
          </div>

          {/* 2-Column Responsive Grid */}
          <div className="stream-detail-body">
            {/* LEFT COLUMN: Metrics & Risk Management */}
            <div className="stream-detail-col">
              {/* Box 1: Decision & ML Model Metrics */}
              <div className="stream-detail-card">
                <h4 className="stream-card-heading">Decision &amp; Model Metrics</h4>
                <div className="stream-metrics-grid">
                  <div className="stream-metric-cell">
                    <span className="stream-metric-lbl">Market Bias</span>
                    <b className={`stream-metric-val ${biasTone}`}>{bias}</b>
                  </div>
                  <div className="stream-metric-cell">
                    <span className="stream-metric-lbl">Decision Engine Score</span>
                    <b className="stream-metric-val mono-num">
                      {score != null ? score.toFixed(3) : '—'}
                    </b>
                  </div>
                  <div className="stream-metric-cell">
                    <span className="stream-metric-lbl">Technical Score</span>
                    <b className="stream-metric-val mono-num">
                      {techScore != null ? techScore.toFixed(3) : '—'}
                    </b>
                  </div>
                  <div className="stream-metric-cell">
                    <span className="stream-metric-lbl">Market Regime</span>
                    <b className="stream-metric-val">
                      {formatStatus(payload.market_regime?.classification || '')}
                    </b>
                  </div>
                </div>

                {/* ML Direction & Raw Probabilities */}
                <div className="stream-ml-section">
                  <div className="stream-ml-row">
                    <span className="stream-ml-row-title">ML Directional Model (10m)</span>
                    <span
                      className={`stream-ml-pill ${
                        ml?.direction === 'UP' ? 'tone-up' : ml?.direction === 'DOWN' ? 'tone-down' : ''
                      }`}
                    >
                      {ml?.direction || 'Unavailable'}
                    </span>
                  </div>
                  {rawUp != null ? (
                    <div className="stream-ml-scores">
                      <span>
                        Raw P(UP): <b className="mono-num">{(rawUp * 100).toFixed(1)}%</b>
                      </span>
                      <span className="stream-sep">·</span>
                      <span>
                        Raw P(DOWN):{' '}
                        <b className="mono-num">
                          {rawDown != null ? (rawDown * 100).toFixed(1) : '—'}%
                        </b>
                      </span>
                    </div>
                  ) : null}
                  <div className="stream-detail-disclaimer">
                    Raw model scores are uncalibrated (directional evidence only, not confidence or execution probabilities).
                  </div>
                </div>
              </div>

              {/* Box 2: Risk Management / Trade Setup */}
              <div className="stream-detail-card">
                <div className="stream-card-head-row">
                  <h4 className="stream-card-heading">Trade Setup &amp; Risk Levels</h4>
                  <span
                    className={`stream-status-pill ${
                      risk?.valid ? 'status-active' : 'status-not_actionable'
                    }`}
                  >
                    {risk?.valid ? 'Valid Plan' : 'Not Actionable'}
                  </span>
                </div>

                {risk?.valid ? (
                  <div className="stream-risk-grid">
                    <div className="stream-risk-cell">
                      <span className="stream-risk-lbl">Trigger Price</span>
                      <b className="stream-risk-val mono-num">{price(risk.trigger_price)}</b>
                    </div>
                    <div className="stream-risk-cell">
                      <span className="stream-risk-lbl">Entry Zone</span>
                      <b className="stream-risk-val mono-num">
                        {risk.entry_zone
                          ? `${price(risk.entry_zone.lower)} – ${price(risk.entry_zone.upper)}`
                          : '—'}
                      </b>
                    </div>
                    <div className="stream-risk-cell">
                      <span className="stream-risk-lbl">Stop Loss</span>
                      <b className="stream-risk-val mono-num tone-down">{price(risk.stop_loss)}</b>
                    </div>
                    <div className="stream-risk-cell">
                      <span className="stream-risk-lbl">Target 1</span>
                      <b className="stream-risk-val mono-num tone-up">{price(risk.target_1)}</b>
                    </div>
                    <div className="stream-risk-cell">
                      <span className="stream-risk-lbl">Target 2</span>
                      <b className="stream-risk-val mono-num tone-up">{price(risk.target_2)}</b>
                    </div>
                    <div className="stream-risk-cell">
                      <span className="stream-risk-lbl">Reward / Risk</span>
                      <b className="stream-risk-val mono-num">
                        {risk.risk_reward_ratio != null
                          ? `${risk.risk_reward_ratio.toFixed(2)} : 1`
                          : '—'}
                      </b>
                    </div>
                  </div>
                ) : (
                  <div className="stream-no-risk-box">
                    <p className="stream-no-risk-msg">
                      {risk?.reason || 'No actionable trade setup approved at this candle.'}
                    </p>
                    {payload.setup_invalidation && (
                      <span className="stream-invalidation-msg">
                        Invalidation: {formatTraderEvidence(payload.setup_invalidation)}
                      </span>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* RIGHT COLUMN: Decision Engine Contributions & Evidence */}
            <div className="stream-detail-col">
              {/* Box 3: Decision Engine Contributions */}
              <div className="stream-detail-card">
                <div className="stream-card-head-row">
                  <h4 className="stream-card-heading">Decision Engine Contributions</h4>
                  <span className="stream-muted-hint">Not ML feature importance</span>
                </div>

                {contributions.length ? (
                  <div className="stream-contrib-table">
                    {contributions.map(([name, val]) => {
                      const isPos = val >= 0;
                      const pct = scale > 0 ? (Math.abs(val) / scale) * 100 : 0;
                      return (
                        <div className="stream-contrib-item" key={name}>
                          <span className="stream-contrib-label">
                            {componentLabels[name] || name.replaceAll('_', ' ')}
                          </span>
                          <div className="stream-contrib-bar-wrap">
                            <div
                              className={`stream-contrib-bar ${isPos ? 'pos' : 'neg'}`}
                              style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
                            />
                          </div>
                          <span className={`stream-contrib-num mono-num ${isPos ? 'pos' : 'neg'}`}>
                            {isPos ? '+' : ''}{val.toFixed(3)}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <span className="stream-muted-hint">Component contributions unavailable.</span>
                )}
              </div>

              {/* Box 4: Evidence & Backend Reasoning */}
              <div className="stream-detail-card">
                <h4 className="stream-card-heading">Evidence &amp; Decision Reasoning</h4>
                <div className="stream-evidence-list">
                  {payload.bullish_evidence?.length ? (
                    <div className="stream-evidence-block group-bullish">
                      <b className="evidence-title">Bullish Evidence</b>
                      <ul>
                        {payload.bullish_evidence.map((item, idx) => (
                          <li key={idx}>{formatTraderEvidence(item)}</li>
                        ))}
                      </ul>
                    </div>
                  ) : null}

                  {payload.bearish_evidence?.length ? (
                    <div className="stream-evidence-block group-bearish">
                      <b className="evidence-title">Bearish Evidence</b>
                      <ul>
                        {payload.bearish_evidence.map((item, idx) => (
                          <li key={idx}>{formatTraderEvidence(item)}</li>
                        ))}
                      </ul>
                    </div>
                  ) : null}

                  {payload.neutral_observations?.length ? (
                    <div className="stream-evidence-block group-neutral">
                      <b className="evidence-title">Neutral Observations</b>
                      <ul>
                        {payload.neutral_observations.map((item, idx) => (
                          <li key={idx}>{formatTraderEvidence(item)}</li>
                        ))}
                      </ul>
                    </div>
                  ) : null}

                  {payload.reasons?.length ? (
                    <div className="stream-evidence-block group-reasons">
                      <b className="evidence-title">Decision Engine Evaluation Rules</b>
                      <ul>
                        {payload.reasons.map((item, idx) => (
                          <li key={idx}>{formatTraderEvidence(item)}</li>
                        ))}
                      </ul>
                      {payload.required_confirmation && (
                        <div className="stream-confirm-note">
                          <b>Confirmation Required:</b> {formatTraderEvidence(payload.required_confirmation)}
                        </div>
                      )}
                    </div>
                  ) : null}
                </div>
              </div>
            </div>
          </div>

          {/* Lifecycle events if present */}
          {detail?.events?.length ? (
            <div className="stream-events-bar">
              <span className="stream-events-heading">Signal Lifecycle Events:</span>
              <div className="stream-events-chips">
                {detail.events.map((ev, idx) => (
                  <span className="stream-event-chip" key={idx}>
                    <b>{formatStatus(ev.event_type)}</b> at {formatTime(ev.timestamp, true)}
                  </span>
                ))}
              </div>
            </div>
          ) : null}

          {/* Disclaimer footer */}
          <div className="stream-detail-foot">
            <span>
              {payload.hypothetical_only
                ? 'Hypothetical analysis only · No live broker execution occurred.'
                : 'Persisted analysis record from SQLite database.'}
            </span>
            {detail && detail.signal_id !== record.signal_id && (
              <span className="stream-loading-msg">Loading full signal record...</span>
            )}
          </div>
        </div>
      </td>
    </tr>
  );
}
