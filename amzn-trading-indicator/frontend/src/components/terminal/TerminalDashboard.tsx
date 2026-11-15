import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { analysisApi } from '../../services/analysisApi';
import { API_BASE } from '../../services/api';
import type { AnalysisLatestResponse, DataSnapshotStatus, MarketSnapshot, SignalRecord } from '../../types/analysis';
import type { MarketIntelligence } from '../../services/analysisApi';
import { formatDate, formatTime, price } from '../../utils/format';
import { AnalysisPanel } from './AnalysisPanel';
import { SignalStream } from './SignalStream';
import { TradingChart } from './TradingChart';
import { MarketIntelligencePanel } from './MarketIntelligencePanel';

type Activity = 'loading' | 'refreshing' | 'analyzing' | 'syncing' | '';
type SyncOperation = { operation_id: string; kind: string; status: string; stage: string; progress: number; updated_at?: string; error?: string | null };

function normalizeLatest(data: AnalysisLatestResponse): SignalRecord | null {
  if (!data.signal_id || !data.payload) return null;
  return {
    signal_id: data.signal_id, candle_timestamp: data.candle_timestamp || data.payload.candle_timestamp,
    generated_at: data.generated_at || data.payload.generated_at, analysis_version: data.payload.analysis_version,
    classification: data.classification || data.payload.classification, directional_bias: data.directional_bias || data.payload.directional_bias || 'NEUTRAL',
    final_score: data.final_score ?? data.payload.final_decision_score, payload: data.payload, status: data.status || 'UNKNOWN',
    outcome: data.outcome, gross_return_pct: data.gross_return_pct, net_return_pct: data.net_return_pct,
  };
}

function getMarketState(timestamp?: string | null, stream?: MarketSnapshot['stream_status']) {
  if (stream?.market_state === 'CLOSED') return { stale: false, state: 'CLOSED', marketClockOpen: false, detail: 'Market calendar reports the regular session closed; latest stored candles are retained and are not presented as live.' };
  if (stream?.market_state === 'OPEN') {
    if (stream.connection_state !== 'CONNECTED') return { stale: true, state: 'STALE', marketClockOpen: true, detail: `The market is open, but the Alpaca stream is ${stream.connection_state || 'not confirmed connected'}; analysis is disabled until the feed recovers.` };
    if (stream.freshness === 'STALE') return { stale: true, state: 'STALE', marketClockOpen: true, detail: 'The Alpaca stream is connected, but its latest completed candle is stale.' };
    return { stale: false, state: 'FRESH', marketClockOpen: true, detail: 'Market session and data freshness are reported by the backend.' };
  }
  if (!timestamp) return { stale: true, state: 'UNAVAILABLE', marketClockOpen: false, detail: 'No completed market candle is available.' };
  const now = new Date();
  const parts = new Intl.DateTimeFormat('en-US', { timeZone: 'America/New_York', weekday: 'short', hour: '2-digit', minute: '2-digit', hourCycle: 'h23' }).formatToParts(now);
  const day = parts.find((p) => p.type === 'weekday')?.value || '';
  const minute = Number(parts.find((p) => p.type === 'hour')?.value || 0) * 60 + Number(parts.find((p) => p.type === 'minute')?.value || 0);
  const weekday = !['Sat', 'Sun'].includes(day);
  const rthClock = weekday && minute >= 570 && minute < 960;
  const age = now.getTime() - new Date(timestamp).getTime();
  const fresh = age >= -60_000 && age <= 15 * 60_000;
  if (rthClock && fresh) return { stale: false, state: 'FRESH', marketClockOpen: true, detail: 'Latest completed candle is within the existing 15-minute data freshness window.' };
  if (rthClock) return { stale: true, state: 'STALE', marketClockOpen: true, detail: 'No completed candle has arrived within the existing 15-minute freshness window.' };
  return { stale: age > 24 * 60 * 60_000, state: 'CLOSED', marketClockOpen: false, detail: 'Regular-hours clock is closed; showing the latest stored completed candle. Holiday and early-close state is not provided by this API.' };
}

export function TerminalDashboard() {
  const [market, setMarket] = useState<MarketSnapshot | null>(null);
  const [dataStatus, setDataStatus] = useState<DataSnapshotStatus | null>(null);
  const [analysis, setAnalysis] = useState<SignalRecord | null>(null);
  const [signals, setSignals] = useState<SignalRecord[]>([]);
  const [intelligence, setIntelligence] = useState<MarketIntelligence | null>(null);
  const [intelligenceError, setIntelligenceError] = useState('');
  const [activity, setActivity] = useState<Activity>('loading');
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [selectedTimestamp, setSelectedTimestamp] = useState<string | null>(null);
  const [visibleChartCandle, setVisibleChartCandle] = useState<MarketSnapshot['candles'][number] | null>(null);
  const [syncOperation, setSyncOperation] = useState<SyncOperation | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const eventCursorRef = useRef(0);
  const lastCandleTimestampRef = useRef<string | null>(null);

  const reload = useCallback(async (quiet = false) => {
    if (!quiet) { setActivity((old) => old ? old : 'refreshing'); setError(''); }
    const [marketResult, statusResult, latestResult, signalsResult, intelligenceResult] = await Promise.allSettled([
      analysisApi.market(1000), analysisApi.status(), analysisApi.latest(), analysisApi.signals(0, 100), analysisApi.marketIntelligence(),
    ]);
    if (marketResult.status === 'fulfilled') {
      setMarket(marketResult.value);
      eventCursorRef.current = marketResult.value.event_cursor || eventCursorRef.current;
      lastCandleTimestampRef.current = marketResult.value.candles.at(-1)?.timestamp || lastCandleTimestampRef.current;
    }
    if (statusResult.status === 'fulfilled') setDataStatus(statusResult.value);
    if (latestResult.status === 'fulfilled') setAnalysis(normalizeLatest(latestResult.value));
    if (signalsResult.status === 'fulfilled') setSignals(signalsResult.value.signals);
    if (intelligenceResult.status === 'fulfilled') { setIntelligence(intelligenceResult.value); setIntelligenceError(''); }
    else setIntelligenceError(intelligenceResult.reason instanceof Error ? intelligenceResult.reason.message : 'Market context unavailable.');
    const failures = [marketResult, statusResult, latestResult, signalsResult].filter((result) => result.status === 'rejected');
    if (failures.length === 4) setError(failures[0].reason instanceof Error ? failures[0].reason.message : 'Cannot reach the analysis API.');
    else if (failures.length && !quiet) setNotice(`${failures.length} data request${failures.length === 1 ? '' : 's'} failed; available panels remain visible.`);
    else if (!failures.length) setNotice('');
    setActivity('');
  }, []);

  const refreshAnalysisPanels = useCallback(async () => {
    const [statusResult, latestResult, signalsResult, streamResult, intelligenceResult] = await Promise.allSettled([
      analysisApi.status(), analysisApi.latest(), analysisApi.signals(0, 100), analysisApi.streamStatus(), analysisApi.marketIntelligence(),
    ]);
    if (statusResult.status === 'fulfilled') setDataStatus(statusResult.value);
    if (latestResult.status === 'fulfilled') setAnalysis(normalizeLatest(latestResult.value));
    if (signalsResult.status === 'fulfilled') setSignals(signalsResult.value.signals);
    if (streamResult.status === 'fulfilled') setMarket((current) => current ? { ...current, stream_status: streamResult.value } : current);
    if (intelligenceResult.status === 'fulfilled') { setIntelligence(intelligenceResult.value); setIntelligenceError(''); }
    else setIntelligenceError(intelligenceResult.reason instanceof Error ? intelligenceResult.reason.message : 'Market context unavailable.');
  }, []);

  useEffect(() => {
    void reload();
    const timer = window.setInterval(() => void refreshAnalysisPanels(), 60_000);
    return () => window.clearInterval(timer);
  }, [reload, refreshAnalysisPanels]);

  useEffect(() => {
    if (market?.event_cursor === undefined) return;
    let disposed = false;
    const connect = () => {
      if (disposed || document.visibilityState === 'hidden') return;
      eventSourceRef.current?.close();
      const source = new EventSource(`${API_BASE}/market/events?last_event_id=${eventCursorRef.current}`);
      eventSourceRef.current = source;
      const parse = (event: MessageEvent) => {
        try {
          const value = JSON.parse(event.data);
          eventCursorRef.current = Math.max(eventCursorRef.current, Number(value.event_id || event.lastEventId || 0));
          if (value.event_type === 'market_status') {
            setMarket((current) => current ? { ...current, stream_status: { ...current.stream_status, ...value } } : current);
          } else if (value.event_type === 'candle_update') {
            setMarket((current) => current ? { ...current, forming_candle: { ...value.candle, observed_minute_count: value.observed_minute_count } } : current);
          } else if (['candle_completed', 'candle_corrected'].includes(value.event_type)) {
            const candle = value.candle;
            if (candle?.timestamp) {
              setMarket((current) => {
                if (!current) return current;
                const map = new Map(current.candles.map((item) => [item.timestamp, item]));
                map.set(candle.timestamp, candle);
                const candles = Array.from(map.values()).sort((a, b) => a.timestamp.localeCompare(b.timestamp)).slice(-1000);
                lastCandleTimestampRef.current = candles.at(-1)?.timestamp || null;
                return { ...current, candles, latest_timestamp: candles.at(-1)?.timestamp || current.latest_timestamp,
                  chart_rows: candles.length, forming_candle: current.forming_candle?.timestamp === candle.timestamp ? null : current.forming_candle,
                  stream_status: { ...current.stream_status, last_completed_candle: candle.timestamp, last_persisted_candle: candle.timestamp, freshness: 'FRESH' } };
              });
            }
            if (value.event_type === 'candle_completed') void refreshAnalysisPanels();
          } else if (value.event_type === 'analysis_updated' || value.event_type === 'signal_created') {
            void refreshAnalysisPanels();
          } else if (value.event_type === 'synchronization_completed') {
            setMarket((current) => current ? { ...current, stream_status: { ...current.stream_status, last_reconciliation_at: value.event_timestamp, reconciliation_error: null } } : current);
          } else if (value.event_type === 'data_gap_detected') {
            setNotice(`Market data gap detected: ${value.missing_interval_count} interval(s) between ${formatTime(value.from, true)} and ${formatTime(value.to, true)}. No candles were fabricated.`);
          } else if (value.event_type === 'analysis_failed') {
            setError(`Analysis failed for ${formatTime(value.candle_timestamp, true)}: ${value.error || 'Unknown analysis error'}`);
          } else if (value.event_type === 'reconciliation_failed') {
            setError(`Market-data reconciliation failed: ${value.error || 'Unknown provider error'}`);
          }
        } catch { /* Ignore malformed events; EventSource continues and server replay remains available. */ }
      };
      for (const name of ['market_status', 'candle_update', 'candle_completed', 'candle_corrected', 'analysis_updated', 'analysis_failed', 'signal_created', 'data_gap_detected', 'reconciliation_failed', 'synchronization_completed']) {
        source.addEventListener(name, parse as EventListener);
      }
      source.onerror = () => setMarket((current) => current ? { ...current, stream_status: { ...current.stream_status, connection_state: 'RECONNECTING' } } : current);
    };
    const visibility = () => {
      if (document.visibilityState === 'hidden') { eventSourceRef.current?.close(); eventSourceRef.current = null; }
      else {
        const last = lastCandleTimestampRef.current;
        if (last) void analysisApi.marketAfter(last).then((delta) => {
          if (disposed) return;
          setMarket((current) => {
            if (!current) return current;
            const map = new Map(current.candles.map((item) => [item.timestamp, item]));
            delta.candles.forEach((item) => map.set(item.timestamp, item));
            const candles = Array.from(map.values()).sort((a, b) => a.timestamp.localeCompare(b.timestamp)).slice(-1000);
            lastCandleTimestampRef.current = candles.at(-1)?.timestamp || last;
            return { ...current, candles, latest_timestamp: candles.at(-1)?.timestamp || current.latest_timestamp,
              forming_candle: delta.forming_candle ?? current.forming_candle, stream_status: delta.stream_status || current.stream_status };
          });
        }).catch(() => undefined);
        connect();
      }
    };
    connect();
    document.addEventListener('visibilitychange', visibility);
    return () => { disposed = true; document.removeEventListener('visibilitychange', visibility); eventSourceRef.current?.close(); eventSourceRef.current = null; };
  }, [market?.event_cursor, refreshAnalysisPanels]);

  useEffect(() => {
    if (!syncOperation || syncOperation.status !== 'RUNNING') return;
    let active = true;
    const timer = window.setInterval(async () => {
      try {
        const response = await analysisApi.operations();
        const current = response.operations.find((operation) => operation.operation_id === syncOperation.operation_id);
        if (!active || !current) return;
        setSyncOperation(current);
        if (current.status === 'COMPLETED') {
          setNotice('Market-data sync completed. Reloading the local snapshot.');
          void reload(true);
        } else if (current.status === 'FAILED') {
          setError(current.error || 'The configured market-data sync failed.');
          setNotice('Market-data refresh failed; the existing local snapshot remains displayed.');
        }
      } catch { /* keep the last known state; manual reload remains available */ }
    }, 2500);
    return () => { active = false; window.clearInterval(timer); };
  }, [syncOperation?.operation_id, syncOperation?.status, reload]);

  const addSignals = useCallback((incoming: SignalRecord[], append: boolean) => {
    setSignals((current) => {
      if (!append) return incoming;
      const map = new Map(current.map((row) => [row.signal_id, row]));
      for (const row of incoming) if (!map.has(row.signal_id)) map.set(row.signal_id, row);
      return Array.from(map.values()).sort((a, b) => b.candle_timestamp.localeCompare(a.candle_timestamp));
    });
  }, []);

  const runAnalysis = async () => {
    setActivity('analyzing'); setError(''); setNotice('');
    try {
      const result = await analysisApi.analyze();
      setAnalysis(result.signal_record);
      setNotice(`Stored analysis for the completed ${formatTime(result.analysis?.candle_timestamp, true)} candle.`);
      await reload(true);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Analysis request failed.');
    } finally { setActivity(''); }
  };

  const syncSnapshot = async () => {
    setActivity('syncing'); setError(''); setNotice('');
    try {
      const op = await analysisApi.refreshSnapshot();
      setSyncOperation(op);
      setNotice(`Market-data refresh queued (${op.operation_id}). The backend uses its configured IEX credentials; no provider credentials are exposed in this UI.`);
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Market-data refresh could not be started.'); }
    finally { setActivity(''); }
  };

  const candleTime = market?.latest_timestamp || dataStatus?.latest_completed_rth_candle;
  const marketState = getMarketState(candleTime, market?.stream_status);
  const latestMarketCandle = market?.candles.at(-1);
  const latestAligned = Boolean(analysis && latestMarketCandle && new Date(analysis.candle_timestamp).getTime() === new Date(latestMarketCandle.timestamp).getTime());
  const currentPrice = latestMarketCandle?.close;
  const analysisPayload = analysis?.payload;
  const signalCount = useMemo(() => signals.filter((s) => ['BUY', 'SELL'].includes(s.classification)).length, [signals]);

  return <main className="terminal-page">
    {marketState.state === 'CLOSED' && <div className="market-closed-banner"><b>MARKET CLOSED · REGULAR-HOURS CLOCK</b><span>{marketState.detail} Last completed candle: {formatTime(candleTime, true)}. {analysis ? `Analysis generated: ${formatTime(analysis.generated_at, true)}.` : 'No stored analysis.'}</span></div>}
    {marketState.stale && <div className="market-alert"><b>MARKET DATA STALE — DO NOT TREAT AS LIVE</b><span>{marketState.detail} Last completed candle: {formatTime(candleTime, true)}. {analysis ? `Analysis generated: ${formatTime(analysis.generated_at, true)}.` : 'No stored analysis.'}</span></div>}
    {error && <div className="terminal-alert" role="alert"><span>{error}</span><button onClick={() => void reload()}>Retry</button></div>}
    {notice && <div className="terminal-notice" role="status"><span>{notice}</span><button aria-label="Dismiss notice" onClick={() => setNotice('')}>×</button></div>}
    <div className="terminal-topline">
      <div className="terminal-instrument"><span className="terminal-symbol">AMZN</span><span className="exchange-tag">NASDAQ</span><span className="instrument-chevron">⌄</span><i />
        <span className="price-source-label">LAST COMPLETED 5M CLOSE</span><b className="mono-price">{price(currentPrice)}</b><span className="feed-tag">{market?.feed || dataStatus?.feed || 'IEX · 5-minute candles'}</span>
      </div>
      <div className="timeframe-strip"><button className="selected" aria-current="true">5m</button><span>Only interval exposed by the market API</span></div>
      <div className="terminal-actions"><span className={`feed-state ${marketState.state.toLowerCase()}`}><i />{marketState.state === 'FRESH' ? 'DATA FRESH' : marketState.state === 'CLOSED' ? 'MARKET CLOSED' : marketState.state}</span>
        {syncOperation?.status === 'RUNNING' && <span className="operation-inline">SYNC {syncOperation.stage.replaceAll('_', ' ')} · {Math.round(syncOperation.progress * 100)}%</span>}
        <button className="terminal-icon-button" disabled={activity !== ''} onClick={() => void reload()}>{activity === 'refreshing' ? 'Loading…' : '↻ Reload data'}</button>
        <button className="terminal-icon-button" disabled={activity !== '' || syncOperation?.status === 'RUNNING'} onClick={() => void syncSnapshot()}>{activity === 'syncing' ? 'Queueing…' : syncOperation?.status === 'RUNNING' ? 'Sync running…' : 'Sync IEX'}</button>
        <button className="terminal-primary-button" disabled={activity !== '' || marketState.stale || !latestMarketCandle} title={marketState.stale ? 'Analysis is disabled because the latest available candle is stale.' : 'Analyze only the latest completed candle'} onClick={() => void runAnalysis()}>{activity === 'analyzing' ? 'Analyzing…' : marketState.stale ? 'Analysis disabled · stale data' : 'Analyze latest candle'}</button>
      </div>
    </div>

    <div className="terminal-grid">
      <section className="chart-region">
        <div className="terminal-panel chart-panel">
          <div className="terminal-chart-heading"><div><span className="terminal-kicker">NASDAQ · AMZN · REGULAR SESSION</span><h1>Price action</h1><p>Actual completed 5-minute OHLCV candles · Eastern Time</p><p className="market-stream-detail">Alpaca IEX · Stream {market?.stream_status?.connection_state || 'STATUS UNKNOWN'} · Last event {formatTime(market?.stream_status?.last_received_event_at, true)} · Persisted {formatTime(market?.stream_status?.last_persisted_candle, true)}{market?.forming_candle ? ` · Forming candle · ${market.forming_candle.observed_minute_count ?? '—'}/5 observed minutes` : ' · No current forming candle'}{market?.stream_status?.reconciliation_error ? ` · Reconciliation error: ${market.stream_status.reconciliation_error}` : ''}</p></div>
            <div className="chart-price-summary"><small>LAST COMPLETED CANDLE CLOSE</small><b>{price(currentPrice)}</b><span>{formatTime(candleTime, true)}</span><small>{market?.source || 'Local processed snapshot'}</small></div>
          </div>
          <TradingChart candles={market?.candles || []} formingCandle={market?.forming_candle} signals={signals} payload={analysisPayload} intelligence={intelligence} selectedTimestamp={selectedTimestamp} onVisibleRightCandle={setVisibleChartCandle} />
          <div className="chart-data-note"><span>{market?.chart_rows?.toLocaleString() || '0'} actual candles plotted</span><span>Chart right edge: {price(visibleChartCandle?.close)} · {formatTime(visibleChartCandle?.timestamp, true)}</span><span>Latest data close: {price(currentPrice)} · {latestAligned ? 'analysis uses this candle' : 'analysis uses another candle'}</span><span>Signals: {signalCount} BUY/SELL markers loaded</span></div>
        </div>
      </section>
      <aside className="terminal-aside">
        <MarketIntelligencePanel data={intelligence} error={intelligenceError} analysisTimestamp={analysis?.candle_timestamp} />
        <AnalysisPanel record={analysis} market={market} dataStatus={dataStatus} fresh={marketState.state === 'FRESH'} marketClosed={marketState.state === 'CLOSED'} latestAligned={latestAligned} />
      </aside>
    </div>
    <SignalStream signals={signals} onSignals={addSignals} onSelect={(record) => setSelectedTimestamp(record.candle_timestamp)} />
    <footer className="terminal-footer"><span>ACADEMIC DECISION SUPPORT · NO ORDER EXECUTION</span><span>Local snapshot · {market?.dataset_rows?.toLocaleString() || '—'} processed rows · Updated {formatTime(dataStatus?.last_sync_at, true)}</span><span>{analysis?.payload.market_timezone || 'America/New_York'}</span></footer>
  </main>;
}
