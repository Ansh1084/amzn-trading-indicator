import { useCallback, useEffect, useMemo, useState } from 'react';
import { BrandHeader } from './components/BrandHeader';
import { ComparisonView } from './components/ComparisonView';
import { ActualChart } from './components/ActualChart';
import { PredictionHistory } from './components/PredictionHistory';
import { PredictionPanel } from './components/PredictionPanel';
import { SessionControls } from './components/SessionControls';
import { SessionDecision } from './components/SessionDecision';
import { ResearchPerformance } from './components/terminal/ResearchPerformance';
import { TerminalDashboard } from './components/terminal/TerminalDashboard';
import { api } from './services/api';
import type { ActualCandle, DataStatus, ForecastVersion, MarketSnapshot, Operation, SessionId, SessionPrediction, SessionResult } from './types';
import { etDate, formatDate, formatTime } from './utils/format';

type Page = 'Dashboard' | 'Actual vs Predicted' | 'Data & Model';

function SessionForecasts() {
  const [page, setPage] = useState<Page>('Dashboard');
  const [date, setDate] = useState(etDate());
  const [session, setSession] = useState<SessionId>('morning');
  const [version, setVersion] = useState<ForecastVersion>('ADVANCE');
  const [updatedAvailable, setUpdatedAvailable] = useState(false);
  const [result, setResult] = useState<SessionResult | null>(null);
  const [candles, setCandles] = useState<ActualCandle[]>([]);
  const [snapshot, setSnapshot] = useState<MarketSnapshot | null>(null);
  const [data, setData] = useState<DataStatus | null>(null);
  const [operations, setOperations] = useState<Operation[]>([]);
  const [status, setStatus] = useState('Connecting to local API…');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [sessionLoading, setSessionLoading] = useState(true);

  const predictions = result?.predictions || [];
  const ordered = useMemo(() => [...predictions].sort((a, b) => a.interval_start.localeCompare(b.interval_start)), [predictions]);
  const nextPrediction = useMemo(() => {
    const now = Date.now();
    return ordered.find((p) => new Date(p.interval_end).getTime() > now) || ordered.at(-1);
  }, [ordered]);

  const loadSession = useCallback(async (silent = false) => {
    if (!date) {
      setResult(null); setCandles([]); setUpdatedAvailable(false);
      setError('Choose a trading date to load its prediction schedule.');
      if (!silent) setSessionLoading(false);
      return;
    }
    if (!silent) setSessionLoading(true);
    setError('');
    const [sessionResponse, candlesResponse, updatedResponse] = await Promise.allSettled([
      api.session(date, session, version),
      api.actualCandles(date, session),
      session === 'afternoon' ? api.session(date, session, 'UPDATED') : Promise.resolve(null),
    ]);
    if (sessionResponse.status === 'fulfilled') setResult(sessionResponse.value);
    else { setResult(null); setError(sessionResponse.reason instanceof Error ? sessionResponse.reason.message : 'Unable to load predictions.'); }
    if (candlesResponse.status === 'fulfilled') setCandles(candlesResponse.value.candles || []);
    else setCandles([]);
    const hasUpdated = updatedResponse.status === 'fulfilled' && Boolean(updatedResponse.value?.predictions?.length);
    setUpdatedAvailable(hasUpdated);
    if (version === 'UPDATED' && !hasUpdated) setVersion('ADVANCE');
    if (sessionResponse.status === 'fulfilled' && sessionResponse.value.predictions.length) {
      setStatus(`${sessionResponse.value.predictions.length} stored intervals · ${version} · ${formatDate(date)}`);
    } else if (sessionResponse.status === 'fulfilled') {
      setStatus(`No ${version} schedule stored for ${formatDate(date)}.`);
    }
    if (!silent) setSessionLoading(false);
  }, [date, session, version]);

  const refreshStatus = useCallback(async (showSpinner = true) => {
    if (showSpinner) setRefreshing(true);
    const [marketResponse, dataResponse, operationsResponse] = await Promise.allSettled([api.market(), api.dataStatus(), api.operations()]);
    if (marketResponse.status === 'fulfilled') setSnapshot(marketResponse.value);
    if (dataResponse.status === 'fulfilled') setData(dataResponse.value);
    if (operationsResponse.status === 'fulfilled') setOperations(operationsResponse.value.operations || []);
    if (marketResponse.status === 'rejected' && dataResponse.status === 'rejected') {
      setError(marketResponse.reason instanceof Error ? marketResponse.reason.message : 'Backend unavailable.');
      setStatus('Backend unavailable');
    } else if (marketResponse.status === 'fulfilled') {
      setStatus(`${marketResponse.value.source} · ${marketResponse.value.dataset_rows.toLocaleString()} processed candles · latest ${formatTime(marketResponse.value.latest_timestamp, true)}`);
    }
    if (showSpinner) setRefreshing(false);
  }, []);

  useEffect(() => { void refreshStatus(false); }, [refreshStatus]);
  useEffect(() => { void loadSession(); }, [loadSession]);
  useEffect(() => {
    const interval = window.setInterval(() => {
      void refreshStatus(false);
      void loadSession(true);
    }, 30_000);
    return () => window.clearInterval(interval);
  }, [loadSession, refreshStatus]);

  const handleGenerate = async () => {
    setBusy(true); setError('');
    try {
      const generated = await api.generate(date, session, version);
      setResult(generated);
      setStatus(`Stored ${generated.interval_count || generated.predictions.length} ${version} predictions for ${formatDate(date)} · ${session}.`);
      await loadSession(true);
    } catch (e) { setError(e instanceof Error ? e.message : 'Prediction request failed.'); }
    finally { setBusy(false); }
  };

  const handleEvaluate = async () => {
    setBusy(true); setError('');
    try {
      const evaluated = await api.evaluate(date, session, version);
      setResult((old) => ({ ...(old || {} as SessionResult), ...evaluated, forecast_type: version, session_id: session, evaluation_date: date }));
      setStatus(`Evaluation refreshed · ${evaluated.completed_count ?? evaluated.predictions.filter((p) => p.actual).length} complete · ${evaluated.pending_count} pending.`);
      await loadSession(true);
    } catch (e) { setError(e instanceof Error ? e.message : 'Outcome refresh failed.'); }
    finally { setBusy(false); }
  };

  const handleRefresh = async () => {
    await refreshStatus(true);
    await loadSession(true);
  };

  const handleDataRefresh = async () => {
    setBusy(true); setError('');
    try {
      const operation = await api.refreshData();
      setStatus(`Data refresh queued · ${operation.operation_id}`);
      await refreshStatus(false);
    } catch (e) { setError(e instanceof Error ? e.message : 'Data refresh could not start.'); }
    finally { setBusy(false); }
  };

  const onSessionChange = (value: SessionId) => {
    setSession(value);
    if (value === 'morning') setVersion('ADVANCE');
  };

  const controls = <SessionControls date={date} session={session} version={version} updatedAvailable={updatedAvailable} busy={busy || sessionLoading}
    onDate={setDate} onSession={onSessionChange} onVersion={setVersion} onGenerate={() => void handleGenerate()}
    onLoad={() => void loadSession()} onEvaluate={() => void handleEvaluate()} exportUrl={predictions.length ? api.exportUrl(date, session, version) : undefined} />;

  return <div className="app-shell">
    <BrandHeader date={date} data={data} refreshing={refreshing} onRefresh={() => void handleRefresh()} status={status} />
    <nav className="primary-nav" aria-label="Main navigation">{(['Dashboard', 'Actual vs Predicted', 'Data & Model'] as Page[]).map((item) =>
      <button key={item} className={page === item ? 'active' : ''} onClick={() => setPage(item)}>{item}</button>)}</nav>
    <main className="page-content">
      {error && <div className="alert error-alert" role="alert"><span>{error}</span><button onClick={() => setError('')} aria-label="Dismiss error">×</button></div>}
      {page === 'Dashboard' && <>
        <div className="dashboard-grid">
          <div className="dashboard-primary">
            <section className="panel forecast-panel dashboard-chart-panel">
              <div className="section-heading"><div><span className="eyebrow">ACTUAL MARKET · 10-MINUTE OHLC</span><h2>AMZN price action</h2>
                <p>{formatDate(date)} · {session} {version} · complete actual candles with stored direction markers</p></div>
                <div className="chart-legend"><span><i className="legend-dot up" />↑ UP</span><span><i className="legend-dot down" />↓ DOWN</span><span>{candles.length} actual bars</span></div></div>
              {sessionLoading ? <div className="empty-state">Loading actual candles and predictions…</div> : candles.length
                ? <ActualChart candles={candles} predictions={ordered} height={500} />
                : <div className="empty-state"><b>No complete actual candles for this selection.</b><span>Actual candles are shown only when complete local 5-minute bars are available to aggregate.</span></div>}
              <div className="chart-footer"><span>ACTUAL OHLC · NO SYNTHETIC CANDLES</span><span>Hover candle/marker for interval, raw score and outcome</span><span>Support · Resistance · Reference</span></div>
            </section>
          </div>
          <aside className="dashboard-aside">
            <PredictionPanel prediction={nextPrediction} modelVersion={result?.model_version || nextPrediction?.model_version} source={snapshot?.source || data?.source} latestCandle={snapshot?.latest_timestamp || data?.latest_completed_rth_candle} />
            <SessionDecision result={result} />
          </aside>
        </div>
        {controls}
        <PredictionHistory predictions={ordered} />
      </>}

      {page === 'Actual vs Predicted' && <>
        <section className="page-title"><div><span className="eyebrow">HISTORICAL SESSION BROWSER</span><h1>Actual vs Predicted</h1><p>Browse stored forecasts and compare them with completed market data for the same session.</p></div>
          <span className="date-pill">{formatDate(date)} · {session} · {version}</span></section>
        {controls}
        <ComparisonView candles={candles} predictions={ordered} />
        <PredictionHistory predictions={ordered} />
      </>}

      {page === 'Data & Model' && <>
        <section className="page-title"><div><span className="eyebrow">LOCAL DATA SNAPSHOT</span><h1>Data & Model</h1><p>Refreshes use the existing Alpaca IEX backend operation. The UI never receives provider credentials.</p></div>
          <button className="button primary" disabled={busy} onClick={() => void handleDataRefresh()}>{busy ? 'Starting…' : 'Refresh market data'}</button></section>
        <div className="data-grid panel">
          <div><small>SOURCE</small><b>{data?.source || snapshot?.source || '—'}</b></div>
          <div><small>FEED</small><b>{data?.feed || snapshot?.feed || '—'}</b></div>
          <div><small>PROCESSED ROWS</small><b>{data?.rows?.toLocaleString() || snapshot?.dataset_rows?.toLocaleString() || '—'}</b></div>
          <div><small>FIRST CANDLE</small><b>{formatTime(data?.dataset_start, true)}</b></div>
          <div><small>LAST COMPLETE CANDLE</small><b>{formatTime(data?.latest_completed_rth_candle || snapshot?.latest_timestamp, true)}</b></div>
          <div><small>LAST SUCCESSFUL SYNC</small><b>{formatTime(data?.last_sync_at, true)}</b></div>
          <div className="wide-tile"><small>PROCESSED SNAPSHOT</small><b className="path-value">{data?.path || 'No local snapshot reported'}</b></div>
        </div>
        <section className="panel operations-panel"><div className="section-heading"><div><span className="eyebrow">PERSISTENT BACKGROUND WORK</span><h2>Recent operations</h2></div><button className="button secondary" onClick={() => void refreshStatus()}>Reload status</button></div>
          {operations.length ? operations.slice(0, 12).map((op) => <article className="operation-row" key={op.operation_id}><div><b>{op.kind.replaceAll('_', ' ')}</b><small>{op.stage.replaceAll('_', ' ')} · {formatTime(op.updated_at, true)}</small>{op.error && <span className="down-copy">{op.error}</span>}</div><span className={`operation-state ${op.status.toLowerCase()}`}>{op.status}</span><div className="progress-track"><i style={{ width: `${Math.max(0, Math.min(100, op.progress * 100))}%` }} /></div></article>) : <div className="empty-state">No background operations are recorded.</div>}
        </section>
      </>}
      <footer className="app-footer">Academic prediction visualization · raw scores are not calibrated confidence · no order execution</footer>
    </main>
  </div>;
}

type MainView = 'terminal' | 'research' | 'sessions';

export default function App() {
  const [view, setView] = useState<MainView>('terminal');
  return <div className="root-terminal-app">
    <header className="global-terminal-nav">
      <div className="global-brand"><span className="global-brand-mark">A</span><span><b>AMZN DECISION TERMINAL</b><small>AI trading analysis · research visualization</small></span></div>
      <nav aria-label="Application sections">
        <button className={view === 'terminal' ? 'active' : ''} onClick={() => setView('terminal')}>Terminal</button>
        <button className={view === 'research' ? 'active' : ''} onClick={() => setView('research')}>Historical research</button>
        <button className={view === 'sessions' ? 'active' : ''} onClick={() => setView('sessions')}>Session forecasts</button>
      </nav>
      <span className="global-mode">PREDICTION ONLY · NO ORDER EXECUTION</span>
    </header>
    {view === 'terminal' && <TerminalDashboard />}
    {view === 'research' && <ResearchPerformance />}
    {view === 'sessions' && <SessionForecasts />}
  </div>;
}
