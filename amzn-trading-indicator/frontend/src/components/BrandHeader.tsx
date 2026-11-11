import type { DataStatus } from '../types';
import { formatDate, formatTime, marketClockStatus } from '../utils/format';

export function BrandHeader({ date, data, refreshing, onRefresh, status }: {
  date: string; data: DataStatus | null; refreshing: boolean; onRefresh: () => void; status?: string;
}) {
  const market = marketClockStatus(data?.latest_completed_rth_candle);
  return <header className="topbar">
    <div className="brand-lockup">
      <div className="brand-mark">A</div>
      <div><strong>AMZN AI Indicator</strong><small>10-Minute Market Prediction Dashboard</small></div>
    </div>
    <div className="header-meta">
      <span className={`status-chip ${market.state}`}><i />{market.label}</span>
      <span className="header-date">{formatDate(date)}</span>
      <span className="freshness">DATA · {data?.latest_completed_rth_candle ? formatTime(data.latest_completed_rth_candle, true) : '—'}</span>
      <button className="icon-button" onClick={onRefresh} disabled={refreshing} title="Refresh local snapshot status">
        {refreshing ? <span className="spinner" /> : <span aria-hidden="true">↻</span>} Refresh
      </button>
    </div>
    {status && <div className="header-status" role="status">{status}</div>}
  </header>;
}
