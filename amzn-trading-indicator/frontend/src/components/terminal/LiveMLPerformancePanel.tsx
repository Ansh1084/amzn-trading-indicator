import { useEffect, useState, useCallback } from 'react';
import { analysisApi } from '../../services/analysisApi';
import type { MLPerformanceResponse } from '../../types/analysis';
import { formatTime } from '../../utils/format';

interface Props {
  lastUpdated?: string;
}

export function LiveMLPerformancePanel({ lastUpdated }: Props) {
  const [windowSize, setWindowSize] = useState<'50' | '100' | '500' | 'all'>('50');
  const [data, setData] = useState<MLPerformanceResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchPerformance = useCallback(async (win: '50' | '100' | '500' | 'all') => {
    setLoading(true);
    setError(null);
    try {
      const resp = await analysisApi.mlPerformance(win, true);
      setData(resp);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch ML performance metrics');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchPerformance(windowSize);
  }, [windowSize, lastUpdated, fetchPerformance]);

  const summary = data?.summary;
  const cm = data?.confusion_matrix;
  const prob = data?.probability_evaluation;
  const model = data?.model_identifier;

  return (
    <section className="tp-card tp-ml-perf-card" aria-label="Live ML Model Performance">
      {/* ── CARD HEADER & WINDOW CONTROLS ──────────────────────────────────── */}
      <div className="tp-card-header">
        <div className="tp-card-header-left">
          <h3 className="tp-heading">Live ML Model Performance</h3>
          <span className="tp-status-pill pill-cyan">LIVE OUT-OF-SAMPLE</span>
        </div>

        <div className="tp-window-selector" role="group" aria-label="Evaluation Window Selector">
          {(['50', '100', '500', 'all'] as const).map((win) => (
            <button
              key={win}
              type="button"
              className={`tp-win-btn ${windowSize === win ? 'tp-win-btn-active' : ''}`}
              onClick={() => setWindowSize(win)}
              disabled={loading}
            >
              {win === 'all' ? 'All Live' : `Last ${win}`}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="tp-alert-box tone-neg">
          <span>Failed to load live ML evaluation: {error}</span>
          <button type="button" className="tp-retry-btn" onClick={() => fetchPerformance(windowSize)}>
            Retry
          </button>
        </div>
      )}

      {/* ── EMPTY STATE WHEN INSUFFICIENT DATA ──────────────────────────────── */}
      {!loading && (!data?.has_data || data?.sample_count === 0) ? (
        <div className="tp-empty-card tp-ml-empty-box">
          <div className="tp-empty-icon">📊</div>
          <h4 className="tp-empty-title">Insufficient Live Evaluation Data</h4>
          <p className="tp-empty-desc">
            {data?.empty_reason ||
              'Metrics will appear after enough live predictions have matured against actual subsequent market candles.'}
          </p>
          <div className="tp-empty-meta">
            <span>Model: <b>{model?.model_name || 'LGBMClassifier'}</b> ({model?.model_version || 'frozen'})</span>
            <span>Horizon: <b>10 Minutes</b> (2 × 5m bars)</span>
            <span>Tracked: <b>{data?.total_predictions_tracked || 0}</b></span>
            <span>Pending: <b>{data?.pending_count || 0}</b></span>
          </div>
        </div>
      ) : (
        <>
          {/* ── SUMMARY METRICS TILES ───────────────────────────────────────── */}
          {summary && (
            <div className="tp-ml-summary-grid">
              <div className="tp-ml-metric-card">
                <span className="tp-tile-label">Accuracy</span>
                <b className={`tp-ml-metric-val ${summary.accuracy >= 0.5 ? 'tone-pos' : 'tone-muted'}`}>
                  {(summary.accuracy * 100).toFixed(1)}%
                </b>
                <span className="tp-tile-sub">
                  {summary.correct} / {summary.total_resolved} hits
                </span>
              </div>

              <div className="tp-ml-metric-card">
                <span className="tp-tile-label">Macro F1</span>
                <b className="tp-ml-metric-val">{summary.macro_f1.toFixed(3)}</b>
                <span className="tp-tile-sub">Weighted: {summary.weighted_f1.toFixed(3)}</span>
              </div>

              <div className="tp-ml-metric-card">
                <span className="tp-tile-label">Precision</span>
                <b className="tp-ml-metric-val">{(summary.precision_overall * 100).toFixed(1)}%</b>
                <span className="tp-tile-sub">
                  UP: {data?.metrics ? (data.metrics.up.precision * 100).toFixed(1) : '—'}% · DOWN:{' '}
                  {data?.metrics ? (data.metrics.down.precision * 100).toFixed(1) : '—'}%
                </span>
              </div>

              <div className="tp-ml-metric-card">
                <span className="tp-tile-label">Recall</span>
                <b className="tp-ml-metric-val">{(summary.recall_overall * 100).toFixed(1)}%</b>
                <span className="tp-tile-sub">
                  UP: {data?.metrics ? (data.metrics.up.recall * 100).toFixed(1) : '—'}% · DOWN:{' '}
                  {data?.metrics ? (data.metrics.down.recall * 100).toFixed(1) : '—'}%
                </span>
              </div>

              <div className="tp-ml-metric-card">
                <span className="tp-tile-label">Resolved Predictions</span>
                <b className="tp-ml-metric-val">{summary.total_resolved}</b>
                <span className="tp-tile-sub">
                  {data?.is_partial
                    ? `${summary.total_resolved} of ${data.requested_window_size} requested`
                    : '100% window size'}
                </span>
              </div>
            </div>
          )}

          {/* ── DETAILED METRICS TABLE & CONFUSION MATRIX SPLIT ─────────────── */}
          <div className="tp-ml-details-split">
            {/* Detailed Table */}
            <div className="tp-ml-table-container">
              <div className="tp-ml-section-subhead">
                <span className="tp-subhead-title">Detailed Performance Breakdown</span>
                <span className="tp-subhead-tag">UP = Positive Class</span>
              </div>
              <table className="tp-ml-table">
                <thead>
                  <tr>
                    <th scope="col">Metric</th>
                    <th scope="col" className="text-right">UP</th>
                    <th scope="col" className="text-right">DOWN</th>
                    <th scope="col" className="text-right">Overall</th>
                  </tr>
                </thead>
                <tbody>
                  {data?.detailed_table.map((row) => (
                    <tr key={row.metric}>
                      <td className="tp-td-metric" title={row.tooltip || ''}>
                        {row.metric}
                      </td>
                      <td className="text-right tp-td-val">{row.up}</td>
                      <td className="text-right tp-td-val">{row.down}</td>
                      <td className="text-right tp-td-val font-semibold">{row.overall}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Compact Confusion Matrix */}
            <div className="tp-ml-cm-container">
              <div className="tp-ml-section-subhead">
                <span className="tp-subhead-title">Confusion Matrix</span>
                <span className="tp-subhead-tag">Out-of-Sample Outcomes</span>
              </div>

              {cm ? (
                <div className="tp-cm-grid">
                  <div className="tp-cm-header-cell">Actual \ Pred</div>
                  <div className="tp-cm-header-cell text-center">Pred UP</div>
                  <div className="tp-cm-header-cell text-center">Pred DOWN</div>
                  <div className="tp-cm-header-cell text-center">Actual Total</div>

                  {/* Actual UP Row */}
                  <div className="tp-cm-label-cell">Actual UP</div>
                  <div className="tp-cm-data-cell cell-tp" title="True Positive (Actual UP, Pred UP)">
                    <span className="cm-num">{cm.true_up_pred_up}</span>
                    <span className="cm-tag">True UP</span>
                  </div>
                  <div className="tp-cm-data-cell cell-fn" title="False Negative (Actual UP, Pred DOWN)">
                    <span className="cm-num">{cm.true_up_pred_down}</span>
                    <span className="cm-tag">False DOWN</span>
                  </div>
                  <div className="tp-cm-total-cell font-semibold">{cm.total_actual_up}</div>

                  {/* Actual DOWN Row */}
                  <div className="tp-cm-label-cell">Actual DOWN</div>
                  <div className="tp-cm-data-cell cell-fp" title="False Positive (Actual DOWN, Pred UP)">
                    <span className="cm-num">{cm.true_down_pred_up}</span>
                    <span className="cm-tag">False UP</span>
                  </div>
                  <div className="tp-cm-data-cell cell-tn" title="True Negative (Actual DOWN, Pred DOWN)">
                    <span className="cm-num">{cm.true_down_pred_down}</span>
                    <span className="cm-tag">True DOWN</span>
                  </div>
                  <div className="tp-cm-total-cell font-semibold">{cm.total_actual_down}</div>

                  {/* Marginal Pred Row */}
                  <div className="tp-cm-label-cell font-semibold">Pred Total</div>
                  <div className="tp-cm-total-cell font-semibold">{cm.total_pred_up}</div>
                  <div className="tp-cm-total-cell font-semibold">{cm.total_pred_down}</div>
                  <div className="tp-cm-grand-total font-bold">{cm.total_resolved}</div>
                </div>
              ) : (
                <div className="tp-empty-desc">No confusion matrix data.</div>
              )}

              {/* Probability & Scoring Notice */}
              {prob && (
                <div className="tp-prob-eval-box">
                  <div className="tp-prob-metrics-row">
                    <div className="tp-prob-chip">
                      <span className="chip-label">Log Loss:</span>
                      <b className="chip-val">{prob.log_loss.toFixed(4)}</b>
                    </div>
                    <div className="tp-prob-chip">
                      <span className="chip-label">Brier Score:</span>
                      <b className="chip-val">{prob.brier_score.toFixed(4)}</b>
                    </div>
                    <div className="tp-prob-chip">
                      <span className="chip-label">ECE:</span>
                      <b className="chip-val">{prob.expected_calibration_error.toFixed(4)}</b>
                    </div>
                  </div>
                  <div className="tp-prob-notice">
                    <span className="tp-notice-icon">ℹ</span>
                    <span>
                      <b>Raw LightGBM Scores:</b> Uncalibrated probabilities evaluated directly against binary outcomes. Not calibrated confidence.
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* ── EVALUATION METADATA FOOTER ───────────────────────────────────── */}
          <div className="tp-ml-meta-footer">
            <div className="tp-meta-col">
              <span className="tp-meta-label">Model Pipeline</span>
              <span className="tp-meta-val">
                {model?.model_name || 'LGBMClassifier'} ({model?.model_version || 'legacy-10m'}) · 40 features
              </span>
            </div>

            <div className="tp-meta-col">
              <span className="tp-meta-label">Prediction Horizon</span>
              <span className="tp-meta-val">
                10-Minute Close vs Feature Candle Close (2 × 5m bars)
              </span>
            </div>

            <div className="tp-meta-col">
              <span className="tp-meta-label">Evaluation Period</span>
              <span className="tp-meta-val">
                {data?.start_date ? formatTime(data.start_date) : '—'} →{' '}
                {data?.end_date ? formatTime(data.end_date) : '—'}
              </span>
            </div>

            <div className="tp-meta-col">
              <span className="tp-meta-label">Store Status</span>
              <span className="tp-meta-val">
                <b>{data?.total_resolved_count || 0}</b> resolved · <b>{data?.pending_count || 0}</b> pending ·{' '}
                <b>{data?.excluded_count || 0}</b> excluded
              </span>
            </div>
          </div>
        </>
      )}
    </section>
  );
}
