# Website User Guide

This guide follows the current React application in `frontend/src/main.tsx` and the API implementation in `backend/app/api/routes.py`. The product displays research forecasts and evaluation results. It does not place trades or orders. A visible status is not necessarily a guarantee that the external feed is live: the chart uses a local processed snapshot, while data refresh and forecast actions request Alpaca IEX data from the backend.

## Before opening the site

Start the FastAPI backend and Vite frontend as described in [`DEMO_RUNBOOK.md`](DEMO_RUNBOOK.md). The frontend expects the API at `http://localhost:8000/api`; the backend CORS allowlist includes frontend port 5173. Open `http://127.0.0.1:5173`. Backend health is `http://127.0.0.1:8000/health`; API exploration is `http://127.0.0.1:8000/docs`.

The default selected evaluation date is the current date in New York time. Use a scheduled market day with a registered eligible model for a real issuance demonstration. The date picker is an evaluation-date selector, not a control that trains a model or fetches historical data by itself.

## Navigation and pages

The left navigation in the single React component switches among Dashboard, Data & Model, Actual vs predicted, Pipeline logs, and Model evaluation. Dashboard data loads on entry, when the target date changes, when you use **Refresh Dashboard**, and after a workflow completes. SSE streams report manual and scheduled workflow progress; if an operation-specific stream disconnects, the UI requests only that operation’s status while it is running.

The top bar has an **Evaluation date** input. Changing it reloads the forecasts and model status for that date; it does not change the candle dataset, retrain a model, or backdate the current clock. The symbol card is informational and shows the latest local candle price. The sidebar footer states “Prediction only / No order execution.”

### Dashboard

**Summary cards.** Latest completed price is the close of the newest local processed RTH candle. Model readiness reflects `/api/model/status` for the selected date. Frozen forecast count is the number of SQLite forecast records for that date. Dataset indicates the selected local snapshot and chart row counts; it is not a live-connection indicator.

**AMZN price action chart.** The chart displays 5-minute candles and volume histogram, with time labels formatted in America/New_York. Green/red candles indicate close relative to open; volume bars share the time axis. The current-price label comes from the latest loaded local candle. The chart takes the most recent 2,000 candles by default (the API supports a larger limit), and supports zooming and horizontal scroll. It is a historical snapshot refreshed on dashboard entry, selected-date change, explicit refresh, and workflow completion; it is not a WebSocket stream.

When a forecast exists and is selected, horizontal lines mark expected price, predicted range lower/upper, support and resistance; forecast start/end markers identify the window. These lines are frozen forecast values and do not move with new candles. Forecast levels only appear after issuance. The chart uses the selected forecast; selecting a forecast card changes which forecast is emphasized.

**Forecast windows and types.** Defaults are `morning` 09:30–12:30 ET and `afternoon` 12:30–15:30 ET. The dashboard shows Morning ADVANCE, Afternoon ADVANCE and Afternoon UPDATED separately. ADVANCE is created after the prior session closes from prior-session-only information; UPDATED can be created after all 36 morning bars are complete using the same afternoon model plus those morning observations. Cards show type, status, direction, expected price movement, range, support/resistance, support-hold and direction probabilities when available, conditional downside, generation/data/training timestamps and model version. Click a forecast card or history item to select it; its immutable overlay appears on the chart.

**Predict Next Trading Day** posts the selected target date to `/api/workflows/next-day`. It starts an asynchronous operation that refreshes IEX data, verifies the per-date model or trains it through the previous session cutoff, and issues both ADVANCE records. A repeated successful request returns the existing workflow/forecasts; it does not replace them. **Generate Updated Afternoon** posts to `/api/workflows/afternoon-update`; it is accepted only after 12:30 ET, with all morning bars complete, the afternoon model intact, and before the afternoon window ends. It never retrains the model. If a configured window runs beyond an early close, that window is skipped or rejected.

The older `POST /api/forecasts/issue?source=alpaca` route remains for compatibility and retains its prior near-window rules. The current website uses the new workflow buttons for ADVANCE and UPDATED records.

After the configured window end, `Compare completed window` remains available as a manual action. The backend scheduler also starts comparison after each window and waits/retries if actual candles are missing. Each version compares independently against the same actual window and is stored separately; the forecast is not changed. The Actual vs predicted page reports paired ADVANCE-versus-UPDATED performance only after both afternoon outcomes exist.

**Pipeline activity panel.** Shows recent log events in newest-first order, model/data/forecast summary, and a link to detailed logs. Workflow updates arrive through server-sent events (SSE); “LIVE” does not mean a streaming price feed. `Active operations` shows data refresh, training and forecast workflow jobs that are currently RUNNING, with stage and progress. Completed or failed tasks remain inspectable in Data & Model and operations/log API history.

**Forecast workflow panel.** Summarizes the persistent scheduler state and recent workflow operations. Use **Predict Next Trading Day**, **Generate Updated Afternoon**, or **Compare completed window** to start a workflow manually. Scheduled workflows run in the separate backend scheduler process, even when the dashboard is closed.

### Data & Model

**Fetch Latest Data.** Click to call `POST /api/data/refresh`. The backend starts `refresh_latest_market_data` in a background worker and returns an operation ID. The page polls `/api/operations`; status includes stage, progress, result or error. It fetches completed Alpaca IEX 5-minute bars, validates and merges into new date-stamped raw and processed CSV snapshots. It does not train a model or issue forecasts. Existing CSVs are preserved. Refresh can return no new bars when the market has not produced new completed candles. Incremental fetching starts after the latest raw timestamp and does not repair arbitrary older missing gaps.

The data summary reports snapshot source/path, rows, RTH start/end, latest successful sync, and displayed chart count. Validation checks schema, timestamps/timezone, OHLCV relationships, duplicates and session hours; gaps are reported and not filled with invented candles. A refresh may fail if credentials, network, schema or historical feed access is unavailable.

**Train Model.** Click only after selecting the intended future evaluation date and refreshing/confirming complete previous-session data. `POST /api/model/train?evaluation_date=YYYY-MM-DD` starts background training. It is blocked when the date is not a scheduled trading day, forecasts already exist, a model is already registered, or the latest processed candle is not the expected final completed candle of the prior session. It writes one window model per configured forecast window and metadata. Training does not fetch data or issue forecasts.

The details show version, trained time, dataset range, exact cutoff, row/feature counts, SHA-256 integrity and eligibility. “READY” requires model artifacts to load and match hashes, metadata dataset end not past cutoff, and training earlier than the first window. A registered date is protected from overwrite. If it is ineligible or an artifact is missing/corrupt, inspect `GET /api/model/status?evaluation_date=...` and operation errors; do not delete artifacts casually.

**Model versions and operations.** The page can show versions from `/api/model/versions` and recent refresh/training operations. Operation status is persisted in SQLite. A backend restart marks any formerly RUNNING operation interrupted/FAILED; refresh or training must be launched again if appropriate.

### Actual vs predicted

This page lists each frozen forecast version for the selected date and its actual payload once comparison is complete. ADVANCE and UPDATED are separate records; their paired afternoon analysis uses the same observed window. It distinguishes not-yet-compared records from completed outcomes. The outcome comes from exact candles in `[window_start, window_end)`:

- Actual direction is UP if final close is at least the forecast reference close; otherwise DOWN.
- Actual range is minimum candle low to maximum candle high.
- Direction result is a boolean match.
- Range coverage is true only if the whole observed low/high range fits inside the predicted bounds.
- Boundary errors are observed low minus predicted lower bound, and observed high minus predicted upper bound.
- Movement error is actual end-price change from reference minus predicted dollar movement.
- `mae_boundary` is the average of absolute lower- and upper-boundary errors in price units.
- Support “held” means no 5-minute close in the window fell below support; a wick below support alone does not count. If no support was selected, current code treats it as held.
- Resistance “reached” means an observed high touched or exceeded the selected resistance.

The backend refuses comparison before the window end, if actual candles are missing/duplicated, or after an outcome is already stored. It obtains actual bars from Alpaca IEX by default. This page is not a guarantee of predictive skill; it displays individual realized outcomes.

### Pipeline logs

The detailed table is sourced from `GET /api/pipeline-logs`. The website shows all loaded rows (newest first); the endpoint also supports filtering by forecast ID. Fields include log ID, evaluation date, window, stage, timestamp (rendered ET), JSON input/output summaries, model version, data cutoff, rule, status and error detail. Issuance creates stages for data cutoff, retrieval, preprocessing, feature generation, model prediction, levels, confidence and chart/frozen output. Comparison appends `actual_comparison`. Refresh/training operation updates are also logged. Failed issuance may be recorded under an `attempt-<date>-<window>` ID because no forecast ID was created.

The logging schema is structured, but it does not promise a separate log row for every low-level library call or every individual HTTP candle. Error detail is stored when available; some unexpected errors are reduced to the exception class for an issuance attempt.

### Model evaluation

The evaluation page presents registered model walk-forward metrics, not just the latest individual forecast. Each window has fold count and a table with validation date range, sample count, directional accuracy, majority baseline, balanced accuracy, F1, end-return MAE (shown as percent), and range coverage. Below it, support probability Brier score, scored sample count and coverage are shown, or the page reports insufficient calibration samples. Results are historical estimates and do not establish future returns. Training/fold metrics and the live evaluation-day result are different populations and should be described separately.

## Understanding the visible numbers

- **Direction probability** is only shown when a direction calibrator is available; otherwise it is null/unavailable rather than a made-up confidence. Direction itself can still be derived using the raw classifier score.
- **Support-hold probability** is an empirical conditional estimate based on prior comparable support-distance/ATR cases; it is not direction confidence. It is withheld until the minimum calibration sample count is met.
- **Expected movement** is the model’s expected end-return converted to dollars from the reference price; it is not the range width.
- **Predicted range** is derived from lower/upper quantile regressors for the window’s future lows/highs. Coverage asks whether both observed extrema fit inside it.
- **Downside scenario** is conditional on a 5-minute close below support. It uses the next lower confirmed swing when available; otherwise it uses an ATR-based fallback. No calibrated probability of that downside scenario is currently produced.

All prices, levels, probabilities and ranges are estimates. Missing or insufficient calibration data is shown as unavailable; no estimate is a guarantee.

## Forecast lifecycle and timing

1. Keep the backend scheduler running. It persists upcoming jobs and can recover eligible missed jobs after a restart, subject to configured retry/deadline rules.
2. Refresh local history and verify the previous session’s last completed candle.
3. Train/register the evaluation-date models using only data through the previous trading-session close. Artifacts are frozen for that evaluation date.
4. The scheduler creates each ADVANCE forecast before its configured window. The UPDATED afternoon forecast is generated after the morning window closes and before the afternoon window starts; it may use completed morning bars but keeps the same frozen model.
5. The backend checks candle freshness/completeness, verifies artifact hashes and stores each forecast as an immutable version with cutoffs and pipeline events.
6. Prices may continue to refresh in the browser. They do not mutate a frozen forecast.
7. After window end, the scheduler compares eligible forecasts automatically; manual comparison is also available. Exact complete actual bars are required and each outcome is appended once.

The backend scheduler, not the open browser, initiates scheduled issuance and comparisons. Check **Pipeline logs** for job state and workflow results. The dashboard can start workflows manually and receives progress through SSE.

## Error messages and practical limits

- **Model not ready**: no registered model, failed hash check, cutoff violation, or trained too late. Check the selected date and model status.
- **Previous-session final candle missing**: refresh/inspect the selected snapshot; model training is intentionally blocked rather than trained on partial cutoff data.
- **Forecast must be generated strictly before its window begins**: issue during the configured lead time; a historical date or late attempt cannot issue.
- **Market data stale/incomplete**: refresh data or check Alpaca availability; the forecast needs completed, expected bars.
- **Actual window incomplete**: wait until the whole window has ended and all 5-minute actual candles are available; retry comparison after verifying feed.
- **No forecast appears after leaving page closed**: confirm `python -m backend.app.scheduler` is running, inspect `/api/scheduler/status` and pipeline logs, then review the job deadline/retry state. An already-started forecast window cannot be issued retroactively.
- **No log rows**: there may have been no operations or forecast attempts. Check the selected date and `GET /api/pipeline-logs`.
- **Chart blank or API errors**: confirm the API and local processed CSV exist; inspect browser console and backend logs. Vite must use port 5173 under current CORS settings.
- **Early-close day**: configured windows may run beyond the session close and be rejected; current schedule is not an authoritative live exchange calendar.

For run commands and a guided presentation sequence, see [`DEMO_RUNBOOK.md`](DEMO_RUNBOOK.md). For implementation details and source map, see [`PROJECT_ARCHITECTURE.md`](PROJECT_ARCHITECTURE.md) and [`ML_AND_FORECASTING.md`](ML_AND_FORECASTING.md).
