# AMZN Trading Indicator: Demo Runbook

Commands below are for Windows with the MSYS2 UCRT64 terminal. They assume the repository is at `/c/Users/anshg/Desktop/SEM5/Projects/ML2 - Copy/amzn-trading-indicator`. Run the API, frontend and (for automatic workflows) scheduler in separate terminals. Do not display or paste `.env` values during a presentation.

## One-time setup

From the repository root in MSYS2 UCRT64:

```bash
cd "/c/Users/anshg/Desktop/SEM5/Projects/ML2 - Copy/amzn-trading-indicator"
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
```

Create `.env` from `.env.example` using a text editor and set backend-only `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`. Keep `.env` local and private. Install the frontend packages once:

```bash
cd frontend
npm install
cd ..
```

Python compatibility depends on installed packages; `docs/SETUP.md` currently gives Python 3.14 as the Windows setup version. If a package has no compatible wheel, use a Python version supported by `requirements.txt` rather than changing project code during the demo.

## Start the application

### Terminal 1 — backend

```bash
cd "/c/Users/anshg/Desktop/SEM5/Projects/ML2 - Copy/amzn-trading-indicator"
source .venv/Scripts/activate
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Leave this terminal running. It should report that Uvicorn is listening on `http://127.0.0.1:8000`.

### Terminal 2 — frontend

```bash
cd "/c/Users/anshg/Desktop/SEM5/Projects/ML2 - Copy/amzn-trading-indicator/frontend"
npm run dev -- --host 127.0.0.1 --port 5173
```

Leave this terminal running. Open **http://127.0.0.1:5173**. The API health page is **http://127.0.0.1:8000/health** and interactive API documentation is **http://127.0.0.1:8000/docs**. Current backend CORS allows port 5173; if Vite reports another port because 5173 is occupied, stop the conflicting process or use an intentionally configured CORS/API setup before presenting. The frontend API base is `http://localhost:8000/api`.

### Terminal 3 — backend scheduler

```bash
cd "/c/Users/anshg/Desktop/SEM5/Projects/ML2 - Copy/amzn-trading-indicator"
source .venv/Scripts/activate
python -m backend.app.scheduler
```

This process is independent of Uvicorn and the browser. Keep it running for scheduled forecast generation and comparisons; use `Ctrl+C` to stop. SQLite leases guard against a second scheduler process.

Expected health JSON: `{"status":"ok"}`. The dashboard should load the API-driven summary and local chart if a processed CSV is available. A blank chart with no local processed data is not evidence that Alpaca refresh succeeded.

## Recommended demonstration procedure

### Phase 1 — startup and connectivity

1. Start both terminals as above.
2. Open `/health` and `/docs` to demonstrate backend availability.
3. Open the dashboard and confirm the AMZN chart, latest local candle, model status and configured windows appear.
4. Explain that the chart uses the local processed snapshot and refreshes on dashboard load, explicit refresh and workflow completion; it is not a live streaming chart.

### Phase 2 — data preparation

1. Navigate to **Data & Model**.
2. Review the snapshot path, row count, RTH date range and latest sync before refreshing.
3. Click **Fetch Latest Data** only if internet access and Alpaca credentials are configured. Explain that this starts an asynchronous operation and writes new date-stamped CSV snapshots.
4. Watch the operation stage until `COMPLETED` or inspect its error. The result may add zero rows outside the feed’s latest completed timestamp.
5. Explain that validation detects invalid OHLCV, timestamps, duplicates and gaps; it does not synthesize missing candles or guarantee repair of older gaps.

### Phase 3 — model readiness

1. Confirm the selected evaluation date is the intended scheduled market day.
2. Inspect registered model version, training time, exact cutoff, dataset end/range, feature count and artifact integrity.
3. The cutoff must be the previous scheduled session’s close. The training operation requires the previous session’s final completed candle in the latest local dataset.
4. If no eligible model exists, train before the evaluation window/day using **Train Model**. Explain that training is a separate asynchronous action and refuses to overwrite an existing version or train after a forecast was issued.
5. Do not attempt last-minute same-day training: the readiness check requires it to have completed before the first configured window. Do not train a retrospective model and present it as a genuine pre-session model.

### Phase 4 — forecast generation

1. For an actual issuance demonstration, select a future/current New York evaluation date and verify fresh prior-session data. The **Predict Next Trading Day** control starts the same orchestrator used by the scheduler; it refreshes data, verifies or trains through the previous session cutoff, and issues morning and afternoon ADVANCE records.
2. The default windows are 09:30–12:30 and 12:30–15:30 ET. Scheduled ADVANCE forecasts are produced after the previous session close. The afternoon ADVANCE uses only prior-session information.
3. After all 36 morning candles are complete and before 15:30 ET, **Generate Updated Afternoon** can issue an UPDATED forecast from the same frozen afternoon model plus completed morning observations. It does not modify the ADVANCE forecast or retrain.
4. The backend scheduler runs independently of the browser; verify its state and scheduled jobs in **Pipeline logs** or `GET /api/scheduler/status`. Manual buttons remain available if the scheduler is not running, subject to timing and completeness checks.
5. On issuance, explain the reference price, predicted direction, expected dollar movement, quantile range, support/resistance, support-hold probability (or “Not calibrated”), downside scenario, generation time, latest input candle/data cutoff, training cutoff and model version.
6. Explain the distinction: the trained model is from previous-day data; an UPDATED afternoon forecast may use completed morning observations. Price refreshes do not change either stored prediction.

Scheduled jobs have deadlines and bounded retries; a window that has started is not forecast retroactively.

### Phase 5 — actual evaluation

1. Wait until the forecast’s end timestamp has passed and all expected actual 5-minute bars are available.
2. The backend scheduler attempts comparisons after each window; for a manual demonstration, click **Compare completed window**. Comparison uses Alpaca IEX by default and rejects incomplete actual windows.
3. Show the Actual vs predicted page. Explain actual end direction, observed high/low range, coverage, price errors, support close-break result and resistance touch result.
4. State the meaning and units for each metric. Distinguish historical walk-forward metrics from this one evaluation-day outcome.
5. The result is recorded separately from the original forecast and cannot be overwritten through the normal flow.

### Phase 6 — logs and model evaluation

1. Open **Pipeline logs** and locate stages by date/window/forecast ID.
2. Explain the data cutoff, retrieval, preprocessing, feature count, prediction, level and confidence rules, frozen output and later comparison entries.
3. Failed issuance attempts can have `attempt-...` IDs rather than forecast IDs. Refresh/training events use operation IDs and appear under window `operations`.
4. Open **Model evaluation** and explain that its fold metrics are chronological historical estimates. Do not call them live trading results or claim the old 10-minute classifier is a 3-hour model.

## Before-demo checklist

- [ ] Run from the intended repository checkout and check `git status`; do not use unreviewed code edits during the presentation.
- [ ] Start both servers and confirm health/API connectivity and frontend port 5173.
- [ ] Confirm `.env` exists privately without showing credentials; check Alpaca connectivity only if the demo requires a live refresh/forecast/compare.
- [ ] Confirm a non-empty processed local CSV exists so the chart can render.
- [ ] For a genuine forecast, choose a future/current evaluation date that has not already been issued, verify its eligible model metadata and artifact hashes, and ensure the current time is in the configured issuance period.
- [ ] Confirm latest complete candle coverage and any gaps before the session/window.
- [ ] Know whether the target date is an early-close day; current config windows extend to 15:30 and may be invalid if the session closes early.
- [ ] If outside market hours or no live forecast can be generated, present stored historical forecasts/results as examples and label them as historical. Do not issue a fabricated forecast or pretend a historical model was trained before the evaluation day.
- [ ] Keep the backend scheduler process running if demonstrating automatic issuance/comparison; the browser can be closed.
- [ ] Avoid destructive cleanup of `artifacts/`, CSVs, or SQLite; those contain models and records.

## Troubleshooting

| Symptom | Verify | Action |
|---|---|---|
| Backend fails to start | Terminal traceback; Python env active; `python -m uvicorn` available | Activate `.venv/Scripts/activate`, install `requirements.txt`, run from repo root. Backend entry point is `backend.app.main:app`. |
| Frontend fails to start | Node/npm version and Vite output | Run `npm install` once in `frontend`, then `npm run dev -- --host 127.0.0.1 --port 5173`. |
| Browser shows API/CORS error | Check `/health`, backend port, browser Network panel and Vite port | Backend API is port 8000 and CORS currently allows frontend port 5173. Do not leave Vite on auto-selected 5174 without adjusting setup. |
| Alpaca authentication/connection error | Backend operation or response; confirm variable names are set without printing values | Set `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` in backend `.env`; restart backend. Never put credentials in frontend. |
| Refresh adds no bars | Inspect returned operation result and last local timestamp | No new completed bars may be normal. The dashboard chart reads the local snapshot. Refresh uses incremental bars and does not fix older holes. |
| Data refresh fails | `GET /api/operations`, response error, `GET /api/data/status` | Resolve credentials/network/feed/schema errors; retry after operation ends. Refresh is asynchronous and duplicate concurrent refresh is blocked. |
| Model training rejected | API response and `GET /api/model/status?evaluation_date=...` | Confirm trading date, no prior forecast or registered model, and exact prior-session final candle in processed dataset. Refresh/repair data before training; do not bypass cutoff checks. |
| Model missing/corrupt/not ready | Model status `artifact_status`, model metadata JSON and operation error | Restore the matching locally generated model artifact from backup or retrain only if no model is registered/forecasts issued and cutoff conditions are satisfied. Hash mismatch means do not use the artifact. |
| No forecast appears | Forecast list, model status, scheduler status, time in ET, logs | Check that the scheduler is running and job is within its deadline/retry budget, or use the manual workflow control if eligible. If the window started, no retrospective issuance is allowed. |
| Forecast issuance failed | `/api/pipeline-logs`, failure response | Read error details; common causes are outside lead time, stale/missing candles, model hash/cutoff failure, incomplete history, or early-close window conflict. |
| Comparison unavailable | Current ET time, forecast end, complete actual bar count | Wait until window has ended and Alpaca has all expected candles; route rejects partial windows and duplicate comparisons. |
| Chart blank | `/api/data/status`, `/api/market/candles?source=local`, browser console | Ensure a local processed snapshot is present and backend responds. The chart uses local CSV; Alpaca credentials alone do not provide chart data until refresh has saved a snapshot. |
| Pipeline logs empty | `GET /api/pipeline-logs`; SQLite path; no actions performed yet | Logs are created by operations/forecast attempts/comparisons. Check the correct backend checkout and `artifacts/forecast_log.sqlite3`; no events means there may be nothing to display. |
| Operation stuck or interrupted | `GET /api/operations` and backend restart history | On startup, stale `RUNNING` operations are marked failed/interrupted. Relaunch only after checking that no active worker is still running. |
| Windows/holiday date disagrees | Compare with official market calendar | The helper covers common holidays and scheduled half-days only; unscheduled closures are not modeled. Validate the date externally and do not rely on a stale static calendar for production. |

## Useful endpoints

```text
GET  http://127.0.0.1:8000/health
GET  http://127.0.0.1:8000/docs
GET  http://127.0.0.1:8000/api/config
GET  http://127.0.0.1:8000/api/data/status
GET  http://127.0.0.1:8000/api/market/candles?limit=2000&source=local
GET  http://127.0.0.1:8000/api/model/status?evaluation_date=YYYY-MM-DD
GET  http://127.0.0.1:8000/api/forecasts?evaluation_date=YYYY-MM-DD
GET  http://127.0.0.1:8000/api/pipeline-logs
GET  http://127.0.0.1:8000/api/operations
GET  http://127.0.0.1:8000/api/scheduler/status
GET  http://127.0.0.1:8000/api/workflows/{operation_id}
```

Forecast issue/compare and data refresh are POST operations; use `/docs` to inspect request schemas. Do not use Alpaca endpoints for a demo unless credentials and network are intentionally available.

## Concise application lifecycle

Alpaca IEX completed AMZN 5-minute bars are validated and saved as local CSV snapshots. For an evaluation date, separate per-window models are trained chronologically through the previous trading session and stored with metadata and hashes. The backend scheduler creates ADVANCE forecasts before each configured window and can create an UPDATED afternoon forecast using completed morning candles with the same frozen model. It stores immutable forecast versions and pipeline events in SQLite. After each window, scheduled or manual comparison uses complete actual candles and records results without changing forecasts. The React dashboard reads records and receives workflow updates through SSE.
