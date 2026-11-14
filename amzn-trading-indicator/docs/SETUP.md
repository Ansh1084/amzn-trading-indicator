# Setup and local operation (Windows + MSYS2 UCRT64)

Use three terminals for a full demonstration: FastAPI, React/Vite, and the independent backend scheduler. The frontend no longer triggers scheduled forecasts. The scheduler is not started by Uvicorn, so it must be launched separately when automatic workflows are wanted.

## One-time setup

From MSYS2 UCRT64:

```bash
cd "/c/Users/anshg/Desktop/SEM5/Projects/ML2 - Copy/amzn-trading-indicator"
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with backend-only `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`. Never commit or expose it. Install frontend packages once:

```bash
cd frontend
npm install
cd ..
```

If no suitable processed data exists, refresh a historical IEX snapshot with `python -m src.data.ingestion --days 400`. Training an evaluation-date model remains a distinct action and requires the previous session’s final completed candle. Existing snapshots and model artifacts are not overwritten.

## Start services

### Terminal 1: API

```bash
cd "/c/Users/anshg/Desktop/SEM5/Projects/ML2 - Copy/amzn-trading-indicator"
source .venv/Scripts/activate
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

### Terminal 2: website

```bash
cd "/c/Users/anshg/Desktop/SEM5/Projects/ML2 - Copy/amzn-trading-indicator/frontend"
npm run dev -- --host 127.0.0.1 --port 5173
```

Open the Vite URL printed in the terminal. If port 5173 is already occupied, Vite automatically selects 5174; the backend CORS rule allows local `localhost` and `127.0.0.1` origins on ports 5170–5179. The API health endpoint is `http://127.0.0.1:8000/health`; API docs are `http://127.0.0.1:8000/docs`.

### Terminal 3: automatic scheduler

```bash
cd "/c/Users/anshg/Desktop/SEM5/Projects/ML2 - Copy/amzn-trading-indicator"
source .venv/Scripts/activate
python -m backend.app.scheduler
```

Keep this terminal open. Press `Ctrl+C` to stop it gracefully. Run only one copy; SQLite leases prevent two active scheduler owners. The scheduler stores/reconstructs jobs in `artifacts/forecast_log.sqlite3`, checks for missed issuance deadlines, and retries with bounded backoff. It uses a standard-library persistent polling loop (no APScheduler package is required). If the scheduler is stopped, manual workflow APIs/buttons remain available.

## Forecast workflow

Select the target trading date (ET) in the website. The **Predict Next Trading Day** action or the scheduler runs the same orchestrator. After the previous session close, it refreshes/validates data, verifies a registered model or trains it through that cutoff, then stores ADVANCE morning and afternoon forecasts. An ADVANCE afternoon does not use target-day morning bars.

After target date’s morning closes and all 36 expected bars are available, the scheduler or **Generate Updated Afternoon** action stores a separate UPDATED forecast. It uses the same prior-close afternoon model plus completed morning observations. It never retrains on the target day. The afternoon ADVANCE remains unchanged. Comparison runs after each window only when all actual bars exist; it can also be triggered manually.

The configured windows are 09:30–12:30 and 12:30–15:30 ET. Scheduled work uses `America/New_York`; the current market calendar is deterministic but not an authoritative live exchange calendar. Configured windows extending beyond a scheduled early close are skipped or rejected.

## Useful API calls

- `POST /api/workflows/next-day` body `{"evaluation_date":"YYYY-MM-DD"}`
- `POST /api/workflows/afternoon-update` body `{"evaluation_date":"YYYY-MM-DD"}`
- `POST /api/workflows/compare` body `{"evaluation_date":"YYYY-MM-DD"}`
- `GET /api/workflows/{operation_id}` and `GET /api/workflows/{operation_id}/events`
- `GET /api/events?evaluation_date=YYYY-MM-DD` (SSE)
- `GET /api/scheduler/status`
- `GET /api/forecasts?evaluation_date=YYYY-MM-DD`
- `GET /api/forecasts/{forecast_id}`

The older `POST /api/forecasts/issue` and `POST /api/forecasts/compare` routes remain for compatibility.

## Persistence and migration

CSV snapshots remain under `data/raw/` and `data/processed/`; versioned models remain under `artifacts/window_models/<evaluation-date>/`. SQLite holds forecasts, actual comparisons, pipeline logs, operations, SSE events, scheduled jobs and a scheduler lease.

On the first database access after upgrade, the migration creates an online SQLite backup named `artifacts/forecast_log.sqlite3.pre-v2-<UTC timestamp>.bak`, adds the `forecast_versions` and workflow/scheduler tables, and copies prior forecasts as `LEGACY` records. It retains the old `forecasts` and `pipeline_logs` tables and does not delete data. Tests use temporary databases and do not migrate the user database.

## Validation

Run offline tests with:

```bash
python -m pytest tests --ignore=tests/test_health.py -q
```

In the inspected Windows environment, `tests/test_health.py` stalls inside the existing Starlette `TestClient` health request; that pre-existing environment issue is tracked separately. Frontend production build:

```bash
cd frontend
npm run build
```

See [`DEMO_RUNBOOK.md`](DEMO_RUNBOOK.md) for a complete presentation procedure and troubleshooting.


## Live AMZN chart feed

The FastAPI process owns one Alpaca IEX streaming connection. It reads `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` from the backend `.env`; keys are never sent to Vite. On startup it seeds `artifacts/market_stream.sqlite3` from the newest processed snapshot (insert-only), then starts Alpaca's AMZN one-minute bar subscription. Genuine minute bars update a visibly marked forming 5-minute aggregate only. Forming bars are excluded from indicator and ML inputs. Alpaca does not provide a native 5-minute bar subscription in this client, so provider-confirmed completed 5-minute bars are reconciled through the existing historical-bars API at startup and every 60 seconds. Completed bars are upserted by UTC start timestamp; a provider correction updates that timestamp. A missing interval is logged and never filled with a synthetic candle.

The backend publishes candle, analysis, signal, gap, status and synchronization events from the persistent market event log at `GET /api/market/events`. Event IDs support replay after browser reconnect. The dashboard has one `EventSource`; it applies candles incrementally and uses a timestamp delta request when a hidden tab becomes visible. The existing forecast scheduler remains separate and is not duplicated. Stopping FastAPI closes its stream and reconciliation task. Alpaca's SDK manages bounded socket reconnects; on reconnect the service performs an immediate historical overlap reconciliation, then resumes its periodic recovery.

Check `GET /api/market/stream/status` for provider/feed, real connection state, market state, last received minute event, last completed/persisted bar, last reconciliation, freshness and errors. `GET /api/market/candles` returns completed candles plus a forming bar only while recent updates are arriving in an open session. The market calendar is the project's existing deterministic calendar helper; it covers configured regular holidays and scheduled early closes, but cannot know unscheduled exchange closures. A delayed/missing IEX bar is reported as stale or as a gap; no price is substituted. IEX is a limited exchange feed, so its observed volume and updates can differ from SIP/consolidated data.

If credentials are missing, the dashboard continues to show the local snapshot and the stream status is `DISCONNECTED`; no live data request is made. To inspect status without printing secrets, call the status endpoint or check startup logs. To verify the latest persisted completed candle, compare `last_persisted_candle` from stream status with the final `timestamp` from `/api/market/candles`.
