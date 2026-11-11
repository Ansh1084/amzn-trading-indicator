# 10-minute session prediction and evaluation

The session prediction path uses the separate feature implementation recovered from Git commit `e1b5cf0` and the existing `model_lightgbm.joblib` artifact. The current 36-feature 3-hour forecasting pipeline and its artifacts are unchanged. At load time the service checks the legacy 40-feature manifest, artifact input count and binary classes. Each frozen schedule stores a SHA-256-derived model version, feature timestamp, data cutoff, prediction ID, signal, raw UP score, and causal support/resistance levels.

## Schedule versions

- **ADVANCE** uses only candles before the target trading date and is rejected once its selected session starts. It uses the previous session's last completed bar as its frozen feature snapshot and varies the model's time-of-day input for each scheduled 10-minute slot. No same-day target-session candles enter it.
- **UPDATED** is only available for the afternoon and requires the exact 36 completed morning 5-minute bars. It uses the afternoon model artifact unchanged and the morning-completed snapshot. It is stored as a separate immutable version alongside ADVANCE.
- The legacy classifier is binary. There is no trained NEUTRAL class. UP/DOWN scores are displayed as raw, uncalibrated model scores, not confidence. No future OHLC predictions are fabricated.
- Normal sessions have 18 slots in each half-day window. On scheduled early closes, the slot schedule ends at the exchange close (for example, three afternoon intervals on a 13:00 close).

## Actuals and automatic evaluation

The scheduler polls the latest local processed candle snapshot on each tick. An interval is evaluated only when its end time has passed and every exact 5-minute timestamp from the start through the 10-minute target endpoint exists. Incomplete or delayed intervals remain pending and are retried on a later tick. The conditional SQLite update persists each actual once, including actual direction, endpoint close, return, outcome, and evaluation timestamp. The scheduler does not call Alpaca as part of this evaluation poll; local data must be refreshed by the existing data-refresh action/workflow.

The Actual vs Predicted page lets users choose the dashboard evaluation date, morning/afternoon session, and ADVANCE/UPDATED version. Its comparison chart uses genuine local 5-minute OHLCV aggregated into actual 10-minute candles and places prediction markers on available timestamps. Pending slots remain visible in the table even if no actual candle exists yet. CSV export is available from the session panel.

## Endpoints

- `GET /api/session-predictions?evaluation_date=YYYY-MM-DD&session_id=morning|afternoon&forecast_type=ADVANCE|UPDATED`
- `POST /api/session-predictions/generate` with `evaluation_date`, `session_id`, and `forecast_type`
- `POST /api/session-predictions/evaluate` for a one-time/manual local check; regular automatic checks are performed by the scheduler
- `GET /api/session-predictions/actual-candles?evaluation_date=YYYY-MM-DD&session_id=...`
- `GET /api/session-predictions/export.csv?...`

The SQLite upgrade to schema v3 adds `session_predictions`; on an existing pre-v3 database, the usual online backup is made before the migration. The prediction table is persistent across API/scheduler restarts and uses a uniqueness constraint over date, session, type, slot and revision.

## Run locally

Use three terminals from the repository root (frontend from its subdirectory):

```powershell
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

```powershell
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

```powershell
.venv\Scripts\python.exe -m backend.app.scheduler
```

The scheduler uses local data for automatic comparison polling. Alpaca refresh is a separate existing operation and requires configured credentials. Legacy model out-of-sample accuracy is near chance and below the test majority baseline; the dashboard is a research visualization, not evidence of predictive advantage.
