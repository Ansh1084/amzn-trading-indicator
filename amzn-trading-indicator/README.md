# AMZN Trading Indicator

Academic AMZN intraday forecasting and evaluation system. The legacy 10-minute classifier remains available separately; the final path generates frozen 3-hour ADVANCE forecasts from previous-session data and a distinct UPDATED afternoon forecast after the morning session. The application displays estimates only and does not place orders.

## Architecture
- `data/`: local raw and processed market data (CSV files excluded from Git).
- `src/data/`: ingestion, validation, preprocessing and chronological splitting.
- `src/features/`: shared causal feature engineering for training and inference.
- `src/targets/`: target construction.
- `src/models/`: training and model selection (LightGBM first; LSTM/Lorentzian KNN experiments).
- `src/evaluation/`: metrics, leakage-safe evaluation and paper-trading simulation.
- `src/inference/`: shared live prediction pipeline.
- `backend/`: FastAPI service, shared forecast orchestrator, and separately run persistent scheduler.
- `frontend/`: React + TypeScript + TradingView Lightweight Charts.
- `artifacts/`: local model artifacts (not committed by default).
- `configs/`, `tests/`, `reports/`, `docs/`.

## Rules
Never use `future_close_10m` or the target as model inputs. Split chronologically and purge boundary labels. Fit transformations on training data only. Share feature logic between training and inference. Finalized signals must never repaint. Keep credentials in `.env`.
See `docs/SETUP.md` for setup and the forecast workflow. Forecast windows and scheduler times are configured in `configs/forecast_windows.json` (defaults: 09:30–12:30 and 12:30–15:30 America/New_York).

## Window forecast workflow

1. Configure `.env` with backend-only Alpaca IEX credentials and prepare processed history.
2. Start the FastAPI API and Vite frontend, then start the dedicated scheduler with `python -m backend.app.scheduler` in a third terminal when automated workflows are wanted.
3. Select a target trading date and run **Predict Next Trading Day**, or let the scheduler run after the previous session close. The shared orchestrator saves morning and afternoon ADVANCE forecasts.
4. After the complete target-day morning window, the scheduler or **Generate Updated Afternoon** stores an UPDATED forecast using the frozen afternoon model and completed morning inputs.
5. Compare forecasts after their windows complete. ADVANCE and UPDATED afternoon outcomes are evaluated independently against the same actual window.

The refresh command writes date-stamped files so existing datasets remain untouched. Training refuses stale/incomplete data and will not silently replace a registered evaluation-date model. Forecasts, versions, comparisons, operations and scheduler events are persisted in `artifacts/forecast_log.sqlite3`; a one-time additive migration takes an online backup and retains the old tables. Model artifacts are stored under `artifacts/window_models/<evaluation-date>/`. These local artifacts are git-ignored.

The project calendar includes configured regular holidays and scheduled early closes; unscheduled closures are not covered. FastAPI owns one Alpaca IEX stream for forming-candle updates and reconciles provider-confirmed 5-minute bars through historical data. The separate forecast scheduler remains a distinct process. The terminal uses replayable market SSE and incremental chart updates.
