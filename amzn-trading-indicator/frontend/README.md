# AMZN AI Indicator frontend

React, TypeScript, Vite, and TradingView Lightweight Charts. The default view is a TradingView-inspired decision terminal using persisted analysis-engine records and actual completed local AMZN candles. The original 10-minute session-prediction workflow remains under **Session forecasts**.

## Data sources and display rules

- The chart reads `/api/market/candles?source=local` and plots genuine completed 5-minute OHLCV bars. The API exposes 5-minute candles only, so the UI does not pretend other timeframes are available.
- Decision, evidence, contribution, regime, risk-plan, and lifecycle fields come from `/api/analysis/latest`, `/api/analysis/signals`, and `/api/analysis/signals/{signal_id}`. The UI does not run analysis during render; **Analyze latest candle** explicitly calls `POST /api/analysis/run`.
- The 40-feature LightGBM output is displayed as a raw, uncalibrated directional score. The combined decision score is a heuristic, not a probability.
- EMA, SMA, Bollinger, and session VWAP overlays are calculated from the actual plotted candles. Session VWAP uses typical price `(high + low + close) / 3` weighted by bar volume.
- Horizontal support/resistance and hypothetical stop/target levels are read from the stored analysis. No synthetic future candles, order buttons, broker fills, options flow, CVD, or gamma values are shown.
- Historical performance is a static view of the saved local chronological replay artifact in `reports/experiments/trading_analysis_engine_audit/`. It is labeled research, includes the 10 bps cost assumption, and does not represent live P&L.
- Market refresh is a user-triggered call to the backend's configured IEX refresh operation. The frontend never receives provider credentials.

## Run

Copy `.env.example` to `.env.local` if the API is not at the default address, then set `VITE_API_BASE_URL`. It must point to the FastAPI `/api` root. Start the backend from the repository root, then run:

```powershell
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

Vite can fall back to port 5174 when 5173 is occupied; the backend allows local Vite origins on ports 5170–5179.

The available npm scripts are `dev`, `build`, and `preview`; there is no configured frontend lint or unit-test script. Run `npm run build` for TypeScript and production-build verification.
