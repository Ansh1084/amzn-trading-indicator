# AMZN Trading Analysis & Decision Engine

This backend-only module adds a transparent analysis layer around completed local AMZN candles. It does not place orders and does not replace, retrain, or modify the production 40-feature LightGBM artifact.

## API

- `POST /api/analysis/run?source=local` analyzes the newest completed local snapshot and stores one immutable analysis record per candle and analysis version.
- `GET /api/analysis/latest` returns the most recently stored analysis without recalculating it.
- `GET /api/analysis/signals` lists stored analysis records (`start`, `end`, `status`, `limit`, `offset` filters).
- `GET /api/analysis/active` lists active hypothetical setups; `GET /api/analysis/outcomes` lists terminal signal outcomes.
- `GET /api/analysis/signals/{signal_id}` returns the stored record and lifecycle events.
- `POST /api/analysis/outcomes/resolve?source=local` resolves hypothetical active setups against completed local bars. The existing scheduler also calls the resolver periodically.

All levels and decision outputs are research estimates, not financial advice. The risk plan is hypothetical; no broker or order API is called. ML `predict_proba` is retained as an **uncalibrated raw score**, and does not represent a probability of success.

## Existing project audit and data flow

The existing IEX retrieval and local snapshot pipeline in `src/data/ingestion.py` supplies completed 5-minute OHLCV candles; `src/forecasts/calendar.py` supplies the market calendar, timezone, and early-close rules. Existing feature engineering remains separate and unchanged. The recovered production inference adapter in `backend/app/services/analysis/ml_evidence.py` delegates to `src/features/legacy_engineering.py` and the frozen production loader in `src/forecasts/legacy_sessions.py`; it verifies the 40-feature manifest and emits raw, uncalibrated evidence. Existing support-level logic was a prior swing-point utility, so this engine reuses the causal 2x2 pivot principle while adding ATR-clustered zones and breakout state. The existing SQLite store/migration path in `src/forecasts/store.py` is extended additively instead of creating another database or signal scheduler.

Data flow: processed snapshot -> completed/session/calendar validation -> causal indicators -> confirmed pivots/zones and regime -> read-only production-model evidence -> weighted conservative decision and hypothetical risk plan -> immutable per-candle SQLite signal -> scheduler/API outcome resolver over later completed bars.

## Method

The analysis rejects duplicate timestamps, invalid OHLC relationships, and off-grid timestamps; it excludes non-session and early-close-overrun rows using the existing market calendar, and excludes incomplete 5-minute candles. It compares the latest bar to the expected latest completed market bar (calendar-aware) and rejects a lag greater than 15 minutes, including stale previous-session snapshots. Missing intraday intervals within the same session are returned in `data_gaps` (ordinary overnight/session breaks are not reported as gaps). Indicators are causal: EMA/SMA, session-reset RSI/ATR, MACD, ROC, Bollinger bands, rolling volatility, volume ratios, and trend alignment. Divergence compares consecutive confirmed 2x2 pivots against RSI-7 and is emitted only on the second pivot's confirmation bar. Nearby levels are clustered within 0.35 ATR; one touch per session counts toward strength. Breakout confirmation requires a close beyond the zone by at least 0.15 ATR and relative volume of at least 1.20; crossing checks use all historical zones, while failed-breakout labels require the initiating close-cross within the configured 12-bar lookback.

The weighted decision score combines trend (0.25), momentum (0.20), volume (0.10), structure (0.25), raw production-model evidence (0.10), and price action (0.10). These values, thresholds, swing windows, stale-data threshold, risk distances and cost assumption load from `configs/analysis_engine.json` at process startup. Weights must sum to 1. Missing ML evidence is omitted and the remaining weights are renormalized. The response reports normalized component weights, contribution values, and top contributors. This score is a heuristic, not a probability. Actionable BUY/SELL output additionally requires score threshold, non-transitional regime, no unresolved evidence conflict, a valid risk plan, at least 1.5 target-2 R/R, and no unconfirmed nearby opposing level. Otherwise the result is WAIT; a same-direction active setup still between its stop and target 2 is HOLD.

Stops use a nearby confirmed zone with a 0.25 ATR buffer or a 1.25 ATR fallback. Targets use the next confirmed zone when sufficiently distant, otherwise 1R/2R. These are scenario levels, not guaranteed execution prices.

Outcome resolution examines subsequent completed 5-minute bars. The signal close is a reference, not an assumed fill; historical evaluation enters at the next same-session candle open only when that open remains inside the planned entry zone. Otherwise the setup expires without P&L. A signal at the final candle of a session expires without an overnight fill. Stop is considered first if a bar touches both stop and target 2 because bar-level data cannot establish intrabar order. Target 1 is an informational event only; the hypothetical position is not partially exited. Terminal stop, target 2, session-end, or 180-minute time-limit outcomes record gross return from the assumed entry open and an explicitly assumed 10-basis-point round-trip cost (5 bps each side); net return is only a scenario under that assumption. Unresolved or missing intervals remain pending. Duplicate analysis and terminal evaluation are guarded by SQLite uniqueness/status checks.

SQLite schema version 4 adds `analysis_signals` and `analysis_signal_events` to the existing database; existing forecast data is retained, with the store's pre-migration backup behavior.

## Limitations

- This module has no separate calibrated ML model or probability estimates.
- Level strength and combined decision scores are heuristic and need chronological backtesting before being treated as useful predictive evidence.
- Candle-based outcome resolution cannot identify the order of intrabar barrier touches; it uses a conservative stop-first rule.
- Cost assumptions are configurable constants, not measured AMZN execution costs; reported net scenario returns are not verified trading performance.
- The analysis API currently accepts the local processed snapshot only; no live Alpaca request is made by these endpoints.
- Data gaps are surfaced; the current indicator computations do not impute missing candles. Users should avoid interpreting signals across materially incomplete sessions.

## Historical local-data smoke example

Using the local processed snapshot with an explicit historical cutoff at the latest row's completion (`2026-11-09T20:55:00Z`, 3:55 PM ET), the engine returned close `266.93`, bias `BEARISH`, heuristic score `-0.619`, production raw UP score `0.6548`, and final decision `WAIT FOR CONFIRMATION` because the computed regime was transitional. The different ML/combined directions illustrate that the classifier is only one evidence input. This was a reproducibility smoke check, not a backtest or performance claim; current-date API execution rejects that snapshot as stale until the existing refresh flow supplies recent data.
