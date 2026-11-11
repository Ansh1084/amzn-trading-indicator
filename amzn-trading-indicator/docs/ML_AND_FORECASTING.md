# Machine Learning and Forecasting

This explains the connected 3-hour window forecasting path and distinguishes it from the older 10-minute experiment. The implementation is an academic prediction/evaluation system; predictions are not financial guarantees or instructions to trade.

## Data and time conventions

Inputs are 5-minute OHLCV bars for AMZN, with `symbol`, `timestamp`, `open`, `high`, `low`, `close`, `volume`, and Alpaca fields such as `trade_count` and bar `vwap`. RTH is 09:30 inclusive to session close exclusive in `America/New_York`; on a standard day bars start 09:30 through 15:55 (78 bars). The internal data path uses UTC-aware timestamps, then converts to New York time to group sessions and display labels. This handles daylight-saving offsets without hardcoding UTC-4/UTC-5.

`src/data/preprocessing.py` checks required columns, timestamp validity, OHLCV consistency, duplicates and gaps, sorts rows and filters sessions. It reports missing intervals rather than fabricating bars. Identical duplicate handling and invalid-row policy are explicit in validation reports. Static early closes are listed in the preprocessor; this is not a full live exchange calendar.

`src/data/ingestion.py` retrieves historical bars from Alpaca IEX, excludes a still-forming 5-minute bar, and writes timestamped CSV snapshots. Refresh merges only newer data onto the latest raw snapshot, validates, and emits new raw/processed files. Dashboard data status/chart are local-snapshot reads; the forecast and comparison API defaults to Alpaca. API credentials remain backend environment variables.

## Feature engineering

`src/features/engineering.py` computes one causal row per candle. `FEATURE_COLUMNS` is the model-input allowlist (36 columns); inputs are selected from that list rather than from every dataframe column. The legacy target fields and future outcomes are not model inputs.

| Feature group | Implemented values and interpretation |
|---|---|
| Returns/price change | `return_1`, `return_2`, `return_3`, `log_return_1`, `price_change`: recent price changes. |
| Candle shape | `candle_body`, `candle_range`, `upper_wick`, `lower_wick`, `body_to_range`: open/close and wick structure. |
| Moving averages | `sma_ratio_5/10/20`, `ema_ratio_5/10/20`: close relative to simple/exponential averages. |
| MACD | `macd`, `macd_signal`, `macd_hist`: trend/momentum differences and histogram. |
| RSI and momentum | `rsi_14`, `roc_5`, `roc_10`: relative momentum; first-candle RSI handling avoids treating overnight gap as an intraday previous close. |
| Volatility/bands | `atr_14`, `bb_width`, `bb_position`: true-range volatility and Bollinger width/location. |
| Volume | `volume_change`, `volume_sma_5/20`, `volume_ratio_5/20`: recent activity relative to rolling volume. |
| VWAP | `vwap_distance`: close relative to VWAP. Session VWAP uses volume-weighted bar VWAP where supplied, falling back to close if necessary. |
| Time | `hour`, `minute`, `day_of_week`, `time_of_day_sin`, `time_of_day_cos`: intraday and weekday context. |

Indicators are calculated from current/past candles. Some rolling averages/MACD use continuous history (`reset_daily=False` during forecast inference), while session-dependent values use New York session boundaries. NaN/infinite latest input values are filled with training medians, then zero as a last fallback; the fitted model does not learn from future rows.

## Two different target problems

### Legacy 10-minute classification

`src/targets/labeling.py` constructs a future close return over an exact 10-minute horizon. It requires a same-day/RTH future timestamp and labels up/down relative to a threshold, with neutral/missing cases represented as missing. `src/models/train.py` uses this task for experiment models (Dummy, Logistic Regression, Random Forest, HistGradientBoosting and optional LightGBM). It uses chronological session splits, a purge around boundaries, training-only imputation and a scaler pipeline for Logistic Regression. This is **not** a 3-hour prediction, and its metrics must not be presented as results for the window system.

### Final configured-window forecast

`src/forecasts/training.py` builds a separate example for each configured window and historical anchor. For a window such as 09:30–12:30, the input is the last allowed pre-window feature row. The outputs represent:

1. Direction: whether the end-of-window close is at least the reference/anchor price (UP vs DOWN).
2. End return: fractional change from the reference price to the window ending close.
3. Lower window return: reference-to-minimum-low return during the window.
4. Upper window return: reference-to-maximum-high return during the window.

The training builder checks expected bars, horizon/session limits and feature history. A window crossing the session close cannot produce a valid example/forecast. Each configured window gets its own fitted model bundle. The morning forecast uses only prior-session/current pre-open context; the afternoon forecast can use completed observations before 12:30. Neither uses candles from its own future window.

## Cutoff, model freeze, validation and artifacts

For evaluation date `D`, `src/forecasts/calendar.py` finds the previous scheduled trading day, accounting for weekends and the coded common holidays. `cutoff_for_evaluation` is the prior session close converted to UTC. Training filters data through that cutoff and requires the previous session’s final completed candle. The backend data-refresh button and model-training button are separate actions.

The final path uses chronological expanding folds (no random shuffle). Fold boundaries expand through the historical examples; the initial train portion is 40%, with validation boundaries around 55%, 70%, 85% and 100%. Models are evaluated on later portions than their fold training data. The final fitted models then use all eligible examples through the cutoff. No evaluation-day window outcomes enter training. Reported walk-forward metrics are historical fold estimates, not the current day’s result.

Per window, the configured final models are:

- `HistGradientBoostingClassifier`: 120 iterations, `max_leaf_nodes=15`, `l2_regularization=1.0`, `random_state=42` for direction.
- `HistGradientBoostingRegressor`: same iteration/leaf/regularization/random-state settings for end return.
- `GradientBoostingRegressor` quantile models: alpha 0.10 and 0.90, 100 estimators and depth 2 for lower/upper future-window price extrema returns.

The training run records window metrics including directional accuracy, majority baseline, balanced accuracy, F1, end-return MAE and range coverage. Metadata records the model version, evaluation date, training cutoff, trained time, training data range/count, exact feature names, configuration, fold metrics and artifact SHA-256 values. Joblib model files and `metadata.json` are written under `artifacts/window_models/<evaluation-date>/`. Existing date artifacts are not overwritten. On status/forecast load, the code verifies the recorded artifact hashes and cutoff metadata.

The registered models are intended to remain frozen during the evaluation day. Issuance loads the date/window bundle and does not train or update it. Forecast issuance is prevented from substituting a newer model, and a registered model cannot be overwritten by the UI training route.

## ADVANCE and UPDATED forecast versions

Each evaluation date can have one `ADVANCE` per configured window and one `UPDATED` afternoon forecast. `revision` is recorded as 1 for these first-generation records; the storage model has a separate forecast ID and optional parent ID. A failed operation may be retried, but a successful type/date/window forecast is returned on a duplicate trigger instead of replaced.

- **Morning ADVANCE:** generated after the previous session’s close and before 09:30 ET. The service drops any later candles from the input and verifies the previous session’s final completed bar. The input cutoff is the timestamp of that last candle; the model training cutoff is the exact previous-session close.
- **Afternoon ADVANCE:** generated in that same next-day workflow, still using only history through the previous session close. It does not use target-day morning candles.
- **Afternoon UPDATED:** generated after 12:30 ET only after the backend fetch contains exactly the expected 36 completed bars from 09:30 to 12:25. Its inference cutoff is the final morning candle, and its features include prior history plus the completed morning. The service loads the same registered afternoon model/hash as ADVANCE; it does not call training. The forecast points to its ADVANCE parent when that record exists.

ADVANCE and UPDATED values are stored as separate immutable payloads. Their actual comparisons are also stored separately and are evaluated against the same 12:30–15:30 actual window. The API reports paired direction, absolute movement-error, boundary-MAE and range-coverage changes only after both comparisons exist. Those changes describe one pair of outcomes, not a statistically reliable claim that updates improve future forecasts.

The app database uses a new `forecast_versions` table. At first use, an online SQLite backup is made, old forecasts are copied with `forecast_type=LEGACY`, and old tables remain in place. The new uniqueness key includes forecast type, so ADVANCE and UPDATED can share a target date/window while duplicates of the same type are rejected.

## Forecast issuance rule and leakage controls

Configuration in `configs/forecast_windows.json` currently says:

```json
{
  "timezone": "America/New_York",
  "break_rule": "5m_close_below_support",
  "issuance_lead_minutes": 2,
  "windows": [
    {"id": "morning", "start": "09:30", "end": "12:30"},
    {"id": "afternoon", "start": "12:30", "end": "15:30"}
  ]
}
```

`issue_window_forecast` enforces rules by forecast type. Legacy `INTRADAY` retains the old near-window lead interval. ADVANCE is allowed after the prior session close and before its target window; input candles are capped at that previous close. UPDATED is afternoon-only and only allowed after the morning is complete. For every type, timestamps are timezone-aware, incomplete bars (`timestamp + 5 minutes > now`) are excluded, duplicates are rejected, and the matching date/window model hash and training cutoff are checked. Features are built from permitted history and only its latest row is passed to inference.

The resulting payload retains `forecast_type`, `revision`, optional parent ID, `generated_at`, `window_start`, `window_end`, `data_cutoff` (latest candle start timestamp used), `feature_generation_timestamp`, `training_cutoff`, model version/hash, reference price and forecast values. It is inserted once into SQLite with associated stage logs. A unique evaluation-date/window/type key prevents replacement while allowing ADVANCE and UPDATED to coexist. Incoming prices do not update the payload. Comparison is rejected until end and stores `actual_payload` in a separate column exactly once.

The dedicated scheduler runs as `python -m backend.app.scheduler`, separate from Uvicorn. It persists/reconstructs jobs in SQLite, uses a renewable lease to prevent duplicate scheduler processes, handles ET/DST-aware times, and retries bounded failures. The frontend receives date-scoped events over SSE and fetches dashboard data on demand and after workflow completion. A browser may close without stopping scheduled jobs. Missed ADVANCE deadlines are marked skipped; stale/incomplete data does not produce a forecast.

**Remaining limits:** the calendar is static and cannot guarantee unscheduled closures. Incomplete/missing data blocks issuance rather than being filled. Historical training code can be invoked directly for a date; eligibility checks the training timestamp and window cutoff, so retrospective artifacts should not be presented as genuinely trained before the evaluation session.

## Support, resistance and downside

`src/forecasts/levels.py` identifies confirmed 2-left/2-right swing lows/highs from the prior history. Since both right-side bars precede the forecast anchor, the swing is known when selected. It restricts the search to the latest 20 sessions, filters candidates to within 3 times `max(ATR, 0.1% of reference price)`, then chooses the nearest support below and nearest resistance above reference. If no candidate qualifies, the level is null; it does not invent a pivot.

Support-holding is defined by the configured `5m_close_below_support`: support is broken if any candle closes below the selected line during the forecast window. A low wick below support without a close does not constitute a break. Historical calibration compares prior examples with support distance/ATR within ±0.5 of the current ratio. It needs at least 30 comparable samples. The probability is a Jeffreys-smoothed historical frequency, `(holds + 0.5)/(samples + 1)`, which avoids literal 0%/100% from finite samples. If support, ATR, or sample count is insufficient, probability is null and the UI says “Not calibrated.” It is not derived from classifier confidence.

The downside case is conditional on a 5-minute close below support. If prior confirmed lower swing support exists, the estimated band runs from that next support to current support. Otherwise a fallback uses roughly one ATR below support, with a 0.5%-of-reference minimum width. The output is explicitly an estimate; the code does not provide a statistically calibrated chance of reaching this downside band.

## Probabilities and evaluation metrics

The direction classifier produces a raw UP score. With enough out-of-fold probabilities and both classes present, a Platt logistic calibrator is fitted from chronological out-of-fold scores. The training report evaluates calibration on a later portion of those OOF scores, then fits the stored calibrator on the available OOF values for inference. Without a usable calibrator, `direction_probability_up` is null and the raw score is used only to pick a direction; it is not presented as a calibrated probability. The report’s calibration evaluation must be interpreted separately from a held-out evaluation-day score.

| Metric | Meaning here |
|---|---|
| Accuracy / directional accuracy | Share of samples/windows whose up/down label is correct. |
| Majority baseline | Accuracy obtained by always choosing the most common training direction; useful context for class imbalance. |
| Balanced accuracy | Mean recall across classes, reducing the benefit of simply predicting the more frequent class. |
| Precision | Of predicted positive/UP examples, fraction actually UP. |
| Recall | Of actual UP examples, fraction predicted UP. |
| F1 | Harmonic mean of UP precision and recall; does not assess price range size. |
| MAE | Mean absolute error. In fold evaluation it is end-return error (fractional return); actual `mae_boundary` is a dollar-price boundary error. State which one is being discussed. |
| RMSE | Not a primary metric in the current final window training report. Do not claim it is displayed unless the actual UI/code is updated. |
| Range coverage | Fraction of examples where the observed full low-to-high window range lies within the predicted lower/upper bounds. A wider range can increase coverage, so read together with errors. |
| Brier score | Mean squared error between probability and binary outcome; lower is better. Used for probability calibration summaries, including support-hold calibration. |
| Support held / resistance reached | Per-window boolean outcomes under the definitions above, not broad model accuracy metrics. |

Post-window metrics use actual candles only after the forecast window has ended. The comparison requires the exact expected set of 5-minute timestamps. It records actual range, ending close/direction, direction correctness, range coverage, lower/upper boundary errors, movement error, support-held/resistance-reached and average absolute boundary error. The forecast itself remains unchanged.

## Legacy vs final model path

Do not conflate `src/models/train.py` and `src/targets/labeling.py` with the final system. Those files concern a 10-minute classifier and legacy model comparison. The API model training/status and forecast issuance paths use `src/forecasts/training.py` and `src/forecasts/service.py`, which build separate configured-window outcomes and use HistGradientBoosting/quantile GradientBoosting models. LightGBM may be an optional legacy experiment; it is not the connected final window predictor. No system path places an order.
