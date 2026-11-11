# AMZN Trading Analysis Engine — Independent Correctness Audit

Replay period: **2025-10-06 through 2026-11-09 ET** (21,420 completed regular-session candles).
Source: `AMZN_clean_20261110T010232.csv` SHA-256 `cca838242e41d9515ac59e6fb7e249c5bff3189a5f0e3928b22c97913138d00a`. No training, tuning, live Alpaca request, broker order, or production DB write occurred.
Production model: `legacy-lightgbm-10m-e1b5cf0` (LGBMClassifier), SHA-256 `f1f5330aafbe9022f1e97c83bf45e968d19a1dab06c020a953f2acd9d37b6aac`, 40 ordered inputs; class order `['0.0', '1.0']`. Raw model scores are not calibrated confidence.
The replay took **28,405 seconds** on this environment. Independent SHA-256 checks matched the input snapshot and model artifact to the hashes recorded in replay metadata.

## Independent correctness findings

- **Causal inputs:** EMA/SMA, rolling returns, MACD and Bollinger features are backward-looking. RSI, ATR and volume averages reset by New York session. The production 40-feature adapter uses the unchanged legacy manifest; replay batch feature values were compared against prefix-only generation at four fixed positions.
- **Pivot timing:** a 2×2 pivot is exposed only at the close of the second right-side candle. RSI divergence is attached to that confirmation candle. Replay prefix checks compared full-series and prefix indicator rows and filtered cached pivots by `confirmed_at <= current candle start` (the candle is completed before the decision).
- **Zones:** only confirmed pivots are clustered; support/resistance prices and strength are computed from data available by each replay timestamp. Zone reaction strength uses up to six candles after pivot confirmation, truncated to the replay prefix. A touch is capped at one pivot per session.
- **Breakout state:** close cross plus ATR penetration and relative volume are required for confirmation. Failed-breakout labels now require a matching upward/downward crossing in the configured 12-bar lookback; this avoids labeling a remote old zone as a recent failed break.
- **Timestamp discipline:** candle timestamps denote starts. Analysis generation time is the completed candle end. Incomplete bars are filtered; duplicate timestamps and invalid OHLC are rejected; non-session/early-close-overrun records are excluded by the existing calendar. The latest live snapshot must be within the configured 15-minute lag.
- **ML:** replay uses the frozen production LightGBM artifact and `legacy_engineering.generate_features(..., reset_daily=False)` with exact 40-column order and class mapping. Prediction for row *t* is made from that row's completed features only. No research challenger is loaded.
- **Risk/outcomes:** signal close is a reference, not a fill. Entry is the next same-session 5-minute open only when inside the planned entry zone; otherwise the setup expires without P&L. A single position is allowed per strategy. Stop takes priority over target 2 when a bar touches both. Missing same-session intervals stay pending; no overnight entry/exit is assumed.

### Corrections made during the audit

- Replaced same-candle close fill assumption with next-bar open/entry-zone evaluation.
- HOLD is now non-actionable and cannot open another stored position; opposite positions do not overlap.
- Prevented intraday open positions from being resolved using next-session prices when the session-end bar is missing.
- Made target-1 lifecycle events idempotent and prevented target 1 being credited when the same candle also hits the stop.
- Restricted failed-breakout labeling to recent initiating crossings.

### Prefix parity checks

| Index | Candle timestamp          | Legacy feature prefix | Indicator prefix | Structure prefix | Future pivots excluded |
| ----- | ------------------------- | --------------------- | ---------------- | ---------------- | ---------------------- |
| 5355  | 2026-01-14T18:15:00+00:00 | True                  | True             | True             | True                   |
| 10710 | 2026-04-24T15:00:00+00:00 | True                  | True             | True             | True                   |
| 16065 | 2026-08-04T15:45:00+00:00 | True                  | True             | True             | True                   |
| 21419 | 2026-11-09T20:55:00+00:00 | True                  | True             | True             | True                   |

## Signal distribution

Valid analyses: **17,786**; rows without required warm-up: **3,634**; completed valid candles: **21,420**; calendar-filtered rows: **36**.
Actionable BUY/SELL signals: **1,173 (6.60% of valid analyses)**. HOLD is not counted as a new trade.

| Decision              | Count | Share of valid analyses (%) |
| --------------------- | ----- | --------------------------- |
| BUY                   | 578   | 3.25                        |
| HOLD                  | 1800  | 10.12                       |
| INSUFFICIENT_HISTORY  | 3634  | —                           |
| SELL                  | 595   | 3.35                        |
| WAIT FOR CONFIRMATION | 14813 | 83.28                       |

## Hypothetical execution results

Each strategy is an independent, one-position-at-a-time ledger. Entries use one share and fixed size; return percentages are summed additively, not compounded. Drawdown is measured in percentage points on cumulative net returns. Baselines use the same entry-zone, stop/target, close/time limit and 10 bps assumed round-trip cost. Different strategy ledgers are not combined into a portfolio.

| Strategy  | Setups | Resolved trades | Gross return sum (%) | Net return sum (%) | Max DD (pp) | Gross PF | Net PF | Avg net/trade (%) | Expired | Unresolved |
| --------- | ------ | --------------- | -------------------- | ------------------ | ----------- | -------- | ------ | ----------------- | ------- | ---------- |
| combined  | 1173   | 810             | 12.5222              | -68.4778           | 70.8297     | 1.1060   | 0.5985 | -0.0845           | 363     | 0          |
| ema_trend | 2423   | 1575            | 8.6059               | -148.8941          | 150.7550    | 1.0357   | 0.5698 | -0.0945           | 847     | 1          |
| rsi_14    | 784    | 629             | 16.4666              | -46.4334           | 54.4818     | 1.1350   | 0.7132 | -0.0738           | 155     | 0          |
| ml_only   | 2240   | 1410            | 95.9591              | -45.0409           | 60.8781     | 1.5232   | 0.8318 | -0.0319           | 829     | 1          |

Combined setup hit counts: target 1 **302** (37.28% of entered setups), target 2 **158** (19.51% of resolved trades), stop loss **448** (55.31% of resolved trades), expired **363**, stop-invalidated **448**, unresolved **0**. Target 1 is a milestone only; no partial exit/P&L is booked.

### Regime breakdown

`valid_candle_count` is the full valid decision sample in each regime; `entered_or_closed_setups` is the strategy-specific setup sample. These are small-sample historical observations, not evidence of profitability.

| Strategy  | Regime          | Valid candles | Setups | Resolved | Net return (%) | Max DD (pp) | Target 1 | Target 2 | Stops |
| --------- | --------------- | ------------- | ------ | -------- | -------------- | ----------- | -------- | -------- | ----- |
| combined  | TRENDING_UP     | 2046          | 125    | 94       | -12.8558       | 13.4966     | 32       | 15       | 50    |
| combined  | TRENDING_DOWN   | 2207          | 150    | 100      | -10.0182       | 11.4530     | 33       | 18       | 60    |
| combined  | SIDEWAYS        | 3906          | 328    | 222      | -17.3431       | 17.5312     | 89       | 51       | 134   |
| combined  | HIGH_VOLATILITY | 2981          | 189    | 166      | -7.9734        | 12.0196     | 60       | 24       | 74    |
| combined  | LOW_VOLATILITY  | 3908          | 381    | 228      | -20.2873       | 21.1034     | 88       | 50       | 130   |
| combined  | TRANSITIONAL    | 2738          | 0      | 0        | 0.0000         | 0.0000      | 0        | 0        | 0     |
| ema_trend | TRENDING_UP     | 2046          | 200    | 125      | -14.0337       | 14.1203     | 45       | 26       | 67    |
| ema_trend | TRENDING_DOWN   | 2207          | 256    | 161      | -15.5747       | 17.3379     | 61       | 36       | 102   |
| ema_trend | SIDEWAYS        | 3906          | 398    | 254      | -21.5122       | 22.6229     | 106      | 58       | 160   |
| ema_trend | HIGH_VOLATILITY | 2981          | 433    | 388      | -26.5714       | 28.9255     | 138      | 60       | 191   |
| ema_trend | LOW_VOLATILITY  | 3908          | 675    | 366      | -38.1727       | 38.6866     | 126      | 70       | 230   |
| ema_trend | TRANSITIONAL    | 2738          | 461    | 281      | -33.0295       | 33.0295     | 90       | 54       | 177   |
| rsi_14    | TRENDING_UP     | 2046          | 86     | 66       | -5.7751        | 6.5574      | 25       | 10       | 39    |
| rsi_14    | TRENDING_DOWN   | 2207          | 154    | 112      | -4.1254        | 6.1874      | 43       | 25       | 64    |
| rsi_14    | SIDEWAYS        | 3906          | 52     | 40       | -7.3966        | 7.3966      | 13       | 8        | 21    |
| rsi_14    | HIGH_VOLATILITY | 2981          | 335    | 309      | -19.7824       | 27.9603     | 93       | 44       | 151   |
| rsi_14    | LOW_VOLATILITY  | 3908          | 64     | 45       | -1.2204        | 2.2446      | 19       | 11       | 19    |
| rsi_14    | TRANSITIONAL    | 2738          | 93     | 57       | -8.1335        | 8.3690      | 17       | 11       | 36    |
| ml_only   | TRENDING_UP     | 2046          | 201    | 125      | 0.0670         | 8.2728      | 61       | 36       | 60    |
| ml_only   | TRENDING_DOWN   | 2207          | 248    | 146      | 0.4735         | 5.1552      | 69       | 42       | 79    |
| ml_only   | SIDEWAYS        | 3906          | 375    | 219      | -15.5574       | 17.1333     | 103      | 54       | 125   |
| ml_only   | HIGH_VOLATILITY | 2981          | 402    | 357      | 0.9932         | 17.6782     | 131      | 54       | 159   |
| ml_only   | LOW_VOLATILITY  | 3908          | 621    | 341      | -21.7331       | 22.6162     | 166      | 90       | 181   |
| ml_only   | TRANSITIONAL    | 2738          | 393    | 222      | -9.2841        | 14.4327     | 102      | 61       | 122   |

## Historical signal examples

**Profitable BUY setup:** 2025-10-07T19:30:00+00:00 (candle start; generated 2025-10-07T19:35:00+00:00); decision=BUY, close=221.365, regime=LOW_VOLATILITY, score=0.710, raw ML UP=0.4995, support=221.23, resistance=221.46; status=TARGET_2_HIT, gross return=0.2620%. Reasons: Directional score +0.710, regime evidence and risk/reward gates support a hypothetical BUY.
**Profitable SELL setup:** 2025-10-06T19:50:00+00:00 (candle start; generated 2025-10-06T19:55:00+00:00); decision=SELL, close=220.84, regime=LOW_VOLATILITY, score=-0.372, raw ML UP=0.5498, support=220.64, resistance=220.99; status=CLOSED, gross return=0.0091%. Reasons: Directional score -0.372, regime evidence and risk/reward gates support a hypothetical SELL.
**WAIT due to conflicting evidence:** 2025-10-06T18:25:00+00:00 (candle start; generated 2025-10-06T18:30:00+00:00); decision=WAIT FOR CONFIRMATION, close=221.16, regime=TRANSITIONAL, score=0.582, raw ML UP=0.4518, support=220.59, resistance=None; status=NOT_ACTIONABLE, gross return=—%. Reasons: trend and ml evidence disagree. Market regime is transitional; directional entry is withheld.
**Failed setup (stop hit):** 2025-10-07T14:35:00+00:00 (candle start; generated 2025-10-07T14:40:00+00:00); decision=BUY, close=222.15, regime=TRENDING_UP, score=0.539, raw ML UP=0.4799, support=221.23, resistance=None; status=STOP_LOSS_HIT, gross return=-0.5617%. Reasons: Directional score +0.539, regime evidence and risk/reward gates support a hypothetical BUY.
**Setup rejected on risk/reward:** No qualifying case occurred in this replay.

## API and operational verification

Actual HTTP calls were made to an isolated Uvicorn process whose SQLite path was redirected to a temporary database. `GET latest`, `GET signals`, `GET active`, and `GET outcomes` returned 200; unknown signal detail returned 404; outcome resolution returned 200. `POST analysis/run` returned 422 because the latest local snapshot is older than the allowed current-session staleness window; this verifies stale-data rejection. No production database was used for these HTTP checks.
No live Alpaca request was made. The scheduler hook was code-inspected and exercised through outcome service unit tests, not by leaving a scheduler process running.

## Correctness status and limitations

**Verified:** timestamp completion and calendar filtering; duplicate prevention; frozen-model manifest/class order; selected prefix parity; pivot confirmation timing; single-position entry/exit assumptions; target/stop order; no P&L for expired/unresolved setups; repeat outcome idempotence; API response routing on isolated storage.
**Heuristic / not validated as predictive:** the combined score and its weights, zone strength, trend/regime thresholds, risk distances and assumed 10 bps cost. The model raw probability is uncalibrated. Stop-first handling is conservative but intrabar event order cannot be known from 5-minute OHLC.
**Results not trustworthy as investment evidence:** this is one historical period with fixed one-share additive returns, no spread/slippage model beyond an assumed cost, no partial exits, and no statistical significance analysis. Baseline choices are simple descriptive comparators, not tuned strategies. Replay does not show executed trades or real P&L.
**Data limitations:** local snapshot ends on November 9, 2026, so current live data is stale; 36 records were excluded by the market calendar. Same-session missing intervals remain pending and may leave setups unresolved. No order book, bid/ask, actual fill, commissions, market impact, or calibrated outcome probabilities are available.

## Reproduction

From the repository root: `python reports/experiments/trading_analysis_engine_audit/replay.py` (use the documented virtual environment's Python on Windows). The script writes the CSV files, isolated `replay.sqlite3`, and `replay_metadata.json`; then run `python reports/experiments/trading_analysis_engine_audit/render_audit_report.py` to render this report.

See `test_results.txt` for commands and test status.
