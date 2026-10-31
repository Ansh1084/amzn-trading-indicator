# AI-Powered AMZN Trading Indicator — Implementation Plan

> **Date:** 28 October 2026  
> **Status:** Planning only — awaiting approval before any implementation

---

## A. Executive Assessment

**Your project is in an early-but-solid data-preparation stage.** You have a working download script, a year of clean AMZN 5-minute data (19,880 raw candles), a feature-engineering pipeline, and chronological train/val/test splits. The foundation is good. However, I found **five issues ranging from serious to critical** that must be fixed before any modeling work begins:

| # | Severity | Issue |
|---|----------|-------|
| 1 | **CRITICAL** | `future_close_10m` is retained as a column in the training CSVs. If accidentally used as a feature, it is pure target leakage. |
| 2 | **SERIOUS** | The chronological split cuts mid-day (train ends 2026-07-13 11:20, val starts 2026-07-13 11:25). The same trading day appears in both train and validation — its indicators were computed jointly, and the last train sample's 10-minute target overlaps temporally with the first val sample. |
| 3 | **SERIOUS** | All technical indicators (RSI-14, MACD, Bollinger Bands) are reset at every day boundary via `groupby('trading_date')`. This means: (a) the first ~20–35 candles per day are warm-up garbage that get dropped by `dropna()`, losing ~27% of data; (b) the indicators computed this way do **not** match the rolling behavior a live system would see, since a live indicator does not reset at market open. |
| 4 | **MODERATE** | `between_time('09:30', '15:55')` is inclusive on both ends. Candles at 15:50 and 15:55 have their 10-minute targets pointing at 16:00 and 16:05 respectively. The 16:00 candle (market close) may exist in the raw IEX data but was filtered out by the RTH cutoff, so the merge returns NaN correctly. However, `15:45` maps to `15:55` which **was** included before the dropna—this works but is fragile. The last usable candle per day is confirmed as 15:45. |
| 5 | **MODERATE** | `AMZN_data.csv` (1,675 rows, ~1 month, Sep 28–Oct 26 2026) appears to be an earlier download attempt. It is fully contained within `AMZN_data_1year.csv` and is redundant. |

**Overall assessment:** The data is clean (no nulls, no duplicates, valid OHLCV), the target generation logic is conceptually correct but needs refinement, and the project needs significant architectural work before modeling. The gap between current state and a working live indicator is large but manageable with disciplined phasing.

---

## B. Requirements Extracted from the Project Brief

I successfully extracted the full text of [AI_Trading_Indicator_Project_Brief(2).pdf](file:///c:/Users/anshg/Desktop/SEM5/Projects/ML/AI_Trading_Indicator_Project_Brief(2).pdf). Below is my mapping of the grading criteria to technical requirements:

### Marking Scheme → Technical Requirements

| Category (Marks) | What the Brief Says | Technical Acceptance Criteria |
|---|---|---|
| **Live-market performance (40)** | Accuracy of confident signals; simulated profit after costs | The indicator must run live for 5 trading days producing UP/DOWN/NO CLEAR SIGNAL on each candle. Confident signals are checked against actual price movement over the stated horizon. A simulated P&L after transaction costs is computed. Must beat simple baselines (always-UP, RSI rule). |
| **Model quality & honest validation (20)** | Proper validation, no overfitting claims | Chronological train/val/test splits without leakage. Walk-forward or time-series cross-validation. Honest metrics on held-out data. Failed experiments documented. |
| **Indicator usefulness (20)** | Readable in one second; direction + confidence; no repainting | Clear UP/DOWN/NO CLEAR SIGNAL visual. Confidence percentage shown. Signals on closed candles never change. One-second readability. |
| **Report & demonstration (20)** | Reproducible results; explained choices; user guide | Code repo reproduces all results. Report covers what worked and what didn't. User guide explains when not to trust the indicator. Live demo followed by Q&A. |

### Hard Constraints (Zero-Score Violations)

1. Indicator must run live — a static backtest is not sufficient.
2. Indicator must produce signals — a silent indicator scores zero.
3. No repainting — a closed-candle signal must never change.
4. The model must be the team's own trained model — a copied public indicator scores zero.
5. All external code/models/papers must be cited and explainable.
6. Paper trading only.
7. Every result must be reproducible from submitted code.

### Confirmed Facts vs. Unknowns

| Item | Status |
|------|--------|
| Asset: AMZN | ✅ Confirmed (your choice) |
| Horizon: 10 minutes | ✅ Your choice (brief allows any) |
| Delivery: Option B (Python + web app) | ✅ Your choice (brief allows it) |
| Candle interval: 5 minutes | ✅ Your choice |
| Data source: Alpaca IEX | ✅ Your choice |
| Live test duration: 5 trading days | ✅ From brief |
| Evaluator's data feed | ❓ Unknown — brief says "live chart of the chosen asset" |
| Exact scoring formula | ❓ Unknown — brief gives categories, not formulas |
| How "simulated profit after costs" is computed | ❓ Unknown — must assume reasonable defaults |
| Whether evaluator runs your web app or their own | ❓ Unknown — brief says "the team's web app" for Option B |
| Transaction cost assumptions | ❓ Unknown |
| Latency/runtime requirements | ❓ Unknown |

---

## C. Existing Project Audit

### File-by-File Analysis

#### [test_amazon.py](file:///c:/Users/anshg/Desktop/SEM5/Projects/ML/test_amazon.py) — Data Download Script

**Purpose:** Downloads 1 year of 5-minute AMZN bars from Alpaca IEX in 30-day chunks.

**Verified correct:**
- Uses environment variables for API keys (good practice)
- Chunks downloads to avoid hitting `limit=10000`
- Deduplicates on `(timestamp, symbol)`
- Sorts chronologically

**Issues found:**
- The filename `test_amazon.py` is misleading — it's a download script, not a test suite. Recommend renaming to `download_data.py`.
- The loop increment uses `cursor = chunk_end` which creates contiguous windows (correct, no gaps).
- The `end = datetime.now(timezone.utc) - timedelta(minutes=20)` means data extends to 20 minutes before runtime — appropriate to avoid incomplete candles.

#### [AMZN_data_1year.csv](file:///c:/Users/anshg/Desktop/SEM5/Projects/ML/AMZN_data_1year.csv) — Raw Data

| Property | Value |
|----------|-------|
| Shape | 19,880 rows × 9 columns |
| Columns | symbol, timestamp, open, high, low, close, volume, trade_count, vwap |
| Date range | 2025-10-27 to 2026-10-26 (UTC timestamps) |
| Nulls | 0 |
| Duplicates | 0 |
| OHLCV integrity | ✅ No violations (high≥low, open/close within range, no zero volume, no negatives) |
| Price range | $197.06 – $286.26 |
| Trading days | 251 |
| Candles per typical day (RTH) | 78 (9:30–15:55, every 5 min) |

**Assessment:** Clean, usable raw data. **Retain as-is.**

#### [AMZN_data.csv](file:///c:/Users/anshg/Desktop/SEM5/Projects/ML/AMZN_data.csv) — Subset

| Property | Value |
|----------|-------|
| Shape | 1,675 rows × 9 columns |
| Date range | 2026-09-28 to 2026-10-26 |

**Assessment:** This is a ~1 month subset, likely from an earlier download. It is entirely contained within `AMZN_data_1year.csv`. **Redundant. Can be ignored or deleted.**

#### [prepare_amazon.py](file:///c:/Users/anshg/Desktop/SEM5/Projects/ML/prepare_amazon.py) — Feature Engineering & Split

**Four critical findings:**

**Finding 1: Daily-reset indicators are architecturally wrong for live inference.**

```python
# Line 20-21: RSI grouped by trading_date
avg_gain = gain.groupby(df['trading_date']).transform(lambda x: x.rolling(14).mean())
```

All indicators are computed within `groupby('trading_date')`. This means:
- RSI-14 needs 14 candles to warm up → first 14 candles per day are NaN
- MACD (EMA-26 + signal-9) needs ~35 candles → first ~35 are NaN
- Bollinger Bands (SMA-20) needs 20 candles → first 20 are NaN

After `dropna()`, only ~57 candles remain per day (from 78 possible). This is a **27% data loss** and a **train-live mismatch**: a live system would compute RSI/MACD/BB as continuous rolling values across days, not resetting at open.

**Recommendation:** Compute indicators as continuous rolling series across the full dataset, then filter to RTH candles. This matches live behavior and recovers lost data.

**Finding 2: `future_close_10m` column is retained in output files.**

```python
# Line 86: saves ALL columns including future_close_10m
df_clean.to_csv("AMZN_prepared.csv", index=False)
```

The column `future_close_10m` contains the actual future price — the answer to the prediction problem. It is saved in train/val/test CSVs. While it's not currently being used as a feature (no model code exists yet), its presence creates a dangerous leakage vector. Any model that accidentally includes all numeric columns will be perfectly overfit.

**Recommendation:** Drop `future_close_10m` and `target_10m_UP` from feature CSVs, or explicitly define feature columns vs. target columns in a configuration.

**Finding 3: Chronological split cuts mid-day.**

```
Train ends:  2026-07-13 11:20:00-04:00
Val starts:  2026-07-13 11:25:00-04:00
```

The split divides July 13th between train and val. The same day's indicators (RSI, MACD, etc.) were computed jointly because `groupby('trading_date')` operates on the full dataset before splitting. This is a subtle form of information leakage: the val set's indicators for July 13th were computed using data that also went into the train set.

Additionally, the last train sample's 10-minute target (predicting 11:30 from 11:20) temporally overlaps with the first val sample (11:25).

**Recommendation:** Split on day boundaries. Implement an embargo (gap) of at least 2 candles (10 minutes = the prediction horizon) between splits.

**Finding 4: The target definition is correct but has an edge case.**

```python
# Line 51: same-day check
same_day = df['timestamp'].dt.date == df['future_timestamp'].dt.date
df.loc[~same_day, 'future_close_10m'] = float('nan')
```

This correctly prevents cross-day targets. The merge-on-`future_timestamp` approach correctly handles the 10-minute offset using exact timestamp matching. However:
- If a 5-minute candle is missing in the raw data (unlikely for IEX but possible), the merge returns NaN and that row is dropped — correct behavior.
- The last usable candle per day is 15:45 (target at 15:55) — confirmed in data.

#### [AMZN_prepared.csv](file:///c:/Users/anshg/Desktop/SEM5/Projects/ML/AMZN_prepared.csv) — Prepared Dataset

| Property | Value |
|----------|-------|
| Shape | 14,235 rows × 24 columns |
| Nulls | 0 (after dropna) |
| Target distribution | 0: 7,232 (50.8%) / 1: 7,003 (49.2%) |
| Date range | 2025-10-27 to 2026-10-26 |
| Mixed timezones | Yes — EDT (9,462 rows) and EST (4,773 rows) |

**Assessment:** Near-perfectly balanced classes (50.8% vs 49.2%). This is expected for a 10-minute horizon on a liquid stock — short-term returns are approximately symmetric. The slight DOWN bias (50.8%) is not meaningful. The mixed timezone representation (EDT/EST) is cosmetic — the underlying UTC timestamps are correct.

#### Train / Val / Test CSVs

| Split | Rows | Date Range | Target 0/1 |
|-------|------|-----------|-----------|
| Train | 9,964 | 2025-10-27 to 2026-07-13 | 5,062 / 4,902 |
| Val | 2,135 | 2026-07-13 to 2026-09-02 | 1,087 / 1,048 |
| Test | 2,136 | 2026-09-02 to 2026-10-26 | 1,083 / 1,053 |

**Ratio:** 70% / 15% / 15% — reasonable for this dataset size.

**Issues:** Mid-day split boundaries (see Finding 3 above).

---

## D. Recommended Problem Formulation

### Target: Binary Classification with a Neutral Zone

**Current approach:** Binary UP (price rises) vs DOWN (price falls or stays flat).

**My recommendation:** Keep binary classification as the primary target, but add a **dead-zone filter** to exclude small movements from training and evaluation.

**Rationale:**
- The near-50/50 class balance tells us that many movements are tiny and essentially noise. Predicting whether a price moves +$0.01 vs -$0.01 in 10 minutes is random — no model can do this reliably.
- The assignment evaluates "accuracy of **confident** signals" and "simulated **profit** after costs." Tiny movements generate no profit after costs, so correctly predicting them adds no value.
- A dead-zone (e.g., movements smaller than 0.02%–0.05% of the current price) would:
  - Make training labels more meaningful
  - Create a natural third class (FLAT/NOISE) that maps to "NO CLEAR SIGNAL"
  - Focus the model on predicting actionable moves

**Implementation options:**

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| A. Binary + confidence threshold | Keep current binary target; use model probability to issue NO CLEAR SIGNAL when uncertain | Simple; no data loss | Model must learn to be uncertain on noise |
| B. Three-class (UP/DOWN/FLAT) | Add a FLAT class for moves within ±threshold | Explicitly models noise | Complicates metrics; threshold is arbitrary |
| C. Binary with dead-zone exclusion | Exclude small-move samples from training; at inference, use confidence for NO CLEAR SIGNAL | Cleaner signal in training data; simpler model | Loses training samples in the dead zone |

**My recommendation: Option A (binary + confidence threshold) as the primary approach, with Option C as an experiment.** Option A is simpler, retains all training data, and the confidence threshold naturally handles the NO CLEAR SIGNAL requirement. Option C is worth testing if Option A produces poor confidence calibration.

**Experiment to validate:** Compare test-set precision of high-confidence signals under Options A vs C.

### Prediction Horizon: 10 Minutes

**Assessment:** 10 minutes is a reasonable choice for this assignment.

- **Too short (seconds, 1 minute):** Requires tick-level data, dominated by microstructure noise and spread costs, nearly impossible with 5-minute candles.
- **10 minutes (2 candles ahead):** Actionable by a day trader, achievable with 5-minute data, long enough that the signal may be distinguishable from noise.
- **Too long (hours, full day):** Very few samples per day; the model has limited training data per signal; harder to accumulate enough live-test signals in 5 days.

**I agree with the 10-minute horizon. No change recommended.**

### Candle Interval: 5 Minutes

**Assessment:** Appropriate for a 10-minute horizon. Each prediction looks 2 candles ahead, which gives meaningful OHLCV resolution for the prediction window. Using 1-minute candles would provide more features per prediction but 5× more data volume with potentially more noise.

**No change recommended.**

### Data Feed: Alpaca IEX

**Key concern:** IEX (Investors Exchange) is a single-exchange feed. It may differ from consolidated feeds (SIP) that platforms like TradingView use. Price differences are typically small for liquid stocks like AMZN (pennies), but this is a potential source of train-live mismatch.

**Recommendation:** Keep Alpaca IEX for now. It provides free historical + live data, which is critical for the project. Document the feed choice in the report and note that the evaluator's data source may produce slightly different OHLCV values. If feasible, validate against Yahoo Finance or another source for a sample period.

---

## E. Recommended Architecture

### Offline Training Pipeline

```
┌─────────────────┐
│ Alpaca API       │ → download_data.py → AMZN_data_1year.csv (raw)
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ prepare_data.py  │ → RTH filter → Continuous indicators → Target generation
│                  │ → Day-boundary split with embargo → Train/Val/Test CSVs
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ train_model.py   │ → Baseline → Model experiments → Best model selection
│                  │ → Confidence threshold tuning on Val set
│                  │ → Final evaluation on Test set
│                  │ → Save: model.pkl + config.json + scaler.pkl
└─────────────────┘
```

### Live Inference + Chart Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                        Python Backend (FastAPI)                  │
│                                                                  │
│  ┌───────────────┐   ┌────────────────┐   ┌──────────────────┐  │
│  │ Data Service   │   │ Feature Engine  │   │ Model Inference  │  │
│  │ (Alpaca WS/    │──▶│ (same pipeline  │──▶│ (loaded model +  │  │
│  │  REST polling)  │   │  as training)   │   │  confidence)     │  │
│  └───────────────┘   └────────────────┘   └──────────────────┘  │
│         │                                          │             │
│         │              ┌───────────────┐           │             │
│         └─────────────▶│ Prediction Log │◀──────────┘             │
│                        │ (SQLite/CSV)   │                        │
│                        └───────────────┘                        │
└──────────────────────────────────────────────────────────────────┘
                              │
                    WebSocket / SSE
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Frontend (HTML + JS)                          │
│                                                                  │
│  ┌───────────────────────────────────────────────────────┐      │
│  │  TradingView Lightweight Charts                        │      │
│  │  + Signal overlay (UP ▲ / DOWN ▼ / — NO SIGNAL)       │      │
│  │  + Confidence bar/label                                │      │
│  │  + Color coding (green/red/gray)                       │      │
│  └───────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────┘
```

### Technology Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Backend framework | **FastAPI** | Async support for WebSockets; lightweight; excellent for API endpoints |
| Live data | **Alpaca** (REST polling every 5 min + WebSocket for real-time) | Same source as training data; free tier sufficient |
| Charting | **TradingView Lightweight Charts** (JS) | Recommended by brief; looks professional; free |
| Model format | **joblib/pickle** (sklearn/LightGBM) | Standard for classical ML models |
| Prediction log | **SQLite** or **CSV with append** | Simple; sufficient for 5-day test |
| Communication | **WebSocket** (FastAPI → browser) | Real-time candle + signal push |
| Configuration | **JSON config file** | Feature list, thresholds, model path, API keys |

### What We Do NOT Need

- ❌ A database server (PostgreSQL, MongoDB)
- ❌ Docker/Kubernetes
- ❌ A separate frontend framework (React, Vue)
- ❌ A message queue (Redis, RabbitMQ)
- ❌ Cloud deployment (can run locally for demo)
- ❌ User authentication

---

## F. Detailed Phase-Wise Implementation Roadmap

### Phase 1: Requirements Audit & Acceptance Criteria

**Objective:** Translate the project brief into measurable pass/fail criteria.

**Why it matters:** Without explicit acceptance criteria, we risk building something that doesn't match the grading rubric.

**Existing files to inspect:** [AI_Trading_Indicator_Project_Brief(2).pdf](file:///c:/Users/anshg/Desktop/SEM5/Projects/ML/AI_Trading_Indicator_Project_Brief(2).pdf)

**Tasks:**
1. Map each of the four marking categories (40/20/20/20) to specific deliverables.
2. Define measurable acceptance criteria for each category.
3. List all unknowns that need instructor clarification.
4. Establish a minimum viable indicator (MVI) that would score non-zero in every category.

**Deliverables:**
- A requirements checklist with pass/fail criteria.
- A list of questions for the instructor.

**Acceptance criteria:** Every marking category has at least one measurable acceptance criterion. All team members agree on the requirements.

**Common mistakes:** Treating this as bureaucracy and skipping it. Building a model without understanding how it will be scored.

**Dependencies:** None.

**Conditions to proceed:** Requirements document reviewed and agreed upon.

---

### Phase 2: Existing Code & Dataset Audit

**Objective:** Verify every file's correctness and identify what must be fixed.

**Why it matters:** Building on a flawed foundation wastes all subsequent effort.

**Existing files to inspect:** All 9 files.

**Tasks:**
1. ✅ (Done in this plan) Verify raw data integrity — no nulls, valid OHLCV, correct date range.
2. ✅ (Done) Identify `AMZN_data.csv` as redundant.
3. Verify that the download script can be re-run to reproduce `AMZN_data_1year.csv`.
4. Document the exact Alpaca API parameters used.
5. Check for candle gaps within trading days (spot-check 10 random days).
6. Verify candle count per day (expect 78 for full days, fewer for early-close days like Nov 28, Dec 24).

**Technical decisions:**
- `AMZN_data.csv` is redundant → ignore it going forward.
- `test_amazon.py` should be renamed to `download_data.py` for clarity.

**Deliverables:**
- Data audit log documenting verified properties.

**Acceptance criteria:** All raw data properties verified. All anomalies explained (e.g., 21 candles on half-days).

**Common mistakes:** Assuming the data is correct because the download completed without errors.

**Dependencies:** None.

**Conditions to proceed:** No unexplained data anomalies.

---

### Phase 3: Data Quality & Feature Pipeline Rebuild

**Objective:** Fix the four issues identified in the audit and rebuild the preparation pipeline.

**Why it matters:** The daily-reset indicators, mid-day split, and leakage-prone columns make the current pipeline unsuitable for modeling.

**Existing files to use:** [AMZN_data_1year.csv](file:///c:/Users/anshg/Desktop/SEM5/Projects/ML/AMZN_data_1year.csv), [prepare_amazon.py](file:///c:/Users/anshg/Desktop/SEM5/Projects/ML/prepare_amazon.py)

**Tasks:**
1. **Fix indicator computation:** Compute RSI, MACD, Bollinger Bands as continuous rolling values across the full time series (not grouped by day). Only then filter to RTH hours.
2. **Warm-up period:** Discard the first N rows globally (e.g., first 100 candles = ~2 trading days) where indicators haven't stabilized.
3. **Fix the split:** Split on day boundaries with an embargo gap of ≥2 candles (10 minutes) between each split. Specifically:
   - Train: day 1 through day T₁
   - (2-candle embargo)
   - Val: day T₁+1 through day T₂
   - (2-candle embargo)
   - Test: day T₂+1 through last day
4. **Drop leakage columns:** Remove `future_close_10m` from feature CSVs. Keep `target_10m_UP` as the label. Define an explicit `FEATURE_COLUMNS` list.
5. **Handle mixed timezones:** Convert all timestamps to UTC internally, display in ET only for human readability.
6. **Verify half-days:** Check that Nov 28 (Black Friday) and Dec 24 (Christmas Eve) have correct early-close behavior. The market closes at 13:00 ET on these days, so candles after 13:00 should not exist.

**Experiments:**
- Compare the per-day RSI values vs. continuous RSI values on a sample day. Document the difference.

**Deliverables:**
- New `prepare_data.py` (replacing `prepare_amazon.py`).
- New `AMZN_prepared_v2.csv`, `AMZN_train_v2.csv`, `AMZN_val_v2.csv`, `AMZN_test_v2.csv`.
- A feature column manifest (`features.json`).

**Acceptance criteria:**
- No NaN values in output.
- No `future_close_10m` in feature columns.
- Splits occur on day boundaries.
- No temporal overlap between splits.
- Indicator values match what a live rolling computation would produce.
- Prepared dataset has more rows than current 14,235 (because continuous indicators recover the per-day warm-up loss).

**Common mistakes:**
- Computing indicators on post-split data instead of pre-split (creates different indicator values at split boundaries).
- Forgetting to exclude the target column from the feature list.
- Not verifying that the continuous RSI matches the pandas_ta or TA-Lib reference implementation.

**Dependencies:** Phase 2 complete.

**Conditions to proceed:** All acceptance criteria pass. Feature pipeline produces identical results when re-run.

---

### Phase 4: Target Definition Refinement

**Objective:** Finalize the target variable and document its exact semantics.

**Why it matters:** The target definition directly determines what the model learns and how it will be evaluated live.

**Tasks:**
1. **Confirm the definition:**
   - `target_10m_UP = 1` if `close[t+2] > close[t]` (where t+2 is 2 candles = 10 minutes later)
   - `target_10m_UP = 0` if `close[t+2] ≤ close[t]`
2. **Evaluate dead-zone filtering:**
   - Compute the distribution of 10-minute returns.
   - Determine what fraction of returns are within ±0.02%, ±0.05%, ±0.10%.
   - Decide whether to filter out near-zero returns from training.
3. **Document edge cases:**
   - Last usable candle per day: 15:45 (target at 15:55) ✅
   - Cross-day targets: NaN (dropped) ✅
   - Missing candles: merge returns NaN (dropped) ✅

**Experiments:**
- Plot histogram of 10-minute returns.
- Compare model performance with and without dead-zone filtering (deferred to Phase 9).

**Deliverables:**
- Target definition document (part of the report).
- Return distribution analysis.

**Acceptance criteria:** Target definition is unambiguous and matches live evaluation semantics.

**Common mistakes:** Changing the target mid-project without re-running all experiments.

**Dependencies:** Phase 3 complete.

**Conditions to proceed:** Target definition documented and frozen for initial experiments.

---

### Phase 5: Exploratory Data Analysis (EDA)

**Objective:** Understand the data deeply before modeling.

**Why it matters:** EDA reveals patterns, anomalies, and constraints that inform feature engineering and model selection.

**Tasks:**
1. **Price analysis:**
   - Plot AMZN price over the full year.
   - Identify major trends, regime changes, earnings dates, volatile periods.
2. **Return distribution:**
   - 5-minute return distribution (mean, std, skewness, kurtosis).
   - 10-minute return distribution.
   - Check for fat tails and non-normality.
3. **Feature distributions:**
   - Histogram of each feature (RSI, MACD, BB position, returns, volatility).
   - Check for outliers, infinite values, extreme skew.
4. **Feature-target relationships:**
   - Correlation matrix.
   - Feature importance preview (mutual information or simple univariate tests).
5. **Temporal patterns:**
   - Average return by time-of-day.
   - Volatility by time-of-day.
   - Day-of-week effects.
6. **Class balance over time:**
   - Rolling 20-day UP/DOWN ratio to check for non-stationarity.

**Deliverables:**
- EDA notebook or script with all visualizations.
- Summary of key findings.

**Acceptance criteria:** No unexplained anomalies. Clear understanding of feature distributions and temporal patterns.

**Common mistakes:**
- Skipping EDA and jumping to modeling.
- Looking at test-set distributions (only analyze train and val during this phase).

**Dependencies:** Phase 3 (clean data) complete.

**Conditions to proceed:** EDA complete with documented findings.

---

### Phase 6: Leakage-Safe Evaluation Design

**Objective:** Design the complete evaluation strategy before training any model.

**Why it matters:** If evaluation is flawed, every subsequent result is meaningless.

**Tasks:**
1. **Define the primary evaluation protocol:**

   **Recommended: Chronological hold-out with walk-forward validation.**

   ```
   ┌───────── Train ──────────┐  E  ┌── Val ──┐  E  ┌── Test ──┐
   │ Oct 2025 ──── Jul 2026   │ ░░ │Jul-Sep 26│ ░░ │Sep-Oct 26│
   └──────────────────────────┘    └─────────┘    └──────────┘
                                 E = Embargo (1 day minimum)
   ```

   - **Primary metrics** computed on the test set ONCE after all model selection is complete.
   - **Model selection** uses validation set only.
   - **Walk-forward validation** (optional, for robustness): Retrain on expanding windows and evaluate on the next N days. This simulates live deployment more faithfully but is computationally expensive.

2. **Define embargo rules:**
   - Minimum 1 full trading day between train/val and val/test.
   - No sample's 10-minute prediction window can cross a split boundary.

3. **Define the metric hierarchy:**

   | Level | Metric | Purpose |
   |-------|--------|---------|
   | Primary | **Precision of confident signals** | Directly maps to the 40-mark live test |
   | Primary | **Simulated net profit** (with costs) | Directly maps to the 40-mark live test |
   | Secondary | **Accuracy** | Overall model quality |
   | Secondary | **Balanced accuracy** | Handles any class imbalance |
   | Secondary | **ROC-AUC** | Threshold-independent discrimination |
   | Diagnostic | **Confusion matrix** | Error analysis |
   | Diagnostic | **Calibration curve** | Confidence reliability |
   | Diagnostic | **Signal coverage** | % of candles receiving UP/DOWN signal |

4. **Define the simulated trading protocol:**
   - On each confident signal, enter a paper trade.
   - Hold for exactly 10 minutes (the prediction horizon).
   - Close at the target candle's close price.
   - Deduct transaction costs: assume 0.05% round-trip (conservative for a liquid stock).
   - No position sizing — fixed $1,000 per trade, or flat $1 per signal.
   - Track: total trades, win rate, average P&L per trade, total P&L, max drawdown.

5. **Define baseline comparisons:**
   - Always predict UP (majority-ish class).
   - Always predict DOWN.
   - Random 50/50.
   - Simple RSI rule (UP if RSI < 30, DOWN if RSI > 70, else NO SIGNAL).

**Deliverables:**
- Evaluation protocol document.
- Baseline results on the validation set.

**Acceptance criteria:** Evaluation protocol prevents all forms of leakage listed in Section 5 of the request.

**Common mistakes:**
- Using the test set for model selection.
- Not accounting for transaction costs in simulated P&L.
- Comparing models on accuracy alone when the assignment grades on confident-signal accuracy.

**Dependencies:** Phase 3 (clean splits) complete.

**Conditions to proceed:** Evaluation protocol documented and approved.

---

### Phase 7: Baseline Modeling

**Objective:** Establish simple, reproducible baselines that every subsequent model must beat.

**Why it matters:** Without baselines, we cannot demonstrate that the ML model adds value.

**Tasks:**
1. **Implement baselines:**
   - Majority-class predictor (always predict the majority class in training data).
   - Random predictor (coin flip).
   - Simple RSI rule.
2. **Implement Logistic Regression** as the simplest ML baseline:
   - Standardize features (fit scaler on train only).
   - Train on train set, evaluate on val set.
   - Record all metrics from Phase 6.
3. **Establish the "bar to clear":**
   - If Logistic Regression cannot beat the majority-class baseline on precision of confident signals, the feature set needs work before trying more complex models.

**Deliverables:**
- Baseline and LogReg results on the validation set.
- A results comparison table.

**Acceptance criteria:**
- All baselines produce reproducible results.
- LogReg results are recorded with full metrics.

**Common mistakes:**
- Fitting the scaler on the full dataset instead of train-only.
- Skipping baselines and going straight to complex models.

**Dependencies:** Phase 6 complete.

**Conditions to proceed:** Baselines established. LogReg result available for comparison.

---

### Phase 8: Model Comparison Experiments

**Objective:** Systematically evaluate candidate models and identify the best one.

**Why it matters:** The model is the core of the indicator. Choosing the right model class determines live performance.

**Tasks:**
1. Train and evaluate each candidate model on the train/val split.
2. Record all metrics consistently.
3. Analyze which model performs best on the **primary metrics** (precision of confident signals, simulated profit).

**Models to test (in priority order):**

See **Section G (Model Experiment Matrix)** below for full details.

**Technical decisions:**
- Use `sklearn`-compatible APIs for all models.
- Standard preprocessing: `StandardScaler` (fit on train).
- Hyperparameter tuning: `RandomizedSearchCV` with `TimeSeriesSplit` on the training set only.

**Deliverables:**
- Model comparison table (all metrics for all models).
- Analysis of why each model succeeded or failed.

**Acceptance criteria:** At least 3 models trained and compared. Best model identified.

**Common mistakes:**
- Tuning hyperparameters on the test set.
- Comparing models only on accuracy, ignoring signal quality.
- Spending too much time on model complexity before confirming features are useful.

**Dependencies:** Phase 7 complete.

**Conditions to proceed:** Best model candidate identified with validation evidence.

---

### Phase 9: Feature Engineering Iteration

**Objective:** Test additional features to improve the best model.

**Why it matters:** Features drive model performance more than model complexity.

**Existing features to review:**

| Feature | Assessment | Recommendation |
|---------|------------|----------------|
| `RSI_14` | ✅ Classic momentum oscillator. Useful. | Keep; also test RSI_7 |
| `MACD_12_26_9`, signal, histogram | ✅ Trend-following. | Keep |
| `BBL/BBM/BBU_20_2.0` | ⚠️ Raw band values are price-level dependent. | Replace with: `bb_position = (close - BBL) / (BBU - BBL)` |
| `returns_1/2/3_period` | ✅ Direct price momentum. | Keep; add returns_5, returns_10 |
| `volatility_10` | ✅ Recent volatility. | Keep; add volatility_20, volatility_5 |
| `volume_change` | ⚠️ Noisy (single-period pct_change). | Replace with: rolling volume ratio (vol / SMA_vol_20) |

**New features to test (prioritized):**

| Priority | Feature Group | Rationale | Leakage Risk |
|----------|--------------|-----------|--------------|
| HIGH | **Candle body & range** (body = close-open, range = high-low, upper/lower wick) | Encode intra-candle price action | None |
| HIGH | **Price vs MAs** (close/SMA_20, close/SMA_50) | Trend position | None |
| HIGH | **Volume ratio** (volume / rolling_mean_volume) | Relative activity level | None |
| MEDIUM | **Lagged features** (RSI_t-1, returns_t-1, etc.) | Temporal context | None if properly shifted |
| MEDIUM | **Time-of-day** (minutes since market open, sin/cos encoded) | Intraday patterns | None |
| MEDIUM | **ATR (Average True Range)** | Volatility normalization | None |
| LOW | **Broader market context** (SPY returns if available) | Market regime | Needs additional data source |
| LOW | **VWAP distance** (close - vwap) | Institutional fair value | None (vwap is in raw data) |

**Feature engineering protocol:**
1. Add one feature group at a time.
2. Re-train the best model with the new features.
3. Compare validation metrics to the previous best.
4. Keep the feature group only if it improves the primary metric.

**Critical rule: Every feature must be computable in real-time during live inference.**

**Deliverables:**
- Feature importance ranking.
- Ablation study showing impact of each feature group.
- Final feature set documented in `features.json`.

**Acceptance criteria:** Final feature set improves validation metrics over the initial feature set. All features are computable from live OHLCV data.

**Common mistakes:**
- Adding 50 features without testing their impact.
- Using absolute price levels as features (price is non-stationary).
- Using future-dependent features (e.g., day's range when the day isn't over).

**Dependencies:** Phase 8 complete.

**Conditions to proceed:** Final feature set documented and frozen for model finalization.

---

### Phase 10: Model Tuning & Final Selection

**Objective:** Fine-tune the best model and make the final selection.

**Why it matters:** This is the last chance to improve model quality before freezing for live deployment.

**Tasks:**
1. Hyperparameter tuning using `RandomizedSearchCV` or `Optuna` with `TimeSeriesSplit` on the training data.
2. Test ensemble approaches (e.g., averaging LightGBM + LogReg) if time permits.
3. **Final test-set evaluation:** Run the selected model on the test set ONCE. Record all metrics. This is the reported result.
4. Compare final model to all baselines on the test set.

**Deliverables:**
- Final model artifact (`model.pkl`, `scaler.pkl`, `config.json`).
- Test-set results table.
- Model card documenting: architecture, hyperparameters, training data, features, performance.

**Acceptance criteria:**
- Model beats majority-class baseline on test set.
- Model beats RSI rule on test set.
- Test-set results are consistent with validation results (no large discrepancy suggesting overfitting to val set).

**Common mistakes:**
- Running the test set multiple times and picking the best run.
- Over-tuning hyperparameters.

**Dependencies:** Phase 9 complete.

**Conditions to proceed:** Model selected and frozen.

---

### Phase 11: Confidence Calibration & Signal Policy

**Objective:** Design the UP/DOWN/NO CLEAR SIGNAL logic with calibrated confidence.

**Why it matters:** The assignment evaluates "accuracy of **confident** signals." An uncalibrated model score (e.g., `predict_proba = 0.55`) is not a trustworthy probability.

**Tasks:**
1. **Analyze raw model probabilities:**
   - Plot reliability diagram (predicted probability vs actual frequency) on the validation set.
   - If poorly calibrated, apply Platt scaling or isotonic regression (fit on validation set).

2. **Define signal policy:**
   ```
   If calibrated_probability > upper_threshold:  signal = UP,   confidence = probability
   If calibrated_probability < lower_threshold:  signal = DOWN, confidence = 1 - probability
   Else:                                         signal = NO CLEAR SIGNAL
   ```

3. **Tune thresholds on the validation set:**
   - Sweep upper_threshold from 0.50 to 0.80.
   - For each threshold, compute: signal accuracy, signal coverage (% of candles with signals), simulated P&L.
   - Choose the threshold that maximizes the weighted combination of accuracy and coverage.

4. **NO CLEAR SIGNAL percentage:**
   - Target: 30–60% of candles should be NO CLEAR SIGNAL. Too few = the model isn't selective enough. Too many = the model is useless.

**Deliverables:**
- Calibration curve (before and after calibration).
- Threshold sweep results table.
- Chosen thresholds and their justification.

**Acceptance criteria:**
- Confident signals have accuracy > 55% on the validation set (better than random).
- Signal coverage is between 20–80%.
- The threshold was NOT tuned on the test set.

**Common mistakes:**
- Using `predict_proba` directly as confidence without checking calibration.
- Setting thresholds by looking at test-set performance.
- Making every candle a signal (no selectivity).

**Dependencies:** Phase 10 complete.

**Conditions to proceed:** Thresholds selected and frozen.

---

### Phase 12: Historical Simulation (Paper Backtest)

**Objective:** Simulate live trading performance on the test set.

**Why it matters:** This is the closest approximation to the 5-day live test before actually going live.

**Tasks:**
1. Run the final model on the test set in chronological order, simulating the live prediction process:
   - For each candle, compute features from available data only (no future information).
   - Generate prediction + confidence.
   - Apply signal policy.
   - Record prediction and actual outcome.
2. Compute all metrics from Phase 6.
3. Compute simulated P&L:
   - Enter at close of signal candle, exit at close of target candle (10 min later).
   - Deduct 0.05% round-trip cost.
4. Plot equity curve.
5. Analyze failure modes: when does the model fail? Specific times of day? Volatile periods? Trend vs range?

**Deliverables:**
- Full backtest results on the test set.
- Equity curve plot.
- Failure mode analysis.
- Comparison table: model vs all baselines.

**Acceptance criteria:**
- Simulated P&L is positive after costs (or at least better than baselines).
- Results are consistent with validation-set performance.
- No evidence of look-ahead bias in the simulation.

**Common mistakes:**
- Using the historical simulation to re-tune the model (this IS the test set).
- Ignoring transaction costs.
- Not checking that simulation features match training features exactly.

**Dependencies:** Phase 11 complete.

**Conditions to proceed:** Results documented. No evidence of major issues.

---

### Phase 13: Live Data & Inference Architecture

**Objective:** Build the real-time data pipeline and model inference system.

**Why it matters:** The indicator must run live for 5 days. This is 40% of the grade.

**Tasks:**
1. **Data ingestion:**
   - Set up Alpaca REST polling: every 5 minutes (or on candle close), fetch the latest complete candle.
   - Alternative: Alpaca WebSocket for real-time bar updates.
   - Validate each candle: check timestamp, OHLCV values, completeness.

2. **Feature computation:**
   - Maintain a rolling window of recent candles (at least 50 for indicator warm-up).
   - Compute features using the **exact same code** as the training pipeline.
   - **Critical:** Extract the feature computation into a shared module (`features.py`) that is imported by both `prepare_data.py` and the live inference code.

3. **Model loading and inference:**
   - Load `model.pkl`, `scaler.pkl`, `config.json` at startup.
   - On each new candle: scale features → predict → apply signal policy → output.

4. **Prediction timing (no-repainting):**
   - Predictions are made ONLY on complete (closed) candles.
   - A candle is complete when the next candle's timestamp arrives.
   - Once a prediction is issued for a closed candle, it is stored and NEVER updated.

5. **Prediction logging:**
   - Log every prediction with: timestamp, candle OHLCV, features used, raw probability, calibrated probability, signal, confidence, model version.
   - Also log the actual outcome when it becomes available (10 minutes later).

**Deliverables:**
- `features.py` — shared feature computation module.
- `inference.py` — model loading and prediction.
- `data_service.py` — live data ingestion.
- Prediction log schema.

**Acceptance criteria:**
- Features computed from live data match features computed from historical data for the same candle.
- No prediction is issued on an incomplete candle.
- Prediction log records every prediction.

**Common mistakes:**
- Using different feature computation code for training and inference.
- Issuing predictions on incomplete candles (causes repainting).
- Not handling market closures, weekends, and holidays.

**Dependencies:** Phase 12 complete.

**Conditions to proceed:** Inference pipeline tested on historical data and produces identical results to the backtest.

---

### Phase 14: Web Application & Live Chart

**Objective:** Build the user-facing chart with signal overlay.

**Why it matters:** This is 20% of the grade (indicator usefulness to a trader).

**Tasks:**
1. **Backend (FastAPI):**
   - `GET /api/candles` — returns historical candles for chart initialization.
   - `WebSocket /ws/live` — pushes new candles and predictions in real-time.
   - `GET /api/predictions` — returns prediction history for verification.

2. **Frontend (HTML + JS):**
   - TradingView Lightweight Charts for the candlestick chart.
   - Signal markers on candles:
     - ▲ Green triangle above candle for UP (with confidence %).
     - ▼ Red triangle below candle for DOWN (with confidence %).
     - Gray dash or no marker for NO CLEAR SIGNAL.
   - Signal panel showing: current signal, confidence, time of next prediction.
   - Performance summary: running accuracy, P&L tracker.

3. **Design requirements (from the brief):**
   - "A clear visual that a trader can read in one second" — Use color + shape + text.
   - "Stable signals: a signal on a closed candle must never change afterwards" — Predictions are stored once and displayed from storage.

**Deliverables:**
- `app.py` — FastAPI application.
- `static/index.html` — chart page.
- `static/app.js` — chart logic and WebSocket handling.
- `static/style.css` — styling.

**Acceptance criteria:**
- Chart loads with historical candles.
- New candles appear in real-time.
- Signals appear on closed candles only.
- Signals never change after appearing.
- A trader can read the signal in one second.

**Common mistakes:**
- Displaying signals on the current (incomplete) candle.
- Using polling instead of WebSocket (causes delay).
- Making the UI too complex.

**Dependencies:** Phase 13 complete.

**Conditions to proceed:** Chart works with simulated data.

---

### Phase 15: Integration & Reliability Testing

**Objective:** Test the complete system end-to-end under realistic conditions.

**Why it matters:** The live test is 5 continuous trading days. Any failure mode that wasn't tested will appear.

**Tasks:**
1. **No-repainting verification:**
   - Run the system for a simulated trading day.
   - Record all predictions.
   - Restart the system.
   - Load predictions from the log.
   - Verify that displayed signals match the logged predictions.

2. **Failure mode testing:**
   - Simulate a missing candle (skip one API response) → system should wait, not crash.
   - Simulate an API error → system should retry, not display a stale prediction.
   - Simulate market close → system should stop predicting and display "Market Closed."
   - Simulate server restart → system should reload model and prediction log.
   - Simulate a candle with 0 volume → system should handle gracefully.

3. **Timezone testing:**
   - Verify that predictions use Eastern Time for market hours.
   - Verify that the 9:30 AM first candle is handled correctly.
   - Verify that the 3:55 PM last candle doesn't generate a prediction (no valid target).

4. **Feature consistency verification:**
   - For 10 random candles, compare features computed by the live pipeline vs the training pipeline. They must be identical (within floating-point tolerance).

**Deliverables:**
- Integration test results.
- Bug fix log.
- Failure mode test report.

**Acceptance criteria:** All failure modes handled gracefully. No-repainting verified. Feature consistency confirmed.

**Common mistakes:**
- Not testing failure modes until the live test.
- Not verifying feature consistency between training and inference.

**Dependencies:** Phase 14 complete.

**Conditions to proceed:** All tests pass.

---

### Phase 16: Model Freeze & Pre-Live Checklist

**Objective:** Freeze the model and verify readiness for the 5-day live test.

**Why it matters:** The brief says "No changes are allowed during the test." This is the last chance.

**Tasks:**
1. **Freeze the model:**
   - Lock `model.pkl`, `scaler.pkl`, `config.json`, confidence thresholds.
   - Record SHA-256 hashes of all model artifacts.
   - Do not retrain or modify after this point.

2. **Pre-live checklist:**
   - [ ] Model artifact files exist and load correctly.
   - [ ] Feature pipeline produces correct output for recent candles.
   - [ ] Alpaca API credentials are valid and not expired.
   - [ ] The web application starts without errors.
   - [ ] The chart displays historical candles correctly.
   - [ ] A manual test prediction produces a reasonable output.
   - [ ] The prediction log is writable.
   - [ ] The system handles market open and close correctly.
   - [ ] No-repainting is verified.
   - [ ] The system can be restarted and resume correctly.

3. **Dry run:**
   - Run the complete system during one trading day (before the official test period).
   - Verify that predictions are generated, displayed, and logged correctly.

**Deliverables:**
- Frozen model artifacts with hashes.
- Pre-live checklist (all items checked).
- Dry-run results.

**Acceptance criteria:** All checklist items pass. Dry run completes without issues.

**Common mistakes:**
- Making last-minute changes that break something.
- Not testing with the live API before the official test.

**Dependencies:** Phase 15 complete.

**Conditions to proceed:** All checks pass. Model is frozen.

---

### Phase 17: Report, User Guide & Demonstration

**Objective:** Write the report, user guide, and prepare the demonstration.

**Why it matters:** 20% of the grade.

**Tasks:**
1. **Report (structured document):**
   - Problem definition and requirements.
   - Data collection and quality.
   - Feature engineering decisions (what worked, what didn't).
   - Model comparison results.
   - Evaluation methodology and metrics.
   - Confidence calibration approach.
   - Architecture description.
   - Live test results (added after the test).
   - Limitations and future work.
   - Failed approaches (required by the brief).

2. **User guide (short document):**
   - How to read the indicator.
   - What UP/DOWN/NO CLEAR SIGNAL means.
   - What the confidence percentage means.
   - When NOT to trust the indicator (e.g., during earnings, extreme market events, first candle of the day).
   - How to start and stop the application.

3. **Demonstration preparation:**
   - Script for the live demo.
   - Answers to anticipated questions:
     - "How does the model work?"
     - "Why this model and not X?"
     - "What is the accuracy?"
     - "Is this profitable?"
     - "What is the Lorentzian distance and why did/didn't you use it?"

**Deliverables:**
- Report (PDF/markdown).
- User guide (1–2 pages).
- Demo script.

**Acceptance criteria:**
- Every result in the report is reproducible from the submitted code.
- All external sources are cited.
- The report documents at least one failed approach.

**Common mistakes:**
- Claiming the model is profitable without sufficient evidence.
- Not documenting failed experiments.
- Not preparing for Q&A.

**Dependencies:** Phase 16 complete. Phase 12 results available for the report.

**Conditions to proceed:** N/A (final phase).

---

## G. Model Experiment Matrix

| # | Model | Hypothesis | Preprocessing | Evaluation | Success Criteria | Main Risks | Priority | Decision Rule |
|---|-------|-----------|--------------|------------|-----------------|------------|----------|---------------|
| 0 | **Majority-class baseline** | The simplest predictor sets the floor. | None | Val accuracy, precision | N/A (reference) | N/A | **Essential** | Always keep as reference |
| 1 | **Logistic Regression** | Linear relationships in feature space may capture some directional signal. | StandardScaler (fit on train) | All metrics | Beats majority baseline | Underfitting; linear assumption too strong | **Essential** | Keep if AUC > 0.52 |
| 2 | **Random Forest** | Non-linear feature interactions may capture patterns missed by LogReg. | No scaling needed | All metrics | Beats LogReg on primary metrics | Overfitting on noisy data; slow inference | **Essential** | Keep if improves over LogReg without signs of overfitting |
| 3 | **LightGBM** | Gradient boosting typically excels on tabular data with moderate feature counts. | No scaling needed; handles missing values | All metrics | Best overall model | Overfitting; hyperparameter sensitivity | **Essential** | Keep if it's the best-performing model |
| 4 | **Standard KNN (Euclidean)** | Nearest-neighbor approach captures local patterns in feature space. | StandardScaler (critical) | All metrics | Comparable to or better than tree models | Slow inference (all training data needed); curse of dimensionality; sensitive to scaling | **Recommended** | Keep if competitive; compare with Lorentzian KNN |
| 5 | **Lorentzian KNN** | Lorentzian distance `d = Σ log(1+|xᵢ-yᵢ|)` is less sensitive to outlier features than Euclidean, potentially better for financial data with fat-tailed distributions. | StandardScaler (still recommended; the log dampens but doesn't eliminate scale effects) | All metrics; compare feature-by-feature distance contributions | Better than Euclidean KNN; comparable to LightGBM | Computation cost; unclear advantage over well-tuned boosting; requires custom implementation | **Recommended** | Keep only if it beats both standard KNN and LightGBM on primary metrics |
| 6 | **Simple RSI rule** | A hand-coded trading rule provides a non-ML baseline that the model must beat. | None | Signal accuracy, P&L | N/A (reference) | N/A | **Essential** | Always keep as reference |

### Lorentzian Classification — Detailed Analysis

The Lorentzian distance `d(x, y) = Σ log(1 + |xᵢ - yᵢ|)` has specific properties relevant to financial time series:

**Advantages over Euclidean distance:**
- **Outlier dampening:** The `log(1 + |·|)` transformation compresses large differences. In financial data, a 5-standard-deviation RSI move is not 5× more informative than a 1σ move — the log accounts for this.
- **Natural scaling:** While not eliminating the need for standardization, the log reduces the impact of feature scale differences.
- **Robustness to fat tails:** Financial returns have heavier tails than Gaussian; Lorentzian distance doesn't amplify tail events as much as Euclidean.

**Disadvantages:**
- **No scikit-learn implementation:** Must be coded as a custom distance metric, which complicates `KNeighborsClassifier` usage (possible via `metric='precomputed'` or a custom callable, but slower).
- **Unclear improvement:** On well-scaled features, the difference between Euclidean and Lorentzian KNN may be small. LightGBM's tree-based splits may outperform both.
- **Computational cost:** KNN with a custom distance metric is O(n·d) per query, where n = training set size. With ~10,000 training samples and ~20 features, this is manageable but slower than tree-based inference.
- **Harder to tune:** k (number of neighbors), distance weighting, feature selection all interact.

**Recommendation:** Include Lorentzian KNN as **Experiment #5** — a recommended but not essential experiment. Test it after LightGBM. If it doesn't beat LightGBM, document the comparison and move on. It's worth testing because it's a distinctive approach that shows independent thinking (good for the report), but don't anchor the project on it.

**Implementation approach for Lorentzian KNN:**
1. Implement `lorentzian_distance(x, y)` as a Python function.
2. Use `sklearn.neighbors.KNeighborsClassifier(metric=lorentzian_distance)` (note: this disables ball-tree optimization).
3. Test k = {3, 5, 10, 20, 50}.
4. Test with and without distance weighting (`weights='distance'`).
5. Test with and without feature scaling.
6. For confidence: use the vote proportion among the k neighbors.

---

## H. Evaluation & Risk Management Strategy

### Data Leakage Prevention Checklist

| Risk | Status in Current Code | Mitigation |
|------|----------------------|------------|
| `future_close_10m` used as feature | ⚠️ Column exists in train/val/test CSVs | Drop from feature CSVs; define explicit feature list |
| `target_10m_UP` used as feature | ⚠️ Column exists in same CSVs | Explicit feature list excludes target |
| Random train/test split | ✅ Chronological split used | Keep; fix mid-day boundary |
| Scaler fit on full dataset | ❓ No model code exists yet | Fit scaler on train set only |
| Indicators computed on full dataset | ⚠️ Indicators computed before split, but this is actually correct for continuous indicators | Document: indicators are computed on the full pre-split data, which is equivalent to what would be available at each point in time |
| Overlapping prediction horizons at split boundaries | ⚠️ No embargo | Add 1-day embargo between splits |
| Test set used for model selection | ❓ No model code exists yet | Strict protocol: test set evaluated ONCE |
| Information from incomplete candles | ❓ No live code exists yet | Live pipeline: predict only on closed candles |
| Training/inference feature mismatch | ❓ No inference code exists yet | Share `features.py` between training and inference |

### No-Repainting Strategy

1. **Definition:** A prediction for candle at time T uses only data available at time T (i.e., candle T is closed, and all candles ≤ T are available).
2. **Implementation:**
   - The live system waits for the candle-close event (when the next candle's first trade arrives, or when the 5-minute boundary passes).
   - Only then does it compute features and generate a prediction.
   - The prediction is logged to SQLite/CSV with an immutable timestamp.
   - The frontend reads from the prediction log, never recomputing.
3. **Verification:**
   - After a trading day, compare logged predictions against the chart's displayed signals.
   - They must be identical.
   - Restart the application mid-day and verify that old signals are restored from the log.

### Confidence Calibration

1. **Step 1:** Train the model on the training set.
2. **Step 2:** Get `predict_proba` on the validation set.
3. **Step 3:** Plot reliability diagram: bin predictions into 10 bins, compare mean predicted probability to actual positive rate.
4. **Step 4:** If poorly calibrated, fit `CalibratedClassifierCV` with `method='isotonic'` on the validation set.
5. **Step 5:** Apply calibrated probabilities to the signal policy.
6. **Step 6:** Verify on the test set (do NOT re-calibrate on the test set).

---

## I. Alternative Solutions Considered

### Alternative 1: Option A (TradingView Pine Script)

**Pros:**
- No server infrastructure needed during the 5-day test.
- Evaluator can simply open TradingView — no dependency on your machine running.
- Professional-looking chart by default.

**Cons:**
- Pine Script has severe limitations: no matrix operations, limited arrays, no external libraries.
- Complex models (LightGBM, KNN) cannot be directly ported; must be approximated.
- Feature engineering must be reimplemented in Pine Script.
- Debugging is much harder.

**Verdict:** Option B (Python web app) is the correct choice for your project. It allows full model flexibility and is more aligned with your ML skills. The risk of "my server needs to be running during the test" is manageable.

### Alternative 2: Daily horizon instead of 10-minute

**Pros:** More time for the signal to materialize; less noise; fewer predictions per day; easier to beat transaction costs.

**Cons:** Only ~50 signals in the 5-day test (one per day, if predicting close-to-close). Very hard to demonstrate statistical significance. The brief likely expects multiple signals per day.

**Verdict:** 10-minute horizon is better for the assignment. It produces ~40 signals per day × 5 days = ~200 total signals, giving a meaningful sample for evaluation.

### Alternative 3: Regression (predict return magnitude) instead of classification

**Pros:** Richer output; can set dynamic thresholds; directly estimates P&L.

**Cons:** Much harder to get good MSE on noisy financial returns; the assignment asks for UP/DOWN/NO CLEAR SIGNAL which is inherently a classification interface.

**Verdict:** Classification is the right formulation. If desired, a regression model's output can be converted to classification (UP if predicted return > threshold, DOWN if < -threshold, else NO SIGNAL), but this adds complexity without clear benefit.

### Alternative 4: LSTM / Transformer

**Pros:** Can model temporal sequences; popular in academic financial ML.

**Cons:** Need much more data; harder to train and debug; harder to explain; slower inference; high risk of overfitting on 14K samples.

**Verdict:** Not recommended for this project. With 14K samples and a student project timeline, classical ML (LightGBM, Logistic Regression, KNN) is more likely to produce a reliable result.

---

## J. Unresolved Requirements

These questions should be clarified with the instructor before final integration:

| # | Question | Impact if Unknown | Provisional Assumption |
|---|----------|-------------------|----------------------|
| 1 | Will the evaluator run our web app, or do they need to view it remotely? | Determines whether we need to deploy to a server or just run locally | **Assume local demo on our machine** — if remote access is needed, we can add ngrok or deploy to a free cloud service |
| 2 | What data feed will the evaluator use to verify predictions? | If they use SIP data and we trained on IEX, OHLCV values may differ slightly | **Assume the evaluator uses our web app's chart** (which uses Alpaca IEX data). Document the data source. |
| 3 | How is "simulated profit after costs" calculated? What are the assumed transaction costs? | Directly affects the P&L metric | **Assume 0.05% round-trip** (conservative). Document the assumption. |
| 4 | How is a NO CLEAR SIGNAL scored? As 0 profit? As ignored? | Affects whether abstaining is penalized | **Assume NO CLEAR SIGNAL = no trade = 0 P&L** (no penalty, no reward) |
| 5 | Does the web app need to run continuously for 5 days, or can it be started each morning? | Affects reliability requirements | **Assume started each morning** — but log predictions persistently so history survives restarts |
| 6 | What exact price defines the start of the prediction? Close of the signal candle? Open of the next? | 5-minute OHLCV may differ from tick-level entry price | **Assume close of the signal candle** (matches training target definition) |
| 7 | Must the system run on a specific OS or Python version? | Affects reproducibility requirements | **Assume Python 3.10+ on any OS** |
| 8 | Is there a maximum latency for generating a prediction after candle close? | Affects architecture (batch vs real-time) | **Assume < 5 seconds** (feature computation + model inference) |

---

## K. Immediate Next Action

### Execute Phase 1 + 2 + 3 together (they are tightly coupled)

**What to do first:**

1. **Ask the instructor** the questions from Section J (at least #1, #2, #4).
2. **Rebuild the feature pipeline** (`prepare_data.py`):
   - Switch from daily-reset indicators to continuous rolling indicators.
   - Split on day boundaries with a 1-day embargo.
   - Drop `future_close_10m` from feature columns.
   - Define an explicit `FEATURE_COLUMNS` list.
3. **Verify the rebuild** by checking:
   - New row count > 14,235 (because continuous indicators recover warm-up loss).
   - No NaN values.
   - Split boundaries fall on day transitions.
   - Indicators match a reference implementation (e.g., `pandas_ta`) for spot-checked candles.

**Evidence needed before proceeding to Phase 4+:**
- Clean, verified dataset with continuous indicators.
- Chronological splits on day boundaries with embargo.
- Explicit feature list documented.

---

## L. Final Recommendation

**Build a LightGBM-based AMZN 10-minute direction classifier with calibrated confidence, served via a FastAPI + TradingView Lightweight Charts web application.**

This recommendation is based on:

1. **LightGBM as the primary model:** Consistently the best-performing classical ML model on tabular data. Fast training, fast inference, handles mixed feature types well. If it fails to beat baselines, the problem is likely in the features or data, not the model.

2. **Lorentzian KNN as a secondary experiment:** Worth testing for the report and to demonstrate independent thinking, but unlikely to be the production model.

3. **Logistic Regression as the sanity check:** If LogReg can't beat random, more complex models won't help — the features need work.

4. **Option B architecture (Python + web):** Correct for your skills and the project's flexibility requirements.

5. **Priority order:**
   - Fix the data pipeline first (this is the foundation).
   - Get baselines running (this validates the evaluation framework).
   - Train LightGBM (this is the most likely production model).
   - Add features iteratively (this improves the model).
   - Build the live indicator (this is 60% of the grade: 40% live performance + 20% usefulness).
   - Polish and document (this is the remaining 20%).

6. **Risk mitigation:** The biggest risks are (a) the model not beating baselines, and (b) the live system failing during the test. Mitigate (a) by investing in feature engineering and honest evaluation. Mitigate (b) by building the live system early and testing it for a full day before the official test.

**Do not over-engineer. A simple, well-validated, reliable system beats a complex, fragile one. The assignment rewards correctness and honesty, not complexity.**

---

> **This plan is complete. No code has been written, no files modified. Awaiting your approval before implementing any phase.**
