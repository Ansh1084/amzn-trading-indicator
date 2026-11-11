# Follow-up audit: AMZN ML research v1

**Scope:** read-only review of the v1 experiments, plus deterministic re-evaluation and audit outputs. No production model, source feature pipeline, backend, frontend, scheduler, or API was changed. All new files are under this research output directory and the audit runner.

## 1. Validation metric reconciliation

The metrics are reproducible and the mismatch is due to comparing different experiment configurations. It is not a metric calculation/reporting inconsistency. All saved fold metrics for the previously reported configurations were regenerated from the fixed local snapshot; each saved metric matched within 1e-10. Row-level validation predictions are now retained in audit_reconstructed_validation_predictions.csv.

| Configuration | Feature set | Training window | Weighting | Mean accuracy | Mean macro-F1 | Mean balanced accuracy | Mean ROC-AUC | Mean MCC |
|---|---:|---|---|---:|---:|---:|---:|---:|
| LightGBM | current 36 | Expanding; 163–248 sessions by fold | Equal | 49.60% | 49.44% | 49.59% | 49.24% | -0.0081 |
| LightGBM | expanded 86 | Expanding; 163–248 sessions by fold | Equal | 49.25% | 49.15% | 49.24% | 49.38% | -0.0153 |
| LightGBM | current 36 | Last 120 eligible sessions | Equal | 50.63% | 50.49% | 50.68% | 50.91% | 0.0135 |
| LightGBM | expanded 86 | Last 120 eligible sessions | Equal | **50.71%** | **50.61%** | **50.77%** | **51.39%** | 0.0154 |
| Logistic Regression | current 36 | Expanding; 163–248 sessions by fold | Equal | 50.64% | 50.61% | 50.69% | 50.94% | 0.0138 |
| LightGBM | recovered legacy 40 | Expanding; 163–248 sessions by fold | Equal | 49.75% | 49.27% | 49.84% | 49.41% | -0.0032 |

Validation months are June, July, August, September, and October 2026. Each expanding fold trains from October 6, 2025 through the prior month end. The 120-session folds instead start December 5, 2025; January 7, 2026; February 9; March 11; and April 10, respectively, and end at the same prior-month ends. All use the same validation dates, label rule, seed, LightGBM parameters, and train-only median imputation. The 120-session row uses 86 features; the 49.60% row uses 36. Both use equal sample weights.

Thus, the 50.71% selected result is not the 49.60% comparison repeated under the same conditions. The feature-set change alone, with expanding training, reduces macro-F1 by about 0.29 percentage points. Restricting the 86-feature model to 120 sessions then raises macro-F1 by about 1.46 points versus 86-feature expanding training. The rolling window is the main source of the selected result. On the same 120-session window, current36 and expanded86 are close: macro-F1 differs by only 0.11 points and accuracy by 0.07 points.

The selection rule ranked validation mean macro-F1. Expanded86/rolling120 scored 0.506069; LogisticRegression/current36/expanding scored 0.506052. That difference is only 0.000017 macro-F1. Selecting the first from several feature/model/window candidates is subject to selection noise; it does not demonstrate meaningful model superiority.

**Correction to the prior report wording:** its 45.27% final-test “majority baseline” is the accuracy of predicting the **training-window majority class (UP)** on the test data. It is not the test set's observed majority-class accuracy. Test labels contain 249 DOWN and 206 UP, so the hindsight test-majority rate is 54.73% (DOWN). That test-label-derived number is descriptive only and cannot be used for model selection or as a deployable baseline. The challenger’s 54.29% accuracy is 0.44 points below that hindsight rate. The earlier conclusion that the challenger exceeds “observed majority accuracy” was imprecise and should not be read as evidence of skill.

The authoritative summary is also saved as audit_authoritative_validation_comparison.csv; exact fold windows and saved-metric match flags are in audit_recomputed_fold_metrics.csv. The row-level predictions file supports direct fold inspection.

### Fold consistency and practical candidate

| Configuration | Mean validation macro-F1 | Fold macro-F1 standard deviation | Folds at/above 0.50 |
|---|---:|---:|---:|
| LightGBM/current36/expanding | 0.49435 | 0.00849 | 1 / 5 |
| LogisticRegression/current36/expanding | 0.50605 | 0.01983 | 3 / 5 |
| LightGBM/current36/rolling120 | 0.50493 | 0.01824 | 3 / 5 |
| LightGBM/expanded86/rolling120 | 0.50607 | 0.01238 | 4 / 5 |

No configuration is reliably above chance. The current36 LightGBM expanding run varies less but stays below 0.50 macro-F1 in four of five folds. Expanded86/rolling120 is the most consistent of these higher-scoring candidates in this five-fold comparison, but the tiny selection edge and candidate search mean it is not a validated winner. Keep current36/LogisticRegression as the compact benchmark and expanded86/rolling120 as a research challenger in a new prospective test; neither is ready for production based on this sample.

## 2. Feature expansion and group ablation

### Why 86 features did not generally improve the 36-feature model

In the controlled expanding-window feature comparison, all three feature sets use LightGBM with the same parameters, equal weights, the same five folds, and the same train-only preprocessing. Expanded86 scored lower than current36 on accuracy, macro-F1, balanced accuracy, and MCC. Its ROC-AUC was marginally higher, so this is not a uniform deterioration, but the scores remain near chance. Under rolling120, expanded86 only edges current36 by roughly 0.1 percentage point macro-F1; the difference is too small and the sample too short to justify the extra inputs on performance alone.

The expanded set adds 50 features to the existing 36. The correlation audit found 39 training-set pairs with absolute Spearman correlation at least 0.95. Examples include minutes_since_open / session_phase_sin (1.00), trend_regime / sideways_regime (1.00), body_pct / candle_body (0.9997), upper_wick_pct / upper_wick (0.9992), atr_pct / atr_14 (0.9927), and overlapping return/time/slope features. The legacy40 schema also has price-level and volume-ratio redundancy risks. Correlated features can divide or destabilize tree split importance and add noise at this dataset size.

On the October validation fold, permutation importance was mixed. The largest positive decreases in macro-F1 after permutation included vwap_distance, return_std_20, atr_pct, ema_slope_20, and close_position. Several features had negative permutation importance, including trend_strength_atr, volume_acceleration, distance_prior_low_36, volume_ratio_20, range_compression_12_atr, and gap_from_previous_close_pct. This is one validation fold, and correlated inputs can make individual permutation rankings unreliable; these are candidates to simplify, not proof of universal harm.

### Ablation controls and results

The original ablation implementation removes one advanced group at a time from expanded86 and refits the same fixed LightGBM on the same five folds with equal weights and expanding training. The current36_baseline is also evaluated on those folds. These runs are comparable to each other, but they are **not** the selected rolling120 policy.

To check the selected training policy as well, this audit additionally ran all eight group removals under the same 120-session rolling window. Every one uses the same five folds, LightGBM configuration, equal weighting, and train-only medians. audit_rolling120_feature_ablation.csv contains the fold metrics.

Mean difference below is **full expanded86 minus the model with that group removed**; positive means that retaining the group scored higher. Counts show folds in which full expanded86 scored higher / lower:

| Group | Rolling120 macro-F1 difference | Folds full higher / lower | Interpretation |
|---|---:|---:|---|
| Time | +0.01084 | 4 / 1 | Strongest apparent contribution in this run |
| Volume | +0.00572 | 4 / 1 | Apparent contribution; still fold-sensitive |
| Trend | +0.00423 | 4 / 1 | Apparent contribution, with one sizeable reversal |
| Structure | +0.00252 | 3 / 2 | Small and inconsistent |
| Candle | +0.00283 | 2 / 3 | Inconsistent |
| Momentum | +0.00119 | 3 / 2 | Small |
| Volatility | +0.00092 | 2 / 3 | No consistent benefit |
| Regime | -0.00082 | 2 / 3 | Removing it slightly improved mean macro-F1 |

These are descriptive ablations across five months, without uncertainty intervals or correction for trying multiple groups. No group is established as a genuine, durable improvement. In expanding-window ablations, candle, momentum, and volatility groups reduced mean macro-F1; that reversal under rolling120 shows that group effects interact with training window and market period. Avoid adding features solely to raise feature count. If simplifying, test a compact time/volume/trend subset against current36 on newly collected holdout data before retaining it.

## 3. Trading metric audit

The independently reconstructed trade results match the saved final-test values to floating-point precision. Detailed check: audit_independent_trading_check.json.

- **Universe:** 455 labeled test predictions from November 2–9, 2026; 228 trades passed the non-overlap rule.
- **Entry:** close of the completed 5-minute candle at timestamp t.
- **Exit:** close of the exact same-session 5-minute candle timestamped t + 10 minutes. The direct price-derived returns matched the target return fraction with maximum absolute error about 1.1e-16.
- **Direction:** UP is long; DOWN is short. Every prediction is a position; there is no abstention in the binary model.
- **Selection:** keep anchors whose New York minute-of-day offset from 09:30 is divisible by 10. Selected entries are at least 10 minutes apart within each session; the held intervals do not overlap. There are 38 eligible entries per full session in this sample.
- **Trade return:** signed_return = (exit_close / entry_close - 1) * (+1 for UP, -1 for DOWN).
- **Cumulative gross return:** compound the chronological trade returns as product(1 + signed_return) - 1, yielding **1.1586%**.
- **Maximum drawdown:** minimum trade-level equity drawdown from a running peak (starting equity 1), yielding **-5.9684%**.
- **Profit factor:** sum of positive signed trade returns divided by the absolute sum of negative signed trade returns, yielding **1.0513**.

The arithmetic and non-overlap handling are correct for these assumptions. A 1.16% gross return and 1.051 profit factor across only 228 trades/six sessions are not robust evidence of profitability. No commissions, bid-ask spread, slippage, market impact, or execution delay were modeled. Net profitability cannot be claimed.

## 4. Retraining experiments: implemented versus tested

src/experiments/retraining.py contains unit-tested helpers for daily/weekly/monthly calendar cadence checks, admitting labels only after the t+10 exit candle has completed (t+15 from the input candle start), reference-binned PSI drift alerts, and manual champion/challenger review checks.

There has been **no chronological forecast replay comparing daily versus weekly retraining**, and no drift-triggered retraining backtest. Those policies have only helper-level unit tests and a design description; there are no policy-level performance results. No live retraining or scheduling is implemented. Champion/challenger checks return eligibility for human review and cannot automatically promote a model.

## 5. Challenger inference compatibility

The challenger package was loaded and its November test metrics were reproduced exactly from the saved artifact. Its 86-column ordered manifest matches generate_experimental_features and the training matrix. Its saved training medians cover those columns. The reproducible inference sequence is:

The final challenger estimator was fitted on the last 120 eligible sessions from May 12 through October 30, 2026; the November 2–9 block is not included in its fit. The saved artifact hash and training-data hash are recorded in selected_challenger_metadata.json.

1. Construct the causal feature matrix with src/experiments/features.py.
2. Select the saved 86 columns in the recorded order.
3. Replace infinities with NaN and fill using medians saved from the training subset.
4. Pass the resulting matrix to the frozen LightGBM estimator.

This confirms a repository-level inference recipe, not a ready production service. The joblib bundle does not include the feature-builder code/version hash or an executable preprocessing wrapper. The feature generator uses the row candle's OHLCV and does not receive an as_of time or reject an incomplete latest candle. A live caller must ensure each input candle is completed before feature generation; that enforcement was not implemented or tested here. The training data is an offline snapshot assumed complete through its last stored candle.

The challenger remains separate at artifacts/experiments/amzn_ml_upgrade_v1_challenger.joblib. The production 40-feature LightGBM was not changed. Challenger predict_proba values are not calibrated confidence estimates.

## 6. Recommendation and unresolved limits

- **Keep the 86-feature/120-session challenger research-only.** It reproduces the reported result, but its selection edge over a simpler current36 Logistic Regression configuration is negligible on validation and comes from a small candidate search.
- **Use current36 as the compact comparison baseline** in future prospective tests. Under rolling120, current36 LightGBM is close to expanded86 while using 50 fewer inputs. No configuration has a demonstrated stable edge over chance across enough independent periods to recommend deployment.
- **Do not replace the production 40-feature model.** Its later-period comparison is also not certified because the saved artifact lacks its exact training cutoff/provenance.
- The post-October final block is only six sessions. More prospective data is required across distinct volatility/trend regimes—preferably several additional months, with multiple years beneficial for regime coverage.
- Record data-source retrieval/completion timestamps, model training cutoff, source/code and feature-pipeline hashes, and exact model parameters for every version.
- Add a live-safe completed-candle gate before using the challenger; validate DST, holidays, early closes, missing/delayed bars, and inferred features from recorded live-like streams.
- Measure spread, fees, slippage, and execution timing in a separate realistic backtest before discussing net returns.
- Add chronological retraining-policy replays, probability calibration evaluated on separate data, and a longer untouched champion/challenger period before any promotion decision.
- XGBoost, CatBoost, SHAP, and live Alpaca data were unavailable/not used in this audit; no results are claimed for them.

## Audit artifacts and reproduction

- Re-run reconciliation, fold predictions, rolling120 ablations, inference reproduction, and independent trade arithmetic: python scripts/audit_ml_research.py
- audit_authoritative_validation_comparison.csv
- audit_recomputed_fold_metrics.csv
- audit_reconstructed_validation_predictions.csv
- audit_feature_group_deltas.csv
- audit_rolling120_feature_ablation.csv
- audit_challenger_inference.json
- audit_independent_trading_check.json

The production model and artifacts were read-only throughout this audit.
