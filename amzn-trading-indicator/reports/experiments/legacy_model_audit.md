# Legacy 10-minute model audit

## Recovery and compatibility

The original feature implementation was recovered from Git commit `e1b5cf0` as `src/features/legacy_engineering.py`. Its 40 ordered features exactly match `artifacts/feature_columns.json`; it remains separate from the current feature pipeline. The stored target is exact close-to-close return at `t + 10 minutes`, within the same New York regular session. Labels are UP (`1`) or DOWN (`0`) beyond a symmetric threshold. With the saved config threshold at 0 bps, NEUTRAL is not a trained target class. The artifact models return class probabilities, but no probability calibration artifact exists.

All serialized artifacts accept the 40-column vector: `best_model.joblib` / HistGradientBoosting, LogisticRegression pipeline with StandardScaler, RandomForest, HistGradientBoosting, and LightGBM. Use the manifest order. The other three tree models have no scaler.

## Chronological test comparison

The local cleaned dataset was featurized with the recovered legacy implementation, labeled with the repository target builder, then split by complete sessions: train 176 sessions, validation 38, test 37. Test period was 2026-09-03 through 2026-10-26 (2,791 labeled rows). The persisted artifacts were evaluated without fitting or overwriting models. Confusion matrices use `[[TN,FP],[FN,TP]]`. Majority baseline test accuracy was 50.66% (always DOWN, since test class distribution was 1,414 DOWN / 1,377 UP).

| Artifact | Accuracy | Precision (UP) | Recall (UP) | F1 (UP) | ROC-AUC | Brier (raw) | Confusion matrix |
|---|---:|---:|---:|---:|---:|---:|---|
| best_model / HistGradientBoosting | 48.73% | 48.01% | 47.28% | 0.4764 | 0.4824 | 0.2611 | [[709,705],[726,651]] |
| Logistic Regression | 49.16% | 48.10% | 38.56% | 0.4281 | 0.4887 | 0.2511 | [[841,573],[846,531]] |
| Random Forest | 49.70% | 48.84% | 41.18% | 0.4468 | 0.4946 | 0.2508 | [[820,594],[810,567]] |
| HistGradientBoosting | 48.73% | 48.01% | 47.28% | 0.4764 | 0.4824 | 0.2611 | [[709,705],[726,651]] |
| LightGBM | 49.01% | 48.29% | 47.20% | 0.4774 | 0.4877 | 0.2604 | [[718,696],[727,650]] |

No artifact exceeds majority-baseline accuracy. LightGBM has the highest positive-class F1 by a very small margin; Random Forest has the highest accuracy among model artifacts but remains below baseline. There is no compelling evidence of a predictive edge. Brier scores are reported only as raw probability diagnostics; they are not calibration evidence. No UP/DOWN/NEUTRAL three-class metrics exist because NEUTRAL is not trained.

## Important forecasting constraint

The recovered model predicts a 10-minute move from a feature row at time `t`, not a whole-session path. A complete schedule generated before the session cannot use later interval observations without leakage. The current session generator freezes an ADVANCE schedule using prior-session features and varies only deterministic time-of-day fields across future slots; UPDATED afternoon uses the completed morning snapshot. These are fixed schedule forecasts, not rolling refreshed forecasts. Because the model has weak out-of-sample performance and uncalibrated scores, the UI labels probability as a raw score and confidence as unavailable. It does not generate predicted OHLC candles or learned NEUTRAL interval outputs.
