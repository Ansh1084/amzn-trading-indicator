"""Render a Markdown report and charts from a completed ML experiment run."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
OUT = BASE / "reports" / "experiments" / "amzn_ml_upgrade_v1"


def read_json(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def table(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    view = frame[columns].copy()
    for column in view.select_dtypes(include="number"):
        view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{value:.{digits}f}")
    headers = [str(column) for column in view.columns]
    rows = [[str(value) for value in row] for row in view.itertuples(index=False, name=None)]
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    line = lambda row: "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |"
    return "\n".join([line(headers), "| " + " | ".join("-" * width for width in widths) + " |", *(line(row) for row in rows)])


def make_charts():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comparison = pd.read_csv(OUT / "model_comparison_folds.csv")
    summary = comparison.groupby("model", as_index=False)[["accuracy", "f1_macro", "balanced_accuracy", "roc_auc"]].mean()
    ax = summary.set_index("model").plot(kind="bar", figsize=(10, 5), ylim=(0.43, 0.57), rot=15)
    ax.set_ylabel("Mean across five monthly validation folds")
    ax.set_title("Classifier comparison · current 36-feature baseline")
    ax.grid(axis="y", alpha=.25)
    plt.tight_layout()
    plt.savefig(OUT / "model_comparison.png", dpi=160)
    plt.close()

    decay = pd.read_csv(OUT / "time_decay_folds.csv").groupby("strategy", as_index=False)[["f1_macro", "trade_cumulative_gross_return"]].mean()
    rolling = pd.read_csv(OUT / "rolling_window_folds.csv").groupby("rolling_sessions", as_index=False)[["f1_macro", "trade_cumulative_gross_return"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(decay.strategy, decay.f1_macro, color="#5470c6")
    axes[0].set_title("Exponential weighting · macro F1")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].plot(rolling.rolling_sessions, rolling.f1_macro, marker="o", label="Macro F1")
    axes[1].set_title("Rolling window · macro F1")
    axes[1].set_xlabel("Training sessions")
    axes[1].grid(alpha=.25)
    plt.tight_layout()
    plt.savefig(OUT / "recency_experiments.png", dpi=160)
    plt.close()

    importance = pd.read_csv(OUT / "validation_permutation_importance.csv").head(15).sort_values("permutation_macro_f1_mean")
    ax = importance.set_index("feature")["permutation_macro_f1_mean"].plot(kind="barh", figsize=(9, 6), color="#4a9b84")
    ax.set_title("Permutation importance · October validation fold")
    ax.set_xlabel("Decrease in macro F1 after permutation")
    plt.tight_layout()
    plt.savefig(OUT / "validation_feature_importance.png", dpi=160)
    plt.close()


def main():
    audit = read_json("data_and_target_audit.json")
    chosen = read_json("selected_challenger_metadata.json")
    test = read_json("untouched_final_test.json")
    incumbent = read_json("incumbent_later_period_score.json")
    manifest = read_json("feature_manifests.json")
    models = pd.read_csv(OUT / "model_comparison_folds.csv")
    feature = pd.read_csv(OUT / "feature_set_comparison_folds.csv")
    decay = pd.read_csv(OUT / "time_decay_folds.csv")
    rolling = pd.read_csv(OUT / "rolling_window_folds.csv")
    ablation = pd.read_csv(OUT / "feature_ablation_folds.csv")
    neutral = pd.read_csv(OUT / "neutral_5bps_experiment_folds.csv")
    regime = pd.read_csv(OUT / "validation_regime_analysis.csv")
    importance = pd.read_csv(OUT / "validation_permutation_importance.csv").head(15)
    ranking = pd.read_csv(OUT / "validation_configuration_ranking.csv").head(8)

    model_summary = models.groupby("model", as_index=False)[["accuracy", "precision_macro", "recall_macro", "f1_macro", "balanced_accuracy", "mcc", "roc_auc", "pr_auc_up"]].mean()
    feature_summary = feature.groupby("feature_set", as_index=False)[["accuracy", "f1_macro", "balanced_accuracy", "mcc", "roc_auc", "trade_cumulative_gross_return"]].mean()
    decay_summary = decay.groupby("strategy", as_index=False)[["accuracy", "f1_macro", "balanced_accuracy", "mcc", "roc_auc", "trade_cumulative_gross_return", "trade_max_drawdown"]].mean()
    rolling_summary = rolling.groupby("rolling_sessions", as_index=False)[["accuracy", "f1_macro", "balanced_accuracy", "mcc", "roc_auc", "trade_cumulative_gross_return", "trade_max_drawdown"]].mean()
    ablation_summary = ablation.groupby(["ablation", "removed_group"], as_index=False)[["f1_macro", "balanced_accuracy", "mcc", "trade_cumulative_gross_return"]].mean()
    baseline = pd.DataFrame([
        {"model": "Selected challenger", "accuracy": test["accuracy"], "macro_f1": test["f1_macro"], "balanced_accuracy": test["balanced_accuracy"], "roc_auc": test["roc_auc"], "gross_return": test["trade_cumulative_gross_return"]},
        {"model": "Frozen production LightGBM (provenance unknown)", "accuracy": incumbent["accuracy"], "macro_f1": incumbent["f1_macro"], "balanced_accuracy": incumbent["balanced_accuracy"], "roc_auc": incumbent["roc_auc"], "gross_return": incumbent["trade_cumulative_gross_return"]},
    ])
    make_charts()
    advanced_groups = read_json("feature_manifests.json")["expanded"]
    group_defs = read_json("experiment_config.json")
    del advanced_groups, group_defs

    report = f"""# AMZN 10-minute ML pipeline research report

Generated from local snapshot `{Path(audit['path']).name}` (SHA-256 `{audit['sha256']}`). This is an offline research run. No live Alpaca calls were made. The challenger is saved separately; no production artifact is replaced.

## Existing pipeline audit

- Production 10-minute inference uses the frozen `artifacts/model_lightgbm.joblib` LightGBM classifier with 40 positional inputs in `artifacts/feature_columns.json`; the model stores generic `Column_0…Column_39`, so order is defined by that manifest and `src/features/legacy_engineering.py`.
- Other legacy artifacts were previously inventoried in `reports/experiments/legacy_model_audit.md`. The current 36-feature window-forecast pipeline is a separate model path. It must not be substituted for the 10-minute path.
- The legacy manifest used here has {len(manifest['legacy40'])} inputs. The current feature pipeline contains {len(manifest['current36'])} model inputs. The experimental expanded matrix contains {len(manifest['expanded'])} inputs. Diagnostic target columns and OHLCV are excluded from all feature matrices.
- Redundancy risks identified from the original schema include raw-price moving average/Bollinger level columns, `bb_middle`/`sma_20`, and `volume_ratio`/`volume_ratio_20`. These are correlated/redundant candidates, not assumed exact equivalents without checking computation.
- Historical audit reports weak, near-chance performance; the old result is not mixed into these walk-forward scores because it used a different test period and feature version.

### Exact original 40-feature order

`{', '.join(manifest['legacy40'])}`

The frozen model accepts 40 inputs, has classes `{[0, 1]}`, and exposes `predict_proba`. The serialized artifact lacks reliable training-cutoff and preprocessing provenance; raw scores are therefore not described as calibrated confidence.

## Data and target

| Audit item | Result |
|---|---:|
| Source rows | {audit['source_rows']:,} |
| Kept regular-session 5-minute candles | {audit['rows']:,} |
| RTH rows removed (calendar/outside session) | {audit['rows_outside_calendar_rth_removed']} |
| Sessions | {audit['unique_sessions']} |
| Date range | {audit['dataset_start_et']} → {audit['dataset_end_et']} |
| Duplicate / invalid OHLCV / off-grid | {audit['duplicate_timestamps_removed']} / {audit['invalid_ohlcv_rows']} / {audit['off_five_minute_grid_rows']} |
| Missing / unexpected RTH bars after calendar filtering | {len(audit['missing_expected_rth_candles'])} / {len(audit['unexpected_rth_candles'])} |
| Session lengths | `{audit['session_candle_count_distribution']}` |
| Final test | {test['test_start']} → {test['test_end']}, {test['test_sessions']} sessions, {test['n']} labeled examples |

The target is exact close-to-close return from the completed candle at *t* to the close of the exact candle timestamped *t+10 minutes*, same New York date and RTH. Positive return is UP, negative is DOWN; exact zero is unlabelled for the binary task. The 5-minute anchors overlap as labels; trading-return metrics select every second anchor to avoid double-counted positions. No synthetic 10-minute OHLC candle is constructed for model labels. Incomplete/missing exact exit candles are excluded.

## Validation and model comparison

Monthly expanding folds validate June–October 2026; training is strictly earlier than each validation month, the last two 5-minute training anchors are purged, and each fold’s median imputation/scaling is fit on its training subset. November 2–9 is the post-October final block and was not used for selection. The test contains only six sessions, so uncertainty is high. Model parameters are fixed and recorded; no random split or Optuna search was used. XGBoost/CatBoost were skipped when their packages were absent.

![Classifier comparison](model_comparison.png)

{table(model_summary, ['model','accuracy','precision_macro','recall_macro','f1_macro','balanced_accuracy','mcc','roc_auc','pr_auc_up'])}

The mean fold-specific train-majority baseline accuracy was **{models.majority_baseline_accuracy.mean():.3f}**. Confusion matrices (fixed label order DOWN, UP), class-wise metrics, and per-fold baselines are retained in `model_comparison_folds.csv`.

Configuration ranking uses mean validation macro-F1, then balanced accuracy, then fewer features. The chosen validation configuration is **{chosen['model_name']} / {chosen['feature_set']} / {chosen['sample_strategy']} / {chosen['rolling_sessions']} sessions**, with mean macro-F1 {chosen['validation_configuration']['mean_f1_macro']:.4f}, balanced accuracy {chosen['validation_configuration']['mean_balanced_accuracy']:.4f}, and ROC-AUC {chosen['validation_configuration']['mean_roc_auc']:.4f}. Logistic Regression’s validation macro-F1 is nearly tied; the tiny difference is not evidence that LightGBM is meaningfully superior.

### Recency strategies

![Recent data experiments](recency_experiments.png)

{table(decay_summary, ['strategy','accuracy','f1_macro','balanced_accuracy','mcc','roc_auc','trade_cumulative_gross_return','trade_max_drawdown'])}

{table(rolling_summary, ['rolling_sessions','accuracy','f1_macro','balanced_accuracy','mcc','roc_auc','trade_cumulative_gross_return','trade_max_drawdown'])}

Results are fold means, not a guarantee of future return. Neither exponential weighting nor shorter windows consistently dominate equal weighting; the selected 120-session LightGBM configuration was the validation winner by the declared macro-F1 rule.

## New feature groups and diagnostics

The research-only implementation adds normalized trend/slopes, RSI-7 and momentum changes, volatility ratios/regimes, volume context, candle proportions, prior-only market structure, session time encodings, and causal regime flags. Rolling swing highs/lows are shifted one candle. Session and calendar features use `America/New_York`; no future swing points or backward fill are used. Discrete nearest support/resistance zones were not added because no reproducible zone-merging rule was validated; shifted prior highs/lows are used as transparent market-structure proxies. Existing production feature code remains unchanged.

{table(feature_summary, ['feature_set','accuracy','f1_macro','balanced_accuracy','mcc','roc_auc','trade_cumulative_gross_return'])}

### Group ablation (LightGBM, mean across folds)

{table(ablation_summary, ['ablation','removed_group','f1_macro','balanced_accuracy','mcc','trade_cumulative_gross_return'])}

### Permutation importance

![Permutation importance](validation_feature_importance.png)

{table(importance, ['feature','permutation_macro_f1_mean','permutation_macro_f1_std'])}

Importance is measured on the October validation fold only, not the final test. SHAP was unavailable in the environment and was not approximated. High correlation pairs (|Spearman| ≥ 0.95) are saved in `training_feature_correlations_ge_095.csv` for review.

### Regime analysis (October validation fold)

{table(regime, ['dimension','category','n','accuracy','f1_macro','balanced_accuracy','mcc','roc_auc'])}

These subgroups are observational and some are small. The causal trend/volatility tags are exploratory, not a validated regime-aware trading policy.

### Optional 5 bps neutral target experiment

The separate 3-class experiment labels returns above +5 bps UP, below −5 bps DOWN, otherwise NEUTRAL; it does not change the binary target or selected model.

{table(neutral, ['fold','validation_n','neutral_actual_share','macro_f1_3class','neutral_precision','neutral_recall','prediction_coverage'])}

Neutral prediction coverage varies by fold and neutral F1 is modest. It is not enabled in production and its class probabilities are not calibrated.

## Final test and incumbent context

{table(baseline, ['model','accuracy','macro_f1','balanced_accuracy','roc_auc','gross_return'])}

Selected challenger final test: {test['accuracy']:.3f} accuracy, {test['f1_macro']:.3f} macro-F1, {test['balanced_accuracy']:.3f} balanced accuracy, {test['roc_auc']:.3f} ROC-AUC. Majority baseline (training majority UP) is {test['majority_baseline_accuracy']:.3f}. For every-second 10-minute entry, hypothetical gross return is {test['trade_cumulative_gross_return']:.2%}, maximum drawdown {test['trade_max_drawdown']:.2%}, and profit factor {test['trade_profit_factor']:.3f}; these omit commissions, spread, slippage, and execution constraints.

The frozen incumbent’s same-period score is **context only**: its serialized artifact does not contain an auditable training cutoff, so strict no-overlap provenance cannot be certified. No production artifact was altered. Challenger raw scores and Brier/log-loss are reported as uncalibrated; calibration was not performed.

## Retraining and drift design

`src/experiments/retraining.py` provides research helpers: daily/weekly/monthly cadence checks; maturity filtering that only admits a sample after its t+10 exit candle has completed (t+15 from the input candle start); reference-binned PSI drift monitoring (default alert threshold 0.20); and explicit manual-review champion/challenger gates (minimum sample size, macro-F1 gain, balanced accuracy, drawdown, fold-level F1 stability). A passing challenger is never auto-promoted.

| Policy | Cadence / trigger | Benefits | Risks and intended use |
|---|---|---|---|
| Daily | Retrain after each completed session, using only matured labels | Reacts fastest to changes | Highest operational and selection variance; evaluate as a challenger only |
| Weekly | Retrain once per week from matured data | Lower compute and less day-to-day noise | Can lag abrupt drift; practical initial candidate cadence |
| Periodic + drift | Monthly review, or request a candidate run when any monitored PSI ≥ 0.20 | Stable baseline plus explicit drift signal | PSI can false-alarm and needs enough recent observations; drift does not itself justify promotion |

These policies are specified and their cadence, label maturity, PSI and promotion guardrails are unit-tested; this dataset run did not replay all three scheduling policies as separate forecasting backtests. Daily/weekly/periodic live retraining is not scheduled or exercised here.

## Reproducibility and artifacts

- Run: `python scripts/run_ml_experiments.py`
- Render this report and charts: `python scripts/render_ml_experiment_report.py`
- Dataset hash and run parameters: `data_and_target_audit.json`, `experiment_config.json`.
- Fold metrics: `model_comparison_folds.csv`, `feature_set_comparison_folds.csv`, `time_decay_folds.csv`, `rolling_window_folds.csv`, `feature_ablation_folds.csv`, `neutral_5bps_experiment_folds.csv`.
- Challenger model: `{chosen['model_artifact']}`; metadata, feature order, parameters and data hash are in `selected_challenger_metadata.json`.
- No frontend, backend, scheduler, or production model files are changed by this experiment code.

## Conclusion and limitations

The new experimental pipeline is reproducible and offers better diagnostics; predictive improvement is **not established robustly**. On the six-session final block the selected challenger modestly exceeds the observed majority accuracy, but the edge is small (MCC {test['mcc']:.3f}, ROC-AUC {test['roc_auc']:.3f}) and gross-return results are before costs. Five-fold validation averages remain close to chance. Continue collecting future data and require a separately frozen, provenance-tracked champion/challenger period before considering deployment. No model promotion is recommended from this short sample.
"""
    (OUT / "research_report.md").write_text(report, encoding="utf-8")
    print(f"Wrote {OUT / 'research_report.md'}")


if __name__ == "__main__":
    main()
