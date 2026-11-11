"""Reconcile saved AMZN research metrics and audit challenger inference."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from src.experiments.evaluation import classification_metrics, trading_metrics
from src.experiments.features import ADVANCED_GROUPS
from src.experiments.pipeline import (DEFAULT_DATA, DEFAULT_OUTPUT, build_folds,
                                      evaluate_one_fold, load_dataset,
                                      prepare_examples, sha256)


def _read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DEFAULT_OUTPUT / name)


def _saved_reference(config: str, fold: str) -> dict:
    if config == "LightGBM · current36 · equal · expanding":
        frame = _read_csv("model_comparison_folds.csv")
        row = frame[(frame.model == "LightGBM") & (frame.fold == fold)].iloc[0]
    elif config == "LightGBM · expanded86 · equal · expanding":
        frame = _read_csv("feature_set_comparison_folds.csv")
        row = frame[(frame.feature_set == "expanded") & (frame.fold == fold)].iloc[0]
    elif config == "LightGBM · expanded86 · equal · rolling120":
        frame = _read_csv("rolling_window_folds.csv")
        row = frame[(frame.rolling_sessions == 120) & (frame.fold == fold)].iloc[0]
    elif config == "LogisticRegression · current36 · equal · expanding":
        frame = _read_csv("model_comparison_folds.csv")
        row = frame[(frame.model == "LogisticRegression") & (frame.fold == fold)].iloc[0]
    elif config == "LightGBM · legacy40 · equal · expanding":
        frame = _read_csv("feature_set_comparison_folds.csv")
        row = frame[(frame.feature_set == "legacy40") & (frame.fold == fold)].iloc[0]
    else:
        return None
    return row.to_dict()


def _metric_diff(actual: dict, expected: dict, keys: list[str]) -> dict:
    result = {}
    for key in keys:
        a, e = actual.get(key), expected.get(key)
        if pd.isna(a) and pd.isna(e):
            result[key] = 0.0
        elif a is None or e is None:
            result[key] = None if a == e else float("inf")
        else:
            result[key] = float(abs(float(a) - float(e)))
    return result


def _run_config(name: str, features: pd.DataFrame, model_name: str, folds, examples,
                rolling_sessions: int | None = None):
    rows, predictions = [], []
    metric_keys = ["accuracy", "f1_macro", "balanced_accuracy", "roc_auc", "mcc",
                   "trade_cumulative_gross_return", "trade_max_drawdown", "trade_profit_factor"]
    for fold in folds:
        metrics, _, detail = evaluate_one_fold(model_name, features, examples, fold,
                                                strategy="equal", rolling_sessions=rolling_sessions)
        train_ix = fold.train_indices.copy()
        if rolling_sessions is not None:
            available_dates = sorted(examples.trading_date.iloc[train_ix].unique())
            recent = set(available_dates[-rolling_sessions:])
            train_ix = train_ix[np.isin(examples.trading_date.iloc[train_ix], list(recent))]
        training_range = {
            "train_start": str(examples.trading_date.iloc[train_ix[0]]),
            "train_end": str(examples.trading_date.iloc[train_ix[-1]]),
            "train_n": len(train_ix),
            "train_sessions": int(examples.trading_date.iloc[train_ix].nunique()),
            "validation_start": fold.validation_start,
            "validation_end": fold.validation_end,
            "validation_n": int(len(fold.validation_indices)),
        }
        record = {"configuration": name, "fold": fold.name, **training_range, **metrics}
        saved = _saved_reference(name, fold.name)
        if saved is None:
            record["saved_reference_available"] = False
            record["max_abs_saved_metric_difference"] = np.nan
            record["matches_saved_metrics"] = np.nan
        else:
            record["saved_reference_available"] = True
            differences = _metric_diff(metrics, saved, metric_keys)
            record["max_abs_saved_metric_difference"] = max(differences.values())
            record["matches_saved_metrics"] = record["max_abs_saved_metric_difference"] < 1e-10
        rows.append(record)
        indices = detail["validation_indices"]
        frame = examples.iloc[indices]
        for local_pos, idx in enumerate(indices):
            predictions.append({"configuration": name, "fold": fold.name,
                                "timestamp_utc": examples.timestamp.iloc[idx].isoformat(),
                                "target": int(detail["y_true"][local_pos]),
                                "prediction": int(detail["y_pred"][local_pos]),
                                "raw_p_up": float(detail["p_up"][local_pos]),
                                "forward_return_pct": float(frame.forward_return_10m.iloc[local_pos])})
    return pd.DataFrame(rows), pd.DataFrame(predictions)


def _group_ablation() -> pd.DataFrame:
    full = _read_csv("feature_set_comparison_folds.csv")
    full = full[full.feature_set == "expanded"].set_index("fold")
    ab = _read_csv("feature_ablation_folds.csv")
    rows = []
    for _, group in ab[ab.ablation != "current36_baseline"].groupby("ablation"):
        joined = group.set_index("fold").add_suffix("_ablated").join(full.add_suffix("_full"), how="inner")
        for metric in ("accuracy", "f1_macro", "balanced_accuracy", "roc_auc", "mcc"):
            delta = joined[f"{metric}_full"] - joined[f"{metric}_ablated"]
            rows.append({"removed_group": group.removed_group.iloc[0], "metric": metric,
                         "full_minus_without_group_mean": float(delta.mean()),
                         "folds_full_better": int((delta > 0).sum()),
                         "folds_equal": int((delta == 0).sum()),
                         "folds_without_group_better": int((delta < 0).sum()),
                         "fold_deltas": json.dumps({idx: float(value) for idx, value in delta.items()})})
    return pd.DataFrame(rows)


def _rolling120_ablation(advanced: pd.DataFrame, examples: pd.DataFrame, folds) -> pd.DataFrame:
    rows = []
    candidates = [("expanded86_full", advanced, "all_advanced")]
    candidates.extend((f"without_{group}", advanced.drop(columns=columns), group)
                      for group, columns in ADVANCED_GROUPS.items())
    for candidate_name, feature_frame, removed_group in candidates:
        for fold in folds:
            result, _, _ = evaluate_one_fold("LightGBM", feature_frame, examples, fold,
                                              strategy="equal", rolling_sessions=120)
            rows.append({"candidate": candidate_name, "removed_group": removed_group,
                         **result})
    return pd.DataFrame(rows)


def main() -> None:
    out = DEFAULT_OUTPUT
    candles, data_info = load_dataset(DEFAULT_DATA)
    prepared = prepare_examples(candles)
    examples = prepared["examples"]
    folds = build_folds(examples)
    configurations = [
        ("LightGBM · current36 · equal · expanding", prepared["current36"], "LightGBM", None),
        ("LightGBM · expanded86 · equal · expanding", prepared["advanced"], "LightGBM", None),
        ("LightGBM · expanded86 · equal · rolling120", prepared["advanced"], "LightGBM", 120),
        ("LogisticRegression · current36 · equal · expanding", prepared["current36"], "LogisticRegression", None),
        ("LightGBM · legacy40 · equal · expanding", prepared["legacy40"], "LightGBM", None),
        ("LightGBM · current36 · equal · rolling120", prepared["current36"], "LightGBM", 120),
    ]
    metric_tables, prediction_tables = [], []
    for name, features, model_name, rolling_sessions in configurations:
        metrics, predictions = _run_config(name, features, model_name, folds, examples,
                                           rolling_sessions=rolling_sessions)
        metric_tables.append(metrics)
        prediction_tables.append(predictions)
    all_metrics = pd.concat(metric_tables, ignore_index=True)
    all_predictions = pd.concat(prediction_tables, ignore_index=True)
    all_metrics.to_csv(out / "audit_recomputed_fold_metrics.csv", index=False)
    all_predictions.to_csv(out / "audit_reconstructed_validation_predictions.csv", index=False)
    _group_ablation().to_csv(out / "audit_feature_group_deltas.csv", index=False)
    rolling_ablation = _rolling120_ablation(prepared["advanced"], examples, folds)
    rolling_ablation.to_csv(out / "audit_rolling120_feature_ablation.csv", index=False)

    summary = all_metrics.groupby("configuration", as_index=False).agg(
        accuracy=("accuracy", "mean"), macro_f1=("f1_macro", "mean"),
        balanced_accuracy=("balanced_accuracy", "mean"), roc_auc=("roc_auc", "mean"),
        mcc=("mcc", "mean"), avg_fold_gross_return=("trade_cumulative_gross_return", "mean"),
        average_train_sessions=("train_sessions", "mean"), min_train_date=("train_start", "min"),
        max_train_date=("train_end", "max"),
        all_folds_match_saved=("matches_saved_metrics",
                               lambda values: "N/A" if values.isna().all() else bool(values.dropna().all())))
    summary.to_csv(out / "audit_authoritative_validation_comparison.csv", index=False)

    bundle_path = BASE / "artifacts" / "experiments" / "amzn_ml_upgrade_v1_challenger.joblib"
    bundle = joblib.load(bundle_path)
    feature_columns = list(bundle["feature_columns"])
    medians = pd.Series(bundle["medians"], dtype=float).reindex(feature_columns)
    if feature_columns != list(prepared["advanced"].columns):
        raise AssertionError("saved challenger feature manifest does not match training feature matrix ordering")
    if medians.isna().any() or len(feature_columns) != 86:
        raise AssertionError("saved challenger preprocessing metadata is incomplete")
    test_start = pd.Timestamp("2026-11-01")
    test_idx = np.flatnonzero((pd.to_datetime(examples.trading_date) >= test_start) & examples.target_valid)
    X = prepared["advanced"].iloc[test_idx][feature_columns].replace([np.inf, -np.inf], np.nan).fillna(medians)
    pred = np.asarray(bundle["model"].predict(X.to_numpy(float)), dtype=int)
    classes = list(bundle["model"].classes_)
    p_up = np.asarray(bundle["model"].predict_proba(X.to_numpy(float)))[:, classes.index(1)]
    y = examples.target_10m.iloc[test_idx].astype(int).to_numpy()
    train_idx = np.flatnonzero((pd.to_datetime(examples.trading_date) < test_start) & examples.target_valid)
    dates = sorted(examples.trading_date.iloc[train_idx].unique())
    train_idx = train_idx[np.isin(examples.trading_date.iloc[train_idx], dates[-120:])]
    inference_metrics = classification_metrics(y, pred, p_up, examples.target_10m.iloc[train_idx].astype(int))
    inference_metrics.update({f"trade_{key}": value for key, value in
                              trading_metrics(examples.iloc[test_idx].reset_index(drop=True), pred).items()})
    stored_metrics = json.loads((out / "untouched_final_test.json").read_text(encoding="utf-8"))
    for key in ("accuracy", "f1_macro", "balanced_accuracy", "roc_auc", "brier_score_raw",
                "trade_cumulative_gross_return", "trade_max_drawdown", "trade_profit_factor"):
        if not np.isclose(inference_metrics[key], stored_metrics[key], atol=1e-12):
            raise AssertionError(f"saved challenger inference did not reproduce {key}")

    # Independently reconstruct the trade sample from candle timestamps and
    # prices, rather than relying on the helper being audited.
    stamp_to_index = {pd.Timestamp(stamp): i for i, stamp in enumerate(candles.timestamp)}
    prediction_by_index = {int(i): int(pred[pos]) for pos, i in enumerate(test_idx)}
    direct_returns, direct_stamps, target_differences = [], [], []
    for idx in test_idx:
        stamp = pd.Timestamp(examples.timestamp.iloc[idx])
        local = stamp.tz_convert("America/New_York")
        minutes_since_open = local.hour * 60 + local.minute - (9 * 60 + 30)
        if minutes_since_open < 0 or minutes_since_open % 10:
            continue
        future_idx = stamp_to_index.get(stamp + pd.Timedelta(minutes=10))
        if future_idx is None or candles.timestamp.iloc[future_idx].tz_convert("America/New_York").date() != local.date():
            continue
        entry = float(candles.close.iloc[idx])
        exit_ = float(candles.close.iloc[future_idx])
        actual_return_fraction = exit_ / entry - 1.0
        target_fraction = float(examples.forward_return_10m.iloc[idx]) / 100.0
        target_differences.append(abs(actual_return_fraction - target_fraction))
        direction_sign = 1.0 if prediction_by_index[int(idx)] == 1 else -1.0
        direct_returns.append(direction_sign * actual_return_fraction)
        direct_stamps.append(stamp)
    direct_returns = np.asarray(direct_returns, dtype=float)
    wealth = np.cumprod(1.0 + direct_returns)
    peaks = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    drawdowns = wealth / peaks - 1.0
    gains = direct_returns[direct_returns > 0]
    losses = direct_returns[direct_returns < 0]
    direct_profit_factor = float(gains.sum() / abs(losses.sum())) if len(losses) else None
    grouped_times = pd.DataFrame({"stamp": direct_stamps}).assign(
        date=lambda d: pd.to_datetime(d.stamp, utc=True).dt.tz_convert("America/New_York").dt.date)
    spacing = grouped_times.groupby("date").stamp.apply(
        lambda values: pd.Series(values).diff().dropna().dt.total_seconds().div(60).tolist())
    interval_spacing_ok = all(all(minutes >= 10 for minutes in day) for day in spacing)
    independent_trading = {
        "trade_count": len(direct_returns),
        "gross_compounded_return": float(wealth[-1] - 1.0),
        "maximum_trade_level_drawdown": float(drawdowns.min()),
        "profit_factor": direct_profit_factor,
        "win_rate": float(np.mean(direct_returns > 0)),
        "max_abs_return_difference_vs_target_fraction": float(max(target_differences)),
        "same_day_exact_t_plus_10_candle_match": True,
        "entry": "completed 5-minute candle close at timestamp t",
        "exit": "close of exact same-session candle timestamped t+10 minutes",
        "position": "UP prediction = long; DOWN prediction = short",
        "selected_anchor_rule": "minute-of-day from 09:30 is divisible by 10",
        "minimum_within_session_entry_spacing_minutes": min(
            [minute for values in spacing for minute in values], default=None),
        "non_overlapping_entries_confirmed": interval_spacing_ok,
        "fees_spread_slippage_modeled": False,
    }
    for actual_key, expected_key in (("gross_compounded_return", "trade_cumulative_gross_return"),
                                     ("maximum_trade_level_drawdown", "trade_max_drawdown"),
                                     ("profit_factor", "trade_profit_factor")):
        if not np.isclose(independent_trading[actual_key], stored_metrics[expected_key], atol=1e-12):
            raise AssertionError(f"independent trading recomputation failed for {actual_key}")
    if not interval_spacing_ok:
        raise AssertionError("selected entry timestamps overlap within at least one session")
    (out / "audit_independent_trading_check.json").write_text(
        json.dumps(independent_trading, indent=2, default=str), encoding="utf-8")
    inference_audit = {
        "model_name": bundle["model_name"], "artifact_sha256": sha256(bundle_path),
        "data_sha256": data_info["sha256"], "feature_count": len(feature_columns),
        "feature_manifest_matches_training_matrix": True, "training_medians_complete": True,
        "inference_reproduces_saved_final_test_metrics": True, "final_test_metrics": inference_metrics,
        "feature_builder": "src.experiments.features.generate_experimental_features",
        "preprocessing": "replace +/-inf with NaN, fill with stored training-only medians, apply frozen estimator",
        "input_bar_contract": "features use the OHLCV of the row candle; caller must ensure row candle has completed; artifact does not enforce completed-candle cutoff",
        "code_hash_or_feature_builder_version_in_bundle": False,
        "production_artifact_modified": False,
    }
    (out / "audit_challenger_inference.json").write_text(json.dumps(inference_audit, indent=2, default=str), encoding="utf-8")
    print(summary.to_string(index=False))
    referenced = all_metrics[all_metrics.saved_reference_available]
    print("All configurations with a saved reference match its fold metrics:",
          bool(referenced.matches_saved_metrics.all()))
    print("Saved challenger inference reproduces final metrics: True")
    print("Independent trade return/drawdown/profit-factor checks match: True")


if __name__ == "__main__":
    main()
