"""Reproducible walk-forward research runner; outputs never touch deployed artifacts."""
from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.experiments.evaluation import classification_metrics, combine_metrics, trading_metrics
from src.experiments.features import ADVANCED_FEATURES, ADVANCED_GROUPS, generate_experimental_features
from src.features.engineering import FEATURE_COLUMNS as CURRENT_FEATURES
from src.features.legacy_engineering import FEATURE_COLUMNS as LEGACY_FEATURES
from src.features.legacy_engineering import generate_features as generate_legacy_features
from src.forecasts.calendar import is_trading_day, market_timestamp, session_close
from src.targets.labeling import generate_targets


BASE = Path(__file__).resolve().parents[2]
DEFAULT_DATA = BASE / "data" / "processed" / "AMZN_clean_20261110T010232.csv"
DEFAULT_OUTPUT = BASE / "reports" / "experiments" / "amzn_ml_upgrade_v1"
FINAL_TEST_START = "2026-11-01"
VALIDATION_MONTHS = ["2026-06", "2026-07", "2026-08", "2026-09", "2026-10"]
RANDOM_SEED = 42


@dataclass(frozen=True)
class Fold:
    name: str
    train_indices: np.ndarray
    validation_indices: np.ndarray
    train_end: str
    validation_start: str
    validation_end: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset(path: Path = DEFAULT_DATA) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(path)
    source_row_count = len(raw)
    raw["timestamp"] = pd.to_datetime(raw.timestamp, utc=True, errors="raise")
    raw = raw.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    local = raw.timestamp.dt.tz_convert("America/New_York")
    numeric = raw[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    invalid = (numeric.isna().any(axis=1) | (numeric.volume < 0) | (numeric.high < numeric.low) |
               (numeric.high < numeric[["open", "close"]].max(axis=1)) |
               (numeric.low > numeric[["open", "close"]].min(axis=1)))
    off_grid = local.dt.minute.mod(5).ne(0) | local.dt.second.ne(0)
    source_dates = local.dt.date
    in_calendar_session = pd.Series(
        [is_trading_day(day) and "09:30" <= stamp.strftime("%H:%M") < session_close(day).strftime("%H:%M")
         for day, stamp in zip(source_dates, local)], index=raw.index)
    outside_session_rows = int((~in_calendar_session).sum())
    # Research only regular-session candles, with early closes determined by the
    # same deterministic market calendar used by forecast generation.
    raw = raw.loc[in_calendar_session].reset_index(drop=True)
    local = raw.timestamp.dt.tz_convert("America/New_York")
    session_rows = pd.DataFrame({"date": local.dt.date, "time": local.dt.strftime("%H:%M")})
    counts = session_rows.groupby("date").size()
    observed_by_date = {day: set(pd.to_datetime(group.timestamp, utc=True)) for day, group in raw.assign(_date=local.dt.date).groupby("_date")}
    missing_expected = []
    unexpected = []
    for day, observed in observed_by_date.items():
        opening = pd.Timestamp(market_timestamp(day, "09:30")).tz_convert("UTC")
        close = pd.Timestamp(market_timestamp(day, session_close(day).strftime("%H:%M"))).tz_convert("UTC")
        expected = set(pd.date_range(opening, close - pd.Timedelta(minutes=5), freq="5min"))
        missing_expected.extend((day.isoformat(), stamp.isoformat()) for stamp in expected - observed)
        unexpected.extend((day.isoformat(), stamp.isoformat()) for stamp in observed - expected)
    dataset_info = {
        "path": str(path.resolve()), "sha256": sha256(path), "source_rows": int(source_row_count),
        "rows": int(len(raw)), "rows_outside_calendar_rth_removed": outside_session_rows,
        "dataset_start_utc": raw.timestamp.min().isoformat(), "dataset_end_utc": raw.timestamp.max().isoformat(),
        "dataset_start_et": local.min().isoformat(), "dataset_end_et": local.max().isoformat(),
        "unique_sessions": int(counts.size), "duplicate_timestamps_removed": int(source_row_count - raw.shape[0] - outside_session_rows),
        "invalid_ohlcv_rows": int(invalid.sum()), "off_five_minute_grid_rows": int(off_grid.sum()),
        "session_candle_count_distribution": {str(k): int(v) for k, v in counts.value_counts().sort_index().items()},
        "session_counts_not_42_or_78": {str(k): int(v) for k, v in counts[~counts.isin([42, 78])].items()},
        "missing_expected_rth_candles": missing_expected, "unexpected_rth_candles": unexpected,
        "timezone": "America/New_York", "bar_interval_minutes": 5,
        "completed_bar_contract": "timestamps are candle starts; offline snapshot is assumed completed through its last saved bar",
    }
    if invalid.any():
        raise ValueError(f"Found {int(invalid.sum())} invalid OHLCV rows in {path}")
    if off_grid.any():
        raise ValueError(f"Found {int(off_grid.sum())} timestamps off the 5-minute ET grid")
    return raw, dataset_info


def prepare_examples(candles: pd.DataFrame) -> dict[str, Any]:
    legacy = generate_legacy_features(candles.copy(), reset_daily=False)
    advanced = generate_experimental_features(candles)
    targets = generate_targets(candles[["timestamp", "close"]].copy(), horizon_minutes=10,
                               neutral_threshold_bps=0.0, include_validity_col=True)
    examples = candles[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    examples["target_10m"] = targets["target_10m"].to_numpy()
    examples["forward_return_10m"] = targets["forward_return_10m"].to_numpy()
    examples["future_close_10m"] = targets["future_close_10m"].to_numpy()
    examples["trading_date"] = examples.timestamp.dt.tz_convert("America/New_York").dt.date.astype(str)
    examples["target_valid"] = examples.target_10m.notna()
    target3 = generate_targets(candles[["timestamp", "close"]].copy(), horizon_minutes=10,
                               neutral_threshold_bps=5.0, include_validity_col=False)
    examples["target_neutral_5bps"] = target3["target_10m"].to_numpy()
    examples["forward_return_10m_5bps"] = target3["forward_return_10m"].to_numpy()
    legacy_features = legacy[LEGACY_FEATURES].copy()
    current_features = advanced[CURRENT_FEATURES].copy()
    advanced_features = advanced[CURRENT_FEATURES + ADVANCED_FEATURES].copy()
    return {"examples": examples, "legacy40": legacy_features, "current36": current_features,
            "advanced": advanced_features, "advanced_all": advanced}


def build_folds(examples: pd.DataFrame, validation_months=VALIDATION_MONTHS, purge_samples: int = 2) -> list[Fold]:
    dates = pd.to_datetime(examples.trading_date)
    folds = []
    test_cutoff = pd.Timestamp(FINAL_TEST_START)
    eligible = dates < test_cutoff
    for month in validation_months:
        start = pd.Timestamp(f"{month}-01")
        end = start + pd.offsets.MonthBegin(1)
        train_idx = np.flatnonzero(eligible & (dates < start) & examples.target_valid)
        val_idx = np.flatnonzero(eligible & (dates >= start) & (dates < end) & examples.target_valid)
        if purge_samples and len(train_idx):
            train_idx = train_idx[:-purge_samples]
        if not len(train_idx) or not len(val_idx):
            continue
        folds.append(Fold(month, train_idx, val_idx,
                          examples.trading_date.iloc[train_idx[-1]], examples.trading_date.iloc[val_idx[0]],
                          examples.trading_date.iloc[val_idx[-1]]))
    return folds


def make_estimator(name: str):
    if name == "LogisticRegression":
        return Pipeline([("scale", StandardScaler()),
                         ("classifier", LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED))])
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=180, max_depth=10, min_samples_leaf=20,
                                      max_features="sqrt", class_weight="balanced", n_jobs=-1,
                                      random_state=RANDOM_SEED)
    if name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(max_iter=160, max_leaf_nodes=15, learning_rate=0.05,
                                              l2_regularization=1.0, min_samples_leaf=25,
                                              early_stopping=False, random_state=RANDOM_SEED)
    if name == "LightGBM":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise RuntimeError("LightGBM is not installed") from exc
        return LGBMClassifier(n_estimators=180, num_leaves=15, max_depth=5, learning_rate=0.04,
                              min_child_samples=25, colsample_bytree=0.9, reg_lambda=1.0,
                              class_weight="balanced", verbosity=-1, n_jobs=-1, random_state=RANDOM_SEED)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=160, max_depth=4, learning_rate=0.04, subsample=0.85,
                             colsample_bytree=0.9, reg_lambda=1.0, eval_metric="logloss",
                             n_jobs=-1, random_state=RANDOM_SEED)
    if name == "CatBoost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=160, depth=5, learning_rate=0.04, verbose=False,
                                  random_seed=RANDOM_SEED, allow_writing_files=False)
    raise ValueError(f"unknown model: {name}")


def available_models() -> tuple[list[str], list[str]]:
    names = ["LogisticRegression", "RandomForest", "HistGradientBoosting", "LightGBM"]
    skipped = []
    for name, module in (("XGBoost", "xgboost"), ("CatBoost", "catboost")):
        try:
            __import__(module)
            names.append(name)
        except ImportError:
            skipped.append(f"{name}: optional package {module} is not installed")
    return names, skipped


def sample_weights(examples: pd.DataFrame, indices: np.ndarray, strategy: str) -> np.ndarray:
    dates = examples.trading_date.iloc[indices].to_numpy()
    unique = sorted(set(dates))
    if strategy == "equal":
        weights = np.ones(len(indices), dtype=float)
    elif strategy.startswith("decay_"):
        half_life = int(strategy.split("_")[1])
        rank = {day: i for i, day in enumerate(unique)}
        ages = np.asarray([len(unique) - 1 - rank[day] for day in dates], dtype=float)
        weights = np.power(2.0, -ages / half_life)
    else:
        raise ValueError(f"unsupported sample weight strategy {strategy}")
    return weights / weights.mean()


def _impute(train_x: pd.DataFrame, val_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    train_x = train_x.replace([np.inf, -np.inf], np.nan)
    val_x = val_x.replace([np.inf, -np.inf], np.nan)
    medians = train_x.median().fillna(0.0)
    return train_x.fillna(medians), val_x.fillna(medians), {k: float(v) for k, v in medians.items()}


def _fit(model, name: str, x, y, weights):
    if name == "LogisticRegression":
        model.fit(x, y, classifier__sample_weight=weights)
    else:
        model.fit(x, y, sample_weight=weights)
    return model


def _predict(model, x) -> tuple[np.ndarray, np.ndarray]:
    classes = list(model.classes_ if hasattr(model, "classes_") else model.named_steps["classifier"].classes_)
    classifier = model.named_steps["classifier"] if hasattr(model, "named_steps") else model
    pred = np.asarray(model.predict(x), dtype=int)
    proba = np.asarray(model.predict_proba(x), dtype=float)
    return pred, proba[:, classes.index(1)]


def evaluate_one_fold(name: str, feature: pd.DataFrame, examples: pd.DataFrame,
                      fold: Fold, strategy: str = "equal", rolling_sessions: int | None = None) -> tuple[dict, Any, dict]:
    train_ix = fold.train_indices.copy()
    val_ix = fold.validation_indices
    if rolling_sessions is not None:
        train_dates = sorted(examples.trading_date.iloc[train_ix].unique())
        allowed = set(train_dates[-rolling_sessions:])
        train_ix = train_ix[np.isin(examples.trading_date.iloc[train_ix], list(allowed))]
    train_x, val_x, medians = _impute(feature.iloc[train_ix], feature.iloc[val_ix])
    y_train = examples.target_10m.iloc[train_ix].astype(int).to_numpy()
    y_val = examples.target_10m.iloc[val_ix].astype(int).to_numpy()
    weights = sample_weights(examples, train_ix, strategy)
    model = _fit(make_estimator(name), name, train_x.to_numpy(float), y_train, weights)
    pred, p_up = _predict(model, val_x.to_numpy(float))
    cls = classification_metrics(y_val, pred, p_up, y_train)
    trade_frame = examples.iloc[val_ix].reset_index(drop=True)
    trade = trading_metrics(trade_frame, pred)
    result = {"model": name, "feature_count": int(feature.shape[1]), "strategy": strategy,
              "rolling_sessions": rolling_sessions, "fold": fold.name, "train_end": fold.train_end,
              "validation_start": fold.validation_start, "validation_end": fold.validation_end,
              "train_n": int(len(train_ix)), "train_sessions": int(examples.trading_date.iloc[train_ix].nunique()),
              "validation_n": int(len(val_ix)), **cls, **{f"trade_{k}": v for k, v in trade.items()}}
    return result, model, {"medians": medians, "y_true": y_val, "y_pred": pred, "p_up": p_up,
                           "validation_indices": val_ix, "feature_columns": list(feature.columns)}


def _eval_suite(feature_sets: dict[str, pd.DataFrame], examples: pd.DataFrame, folds: list[Fold],
                model_names: list[str], strategies: list[str] | None = None) -> pd.DataFrame:
    rows = []
    strategies = strategies or ["equal"]
    for feature_name, feature in feature_sets.items():
        for model_name in model_names:
            for strategy in strategies:
                for fold in folds:
                    result, _, _ = evaluate_one_fold(model_name, feature, examples, fold, strategy)
                    result["feature_set"] = feature_name
                    rows.append(result)
    return pd.DataFrame(rows)


def _ablation_suite(advanced: pd.DataFrame, examples: pd.DataFrame, folds: list[Fold]) -> pd.DataFrame:
    rows = []
    for group, columns in ADVANCED_GROUPS.items():
        features = advanced.drop(columns=columns)
        for fold in folds:
            result, _, _ = evaluate_one_fold("LightGBM", features, examples, fold)
            result.update({"ablation": f"without_{group}", "removed_group": group})
            rows.append(result)
    for fold in folds:
        result, _, _ = evaluate_one_fold("LightGBM", advanced[CURRENT_FEATURES], examples, fold)
        result.update({"ablation": "current36_baseline", "removed_group": "all_advanced"})
        rows.append(result)
    return pd.DataFrame(rows)


def _neutral_suite(advanced: pd.DataFrame, examples: pd.DataFrame, folds: list[Fold], threshold_bps: float = 5.0) -> pd.DataFrame:
    rows = []
    # target_10m_5bps is deliberately NaN for the neutral band; use finite
    # forward returns to retain those rows while excluding unavailable labels.
    valid = np.isfinite(examples.forward_return_10m_5bps.to_numpy(dtype=float))
    y3 = np.where(examples.forward_return_10m_5bps.to_numpy() > threshold_bps / 100.0, 1,
                  np.where(examples.forward_return_10m_5bps.to_numpy() < -threshold_bps / 100.0, 0, 2))
    y3[~valid] = -1
    for fold in folds:
        start = pd.Timestamp(f"{fold.name}-01")
        end = start + pd.offsets.MonthBegin(1)
        dates = pd.to_datetime(examples.trading_date)
        train_ix = np.flatnonzero(valid & (dates < start) & (dates < pd.Timestamp(FINAL_TEST_START)))
        val_ix = np.flatnonzero(valid & (dates >= start) & (dates < end) & (dates < pd.Timestamp(FINAL_TEST_START)))
        train_ix = train_ix[:-2]  # two 5-minute anchors cover the 10-minute label horizon
        train_x, val_x, _ = _impute(advanced.iloc[train_ix], advanced.iloc[val_ix])
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(n_estimators=180, num_leaves=15, max_depth=5, learning_rate=0.04,
                               min_child_samples=25, class_weight="balanced", verbosity=-1,
                               n_jobs=-1, random_state=RANDOM_SEED)
        model.fit(train_x.to_numpy(float), y3[train_ix])
        pred = model.predict(val_x.to_numpy(float)).astype(int)
        truth = y3[val_ix]
        directional = truth != 2
        directional_pred = pred != 2
        rows.append({"fold": fold.name, "threshold_bps": threshold_bps, "train_n": len(train_ix),
                     "validation_n": len(val_ix), "neutral_actual_share": float(np.mean(truth == 2)),
                     "macro_f1_3class": float(f1_score(truth, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
                     "balanced_accuracy_3class": float(balanced_accuracy_score(truth, pred)),
                     "neutral_precision": float(np.mean(truth[pred == 2] == 2)) if np.any(pred == 2) else None,
                     "neutral_recall": float(recall_score(truth, pred, labels=[2], average="macro", zero_division=0)),
                     "prediction_coverage": float(np.mean(pred != 2)),
                     "direction_accuracy_when_non_neutral": float(accuracy_score(truth[directional & directional_pred], pred[directional & directional_pred])) if np.any(directional & directional_pred) else None,
                     "confusion_matrix_down_up_neutral": __import__("sklearn.metrics", fromlist=["confusion_matrix"]).confusion_matrix(truth, pred, labels=[0, 1, 2]).tolist()})
    return pd.DataFrame(rows)


def _regime_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dimension, categories in (("time_bucket", ["opening", "midday", "closing"]),
                                  ("volatility_regime", ["high", "low"]),
                                  ("trend_regime_label", ["trending", "sideways"])):
        for category in categories:
            part = predictions.loc[predictions[dimension] == category]
            if part.empty:
                continue
            metrics = classification_metrics(part.y_true, part.y_pred, part.p_up)
            rows.append({"dimension": dimension, "category": category, **metrics})
    return pd.DataFrame(rows)


def _market_audit(examples: pd.DataFrame, features: pd.DataFrame, data_info: dict) -> dict:
    targets = examples.loc[examples.target_valid]
    test = examples.loc[examples.trading_date >= FINAL_TEST_START]
    return {
        **data_info,
        "target_definition": "Exact close[t] to close[t+10 minutes], same New York date and RTH; UP if return > 0 bps, DOWN if return < 0 bps; exact zero return is omitted as neutral/unlabeled.",
        "horizon_minutes": 10, "reference_price": "close of completed 5-minute candle at t",
        "exit_price": "close of exact 5-minute candle timestamped t+10 minutes",
        "labeled_rows": int(len(targets)), "down_rows": int((targets.target_10m == 0).sum()),
        "up_rows": int((targets.target_10m == 1).sum()), "zero_return_rows": int((examples.forward_return_10m == 0).sum()),
        "unavailable_or_neutral_rows": int((~examples.target_valid).sum()),
        "validation_months": VALIDATION_MONTHS, "final_test_start": FINAL_TEST_START,
        "final_test_labeled_rows": int(test.target_valid.sum()),
        "final_test_sessions": int(test.loc[test.target_valid, "trading_date"].nunique()),
        "feature_sets": {"legacy40": len(LEGACY_FEATURES), "current36": len(CURRENT_FEATURES), "expanded": int(features.shape[1])},
        "test_policy": "Do not use rows on or after final_test_start during fold/model selection; evaluate the selected candidate only after selection.",
    }


def run_experiments(data_path: Path = DEFAULT_DATA, output_dir: Path = DEFAULT_OUTPUT,
                    validation_months=VALIDATION_MONTHS) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candles, data_info = load_dataset(data_path)
    prepared = prepare_examples(candles)
    examples = prepared["examples"]
    folds = build_folds(examples, validation_months)
    if len(folds) != len(validation_months):
        raise ValueError(f"Expected {len(validation_months)} non-empty chronological folds, got {len(folds)}")
    model_names, skipped_models = available_models()
    all_before_test = examples.trading_date < FINAL_TEST_START
    feature_sets = {"legacy40": prepared["legacy40"], "current36": prepared["current36"], "expanded": prepared["advanced"]}
    audit = _market_audit(examples, prepared["advanced"], data_info)
    audit["optional_models_skipped"] = skipped_models
    audit["model_family_comparison_protocol"] = "Same expanding monthly date folds, same target, training-only median imputation, fixed documented parameters; no tuning on test."
    (output_dir / "data_and_target_audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")

    # Main family comparison: same current 36-feature set for all classifiers.
    family_results = _eval_suite({"current36": prepared["current36"]}, examples, folds, model_names)
    family_results.to_csv(output_dir / "model_comparison_folds.csv", index=False)
    # Isolate feature-set effects by holding LightGBM family, weights, and folds fixed.
    feature_results = _eval_suite(feature_sets, examples, folds, ["LightGBM"])
    feature_results.to_csv(output_dir / "feature_set_comparison_folds.csv", index=False)
    # Same feature/model/folds; equal, 4 decay half-lives and 4 rolling session windows.
    weight_results = _eval_suite({"expanded": prepared["advanced"]}, examples, folds, ["LightGBM"],
                                 ["equal", "decay_10", "decay_20", "decay_30", "decay_60"])
    rolling_rows = []
    for window in (30, 60, 90, 120):
        for fold in folds:
            result, _, _ = evaluate_one_fold("LightGBM", prepared["advanced"], examples, fold,
                                               strategy="equal", rolling_sessions=window)
            result["feature_set"] = "expanded"
            rolling_rows.append(result)
    rolling_results = pd.DataFrame(rolling_rows)
    weight_results.to_csv(output_dir / "time_decay_folds.csv", index=False)
    rolling_results.to_csv(output_dir / "rolling_window_folds.csv", index=False)

    ablation = _ablation_suite(prepared["advanced"], examples, folds)
    ablation.to_csv(output_dir / "feature_ablation_folds.csv", index=False)
    neutral = _neutral_suite(prepared["advanced"], examples, folds)
    neutral.to_csv(output_dir / "neutral_5bps_experiment_folds.csv", index=False)

    # Fixed, validation-only tuning rule: highest mean macro-F1; tie-break by
    # mean balanced accuracy, then fewer features. Test dates are not consulted.
    comparison = pd.concat([family_results, feature_results, weight_results, rolling_results], ignore_index=True, sort=False)
    comparison = comparison.drop_duplicates(
        ["model", "feature_set", "strategy", "rolling_sessions", "fold"], keep="first")
    val_rows = comparison.groupby(["model", "feature_set", "strategy", "rolling_sessions"], dropna=False).agg(
        mean_f1_macro=("f1_macro", "mean"), mean_balanced_accuracy=("balanced_accuracy", "mean"),
        mean_accuracy=("accuracy", "mean"), mean_mcc=("mcc", "mean"), mean_roc_auc=("roc_auc", "mean"),
        mean_cumulative_gross_return=("trade_cumulative_gross_return", "mean"),
        mean_max_drawdown=("trade_max_drawdown", "mean"), feature_count=("feature_count", "first"),
        fold_count=("fold", "count")).reset_index()
    val_rows = val_rows.sort_values(["mean_f1_macro", "mean_balanced_accuracy", "feature_count"],
                                    ascending=[False, False, True], na_position="last")
    val_rows.to_csv(output_dir / "validation_configuration_ranking.csv", index=False)
    chosen = val_rows.iloc[0].to_dict()
    feature_key = chosen["feature_set"]
    strategy = chosen["strategy"] if isinstance(chosen["strategy"], str) else "equal"
    rolling_sessions = int(chosen["rolling_sessions"]) if pd.notna(chosen["rolling_sessions"]) else None
    chosen_model = str(chosen["model"])

    # Final fit ends before the untouched post-October test block. Training-only
    # imputation and chosen sample weights are stored with the challenger.
    test_start = pd.Timestamp(FINAL_TEST_START)
    train_idx = np.flatnonzero((pd.to_datetime(examples.trading_date) < test_start) & examples.target_valid)
    if rolling_sessions:
        recent = sorted(examples.trading_date.iloc[train_idx].unique())[-rolling_sessions:]
        train_idx = train_idx[np.isin(examples.trading_date.iloc[train_idx], recent)]
    x_all = feature_sets[feature_key]
    x_train, x_test, medians = _impute(x_all.iloc[train_idx], x_all.iloc[np.flatnonzero((pd.to_datetime(examples.trading_date) >= test_start) & examples.target_valid)])
    test_idx = np.flatnonzero((pd.to_datetime(examples.trading_date) >= test_start) & examples.target_valid)
    y_train = examples.target_10m.iloc[train_idx].astype(int).to_numpy()
    y_test = examples.target_10m.iloc[test_idx].astype(int).to_numpy()
    test_model = _fit(make_estimator(chosen_model), chosen_model, x_train.to_numpy(float), y_train,
                      sample_weights(examples, train_idx, strategy))
    test_pred, test_p = _predict(test_model, x_test.to_numpy(float))
    final_test = classification_metrics(y_test, test_pred, test_p, y_train)
    final_test.update({f"trade_{k}": v for k, v in trading_metrics(examples.iloc[test_idx].reset_index(drop=True), test_pred).items()})
    final_test.update({"model": chosen_model, "feature_set": feature_key, "sample_strategy": strategy,
                       "rolling_sessions": rolling_sessions, "train_start": examples.trading_date.iloc[train_idx[0]],
                       "train_end": examples.trading_date.iloc[train_idx[-1]],
                       "test_start": examples.trading_date.iloc[test_idx[0]], "test_end": examples.trading_date.iloc[test_idx[-1]],
                       "test_sessions": int(examples.trading_date.iloc[test_idx].nunique()),
                       "return_convention": "UP=long, DOWN=short; close[t] entry to close[t+10m] exit; gross, no costs; every second 5-minute anchor only to avoid overlapping 10m positions."})
    (output_dir / "untouched_final_test.json").write_text(json.dumps(final_test, indent=2, default=str), encoding="utf-8")

    # Frozen incumbent benchmark on the same post-October block. Its training
    # provenance is not embedded in the artifact, so report metrics but flag
    # that strict historical isolation cannot be independently certified.
    incumbent_path = BASE / "artifacts" / "model_lightgbm.joblib"
    incumbent = joblib.load(incumbent_path)
    if int(getattr(incumbent, "n_features_in_", -1)) != len(LEGACY_FEATURES):
        raise ValueError("deployed incumbent feature count differs from the 40-feature legacy manifest")
    incumbent_x = prepared["legacy40"].iloc[test_idx].replace([np.inf, -np.inf], np.nan).to_numpy(float)
    incumbent_pred = np.asarray(incumbent.predict(incumbent_x), dtype=int)
    incumbent_classes = list(incumbent.classes_)
    incumbent_p_up = np.asarray(incumbent.predict_proba(incumbent_x))[:, incumbent_classes.index(1)]
    incumbent_metrics = classification_metrics(y_test, incumbent_pred, incumbent_p_up, y_train)
    incumbent_metrics.update({f"trade_{k}": v for k, v in trading_metrics(examples.iloc[test_idx].reset_index(drop=True), incumbent_pred).items()})
    incumbent_metrics.update({"artifact": str(incumbent_path.relative_to(BASE)),
                              "artifact_sha256": sha256(incumbent_path), "feature_count": len(LEGACY_FEATURES),
                              "training_provenance_verified": False,
                              "note": "Same later-period score for context; artifact training cutoff is absent, so this is not an independently certified untouched benchmark."})
    (output_dir / "incumbent_later_period_score.json").write_text(json.dumps(incumbent_metrics, indent=2, default=str), encoding="utf-8")

    # Explanations use the final validation fold only; the final test is excluded.
    importance_fold = folds[-1]
    val_train_idx = importance_fold.train_indices
    tr_x, va_x, va_medians = _impute(x_all.iloc[val_train_idx], x_all.iloc[importance_fold.validation_indices])
    importance_model = _fit(make_estimator(chosen_model), chosen_model, tr_x.to_numpy(float),
                            examples.target_10m.iloc[val_train_idx].astype(int).to_numpy(),
                            sample_weights(examples, val_train_idx, strategy))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perm = permutation_importance(importance_model, va_x.to_numpy(float),
            examples.target_10m.iloc[importance_fold.validation_indices].astype(int).to_numpy(),
            scoring="f1_macro", n_repeats=3, random_state=RANDOM_SEED, n_jobs=1)
    importance = pd.DataFrame({"feature": list(x_all.columns), "permutation_macro_f1_mean": perm.importances_mean,
                               "permutation_macro_f1_std": perm.importances_std}).sort_values("permutation_macro_f1_mean", ascending=False)
    importance.to_csv(output_dir / "validation_permutation_importance.csv", index=False)

    # Regime report based on causal, precomputed features and validation-only OOF predictions.
    pred_rows = []
    for ix, p, prob in zip(importance_fold.validation_indices, _predict(importance_model, va_x.to_numpy(float))[0],
                           _predict(importance_model, va_x.to_numpy(float))[1]):
        row = examples.iloc[ix]
        feature_row = prepared["advanced_all"].iloc[ix]
        minute = pd.Timestamp(row.timestamp).tz_convert("America/New_York").hour * 60 + pd.Timestamp(row.timestamp).tz_convert("America/New_York").minute
        bucket = "opening" if minute < 10*60 else "closing" if minute >= 15*60 else "midday"
        pred_rows.append({"timestamp": row.timestamp, "y_true": int(row.target_10m), "y_pred": int(p), "p_up": float(prob),
                          "time_bucket": bucket,
                          "volatility_regime": "high" if feature_row.get("volatility_high_regime", 0) == 1 else "low",
                          "trend_regime_label": "trending" if feature_row.get("trend_regime", 0) == 1 else "sideways"})
    regime = _regime_metrics(pd.DataFrame(pred_rows))
    regime.to_csv(output_dir / "validation_regime_analysis.csv", index=False)

    # Fit the selected challenger on all eligible pre-test examples and persist
    # it only beneath artifacts/experiments (production artifact paths untouched).
    artifact = {"model": test_model, "feature_columns": list(x_all.columns), "medians": medians,
                "model_name": chosen_model, "feature_set": feature_key, "sample_strategy": strategy,
                "rolling_sessions": rolling_sessions, "target": "close[t] to close[t+10m], exact same-session match, 0 bps neutral",
                "training_end": examples.trading_date.iloc[train_idx[-1]], "seed": RANDOM_SEED,
                "model_selection_rule": "highest expanding-fold mean macro-F1; tie by balanced accuracy; then fewer features",
                "final_test_summary": final_test}
    artifact_dir = BASE / "artifacts" / "experiments"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "amzn_ml_upgrade_v1_challenger.joblib"
    joblib.dump(artifact, model_path)
    metadata = {"model_version": "amzn-10m-experimental-v1", "model_name": chosen_model,
                "model_artifact": str(model_path.relative_to(BASE)), "model_artifact_sha256": sha256(model_path),
                "data_sha256": data_info["sha256"], "data_path": data_info["path"],
                "feature_set": feature_key, "feature_columns": list(x_all.columns), "feature_count": len(x_all.columns),
                "estimator_parameters": test_model.get_params(),
                "sample_strategy": strategy, "rolling_sessions": rolling_sessions, "validation_months": VALIDATION_MONTHS,
                "validation_configuration": chosen, "final_test": final_test,
                "production_artifacts_modified": False, "optional_models_skipped": skipped_models,
                "packages_not_installed": [name for name in ("xgboost", "catboost", "optuna", "shap")
                                           if __import__("importlib.util").util.find_spec(name) is None]}
    (output_dir / "selected_challenger_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    manifests = {"legacy40": LEGACY_FEATURES, "current36": CURRENT_FEATURES,
                 "expanded": list(prepared["advanced"].columns)}
    (output_dir / "feature_manifests.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    config = {"seed": RANDOM_SEED, "validation_months": VALIDATION_MONTHS,
              "final_test_start": FINAL_TEST_START, "purge_samples": 2,
              "target": audit["target_definition"], "model_candidates": model_names,
              "optional_models_skipped": skipped_models,
              "weighting_strategies": ["equal", "decay_10", "decay_20", "decay_30", "decay_60"],
              "rolling_training_sessions": [30, 60, 90, 120],
              "final_test_used_for_selection": False,
              "production_artifacts_modified": False}
    (output_dir / "experiment_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    # Correlation on pre-test train data only; report highly correlated pairs,
    # without removing features based on validation/test observations.
    train_corr = x_all.iloc[train_idx].corr(method="spearman").abs()
    upper = train_corr.where(np.triu(np.ones(train_corr.shape), k=1).astype(bool))
    pairs = [{"feature_a": column, "feature_b": idx, "abs_spearman": float(upper.loc[idx, column])}
             for column in upper.columns for idx in upper.index if pd.notna(upper.loc[idx, column]) and upper.loc[idx, column] >= 0.95]
    pd.DataFrame(pairs).to_csv(output_dir / "training_feature_correlations_ge_095.csv", index=False)

    return {"output_dir": output_dir, "metadata": metadata, "model_comparison": family_results,
            "feature_comparison": feature_results, "weighting": weight_results, "rolling": rolling_results,
            "ablation": ablation, "neutral": neutral, "ranking": val_rows, "importance": importance,
            "regime": regime, "final_test": final_test, "incumbent_test": incumbent_metrics, "audit": audit}
