"""Model training entry point.

Reusable training pipeline for the AMZN intraday trading indicator.

This module implements the full model training lifecycle:
  1. Load the cleaned dataset and apply feature engineering + target labeling
     using existing project modules.
  2. Select valid feature columns and exclude all leakage-prone columns.
  3. Handle missing values without using future information.
  4. Split data chronologically by complete trading sessions with purge/embargo.
  5. Fit preprocessing (scaling) on training data only.
  6. Train and evaluate multiple classification models.
  7. Save results, artifacts, and metadata.

Leakage Prevention Measures:
  - Target, future_close, forward_return, and target_valid columns are explicitly
    excluded from features using the labeling module's ``get_leakage_target_columns``
    and ``is_target_column`` functions.
  - Raw OHLCV prices, timestamps, symbol, and other non-feature columns are excluded.
  - Forward-fill (ffill) is used for NaN imputation — strictly causal (no bfill).
  - Preprocessing (StandardScaler) is fit on training data ONLY.
  - Chronological session-based splitting with temporal purge/embargo.
  - No random shuffling at any stage.
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.preprocessing import preprocess
from src.data.splitting import split_by_sessions
from src.evaluation.metrics import build_comparison_table, compute_classification_metrics
from src.features.engineering import FEATURE_COLUMNS, generate_features, get_feature_columns
from src.targets.labeling import (
    generate_targets,
    get_leakage_target_columns,
    is_target_column,
)

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports" / "experiments"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Columns that must NEVER enter the feature matrix
# These include: identifiers, timestamps, raw prices that leak future info,
# target/diagnostic columns, and intermediate columns.
NON_FEATURE_COLUMNS: set[str] = {
    # Identifiers and metadata
    "symbol",
    "timestamp",
    "trading_date",
    "trade_count",
    # Raw OHLCV (features are derived from these, not raw values)
    # Note: we keep 'open', 'high', 'low', 'close', 'volume', 'vwap' OUT of
    # features because the engineered features already encode this information.
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
}


def _get_safe_feature_columns(df: pd.DataFrame) -> list[str]:
    """Select valid feature columns from a DataFrame, excluding all leakage sources.

    This function uses a whitelist approach: only columns explicitly listed in
    ``FEATURE_COLUMNS`` from the engineering module are included. Additionally,
    every column is checked against the leakage target column detector.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with engineered features and target columns.

    Returns
    -------
    list[str]
        Ordered list of safe feature column names present in df.
    """
    # Whitelist: only engineered feature columns
    valid_features = get_feature_columns()

    # Additional exclusion: any column that the labeling module flags as leakage
    leakage_cols = set(get_leakage_target_columns())

    safe = []
    for col in valid_features:
        if col not in df.columns:
            logger.warning("Expected feature column '%s' not found in DataFrame.", col)
            continue
        if col in NON_FEATURE_COLUMNS:
            continue
        if col in leakage_cols:
            logger.warning("Feature column '%s' flagged as leakage target — excluded.", col)
            continue
        if is_target_column(col):
            logger.warning("Feature column '%s' detected as target column — excluded.", col)
            continue
        safe.append(col)

    return safe


def _handle_missing_values(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Handle missing values causally without using future information.

    Strategy:
      1. Forward-fill (ffill) within each split independently — strictly causal.
      2. For any remaining NaNs at the start (no prior value to ffill from),
         fill with the training set column median (computed on train only).
      3. Replace any remaining inf/-inf with NaN then fill with train median.

    Parameters
    ----------
    X_train, X_val, X_test : pd.DataFrame
        Feature matrices for each split.

    Returns
    -------
    tuple of (X_train, X_val, X_test, info_dict)
    """
    info: dict[str, Any] = {}

    # Record pre-fill NaN counts
    info["nan_before_train"] = int(X_train.isna().sum().sum())
    info["nan_before_val"] = int(X_val.isna().sum().sum())
    info["nan_before_test"] = int(X_test.isna().sum().sum())

    # Replace inf/-inf with NaN
    for split_name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
        n_inf = int(np.isinf(X.to_numpy(dtype=float, na_value=0.0)).sum())
        if n_inf > 0:
            logger.warning("%d infinite values found in %s split, replacing with NaN.", n_inf, split_name)
            info[f"inf_{split_name}"] = n_inf

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_val = X_val.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    # Forward-fill within each split (causal: no future data used)
    X_train = X_train.ffill()
    X_val = X_val.ffill()
    X_test = X_test.ffill()

    # Compute fill values from training data ONLY
    train_medians = X_train.median()
    info["fill_values_source"] = "train_median"

    # Fill remaining NaNs (e.g. initial rows with no ffill source)
    X_train = X_train.fillna(train_medians)
    X_val = X_val.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    info["nan_after_train"] = int(X_train.isna().sum().sum())
    info["nan_after_val"] = int(X_val.isna().sum().sum())
    info["nan_after_test"] = int(X_test.isna().sum().sum())

    return X_train, X_val, X_test, info


def _get_model_definitions(random_state: int = 42) -> list[dict[str, Any]]:
    """Return model definitions to train.

    Parameters
    ----------
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    list[dict]
        Each dict has keys: ``name``, ``model``, ``has_proba``.
    """
    models: list[dict[str, Any]] = [
        {
            "name": "DummyClassifier (majority)",
            "model": DummyClassifier(strategy="most_frequent", random_state=random_state),
            "has_proba": False,
        },
        {
            "name": "LogisticRegression",
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=random_state,
                    class_weight="balanced",
                )),
            ]),
            "has_proba": True,
        },
        {
            "name": "RandomForest",
            "model": RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=20,
                random_state=random_state,
                class_weight="balanced",
                n_jobs=-1,
            ),
            "has_proba": True,
        },
        {
            "name": "HistGradientBoosting",
            "model": HistGradientBoostingClassifier(
                max_iter=200,
                max_depth=6,
                learning_rate=0.05,
                min_samples_leaf=20,
                random_state=random_state,
                early_stopping=False,
            ),
            "has_proba": True,
        },
    ]

    # LightGBM — include only if installed
    try:
        import lightgbm as lgb

        models.append({
            "name": "LightGBM",
            "model": lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                min_child_samples=20,
                random_state=random_state,
                class_weight="balanced",
                verbosity=-1,
                n_jobs=-1,
            ),
            "has_proba": True,
        })
    except ImportError:
        logger.info("LightGBM not installed — skipping.")

    return models


# ── Main Training Pipeline ───────────────────────────────────────────────────


def load_and_prepare_dataset(
    raw_path: str | Path | None = None,
    clean_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load raw data, preprocess, engineer features, and generate targets.

    Uses the existing project modules for each step. If a clean CSV already exists,
    loads it and applies feature engineering + target generation on top.

    Parameters
    ----------
    raw_path : str, Path, or None
        Path to the raw Alpaca CSV. Defaults to ``data/raw/AMZN_data_1year.csv``.
    clean_path : str, Path, or None
        Path to a pre-cleaned CSV. If provided and exists, raw preprocessing is
        skipped and this file is loaded directly.

    Returns
    -------
    pd.DataFrame
        Fully prepared DataFrame with features and targets.
    """
    # Determine paths
    if clean_path is None:
        clean_path = DATA_PROCESSED_DIR / "AMZN_clean.csv"
    clean_path = Path(clean_path)

    if raw_path is None:
        raw_path = DATA_RAW_DIR / "AMZN_data_1year.csv"
    raw_path = Path(raw_path)

    # Load cleaned data or run preprocessing
    if clean_path.exists():
        logger.info("Loading pre-cleaned data from %s", clean_path)
        df = pd.read_csv(clean_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    else:
        logger.info("Running preprocessing on %s", raw_path)
        df, report = preprocess(
            raw_path,
            filter_session=True,
            naive_tz_policy="assume_utc",
            partial_dataset=True,
        )
        if not report.is_valid:
            raise RuntimeError(f"Preprocessing failed: {report.errors}")

    # Sort chronologically
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Generate engineered features
    logger.info("Generating engineered features on %d candles...", len(df))
    df = generate_features(df, reset_daily=False)

    # Generate targets
    logger.info("Generating 10-minute forward targets...")
    df = generate_targets(df, horizon_minutes=10, neutral_threshold_bps=0.0)

    return df


def run_training_pipeline(
    df: pd.DataFrame | None = None,
    *,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    purge_candles: int = 2,
    random_state: int = 42,
    save_artifacts: bool = True,
) -> dict[str, Any]:
    """Execute the complete model training and evaluation pipeline.

    Parameters
    ----------
    df : pd.DataFrame or None
        Prepared DataFrame with features and targets. If None, calls
        ``load_and_prepare_dataset()`` to build it.
    train_fraction, val_fraction, test_fraction : float
        Chronological split fractions.
    purge_candles : int
        Number of candles to purge at split boundaries.
    random_state : int
        Random seed for reproducibility.
    save_artifacts : bool
        Whether to save model artifacts and reports.

    Returns
    -------
    dict[str, Any]
        Complete experiment results including metrics, metadata, and best model info.
    """
    experiment_start = datetime.now(timezone.utc)

    # ── 1. Load and prepare data ─────────────────────────────────────────────
    if df is None:
        df = load_and_prepare_dataset()

    logger.info("Dataset shape: %s", df.shape)

    # ── 2. Identify target column ────────────────────────────────────────────
    target_col = "target_10m"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {list(df.columns)}")

    # ── 3. Filter to rows with valid targets ─────────────────────────────────
    valid_mask = df[target_col].notna()
    n_invalid = int((~valid_mask).sum())
    logger.info(
        "Target validity: %d valid, %d NaN (excluded from training).",
        int(valid_mask.sum()), n_invalid,
    )
    df_valid = df.loc[valid_mask].copy().reset_index(drop=True)

    # ── 4. Select safe feature columns ───────────────────────────────────────
    feature_cols = _get_safe_feature_columns(df_valid)
    logger.info("Selected %d feature columns: %s", len(feature_cols), feature_cols)

    # Verify no leakage columns in features
    all_leakage = set(get_leakage_target_columns())
    leaked = set(feature_cols) & all_leakage
    if leaked:
        raise RuntimeError(f"LEAKAGE DETECTED: {leaked} found in feature columns!")

    # Verify no non-feature columns slipped in
    non_feature_in = set(feature_cols) & NON_FEATURE_COLUMNS
    if non_feature_in:
        raise RuntimeError(f"Non-feature columns in feature matrix: {non_feature_in}")

    # ── 5. Chronological session-based split ─────────────────────────────────
    splits = split_by_sessions(
        df_valid,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        purge_candles=purge_candles,
    )

    df_train = splits["train"]
    df_val = splits["val"]
    df_test = splits["test"]
    split_metadata = splits["metadata"]

    logger.info(
        "Split sizes — Train: %d | Val: %d | Test: %d",
        len(df_train), len(df_val), len(df_test),
    )

    # ── 6. Extract features and targets ──────────────────────────────────────
    X_train = df_train[feature_cols].copy()
    X_val = df_val[feature_cols].copy()
    X_test = df_test[feature_cols].copy()

    y_train = df_train[target_col].to_numpy(dtype=float)
    y_val = df_val[target_col].to_numpy(dtype=float)
    y_test = df_test[target_col].to_numpy(dtype=float)

    # ── 7. Handle missing values ─────────────────────────────────────────────
    X_train, X_val, X_test, missing_info = _handle_missing_values(X_train, X_val, X_test)

    # Final NaN/inf check
    for name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
        arr = X.to_numpy(dtype=float)
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        if n_nan > 0 or n_inf > 0:
            raise RuntimeError(
                f"Feature matrix {name} still has {n_nan} NaN and {n_inf} inf values "
                "after imputation."
            )

    # Convert to numpy for model training
    X_train_np = X_train.to_numpy(dtype=float)
    X_val_np = X_val.to_numpy(dtype=float)
    X_test_np = X_test.to_numpy(dtype=float)

    # ── 8. Train and evaluate models ─────────────────────────────────────────
    model_defs = _get_model_definitions(random_state=random_state)

    val_results: list[dict[str, Any]] = []
    test_results: list[dict[str, Any]] = []
    trained_models: dict[str, Any] = {}

    for mdef in model_defs:
        name = mdef["name"]
        model = mdef["model"]
        has_proba = mdef["has_proba"]

        logger.info("Training %s...", name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train_np, y_train)

        trained_models[name] = model

        # ── Validation evaluation ────────────────────────────────────────────
        y_val_pred = model.predict(X_val_np)
        y_val_prob = None
        if has_proba:
            try:
                y_val_prob = model.predict_proba(X_val_np)[:, 1]
            except (AttributeError, IndexError):
                pass

        val_metrics = compute_classification_metrics(
            y_val, y_val_pred, y_val_prob,
            model_name=name, split_name="val",
        )
        val_results.append(val_metrics)

        # ── Test evaluation ──────────────────────────────────────────────────
        y_test_pred = model.predict(X_test_np)
        y_test_prob = None
        if has_proba:
            try:
                y_test_prob = model.predict_proba(X_test_np)[:, 1]
            except (AttributeError, IndexError):
                pass

        test_metrics = compute_classification_metrics(
            y_test, y_test_pred, y_test_prob,
            model_name=name, split_name="test",
        )
        test_results.append(test_metrics)

    # ── 9. Determine best model (based on validation F1) ─────────────────────
    # Exclude DummyClassifier from best model selection
    candidate_results = [r for r in val_results if "Dummy" not in r["model_name"]]
    if candidate_results:
        best_val = max(candidate_results, key=lambda r: r["f1_score"])
    else:
        best_val = max(val_results, key=lambda r: r["f1_score"])

    best_model_name = best_val["model_name"]
    logger.info("Best model (by validation F1): %s (F1=%.4f)", best_model_name, best_val["f1_score"])

    # ── 10. Build comparison tables ──────────────────────────────────────────
    val_table = build_comparison_table(val_results)
    test_table = build_comparison_table(test_results)

    # ── 11. Assemble experiment results ──────────────────────────────────────
    experiment_results = {
        "experiment_timestamp": experiment_start.isoformat(),
        "dataset": {
            "total_candles": len(df),
            "valid_target_candles": len(df_valid),
            "invalid_target_candles": n_invalid,
            "feature_count": len(feature_cols),
            "feature_columns": feature_cols,
        },
        "split": split_metadata,
        "missing_value_handling": missing_info,
        "target_distribution": {
            "train": {
                "n_up": int((y_train == 1).sum()),
                "n_down": int((y_train == 0).sum()),
                "pct_up": float(round((y_train == 1).mean() * 100, 2)),
            },
            "val": {
                "n_up": int((y_val == 1).sum()),
                "n_down": int((y_val == 0).sum()),
                "pct_up": float(round((y_val == 1).mean() * 100, 2)),
            },
            "test": {
                "n_up": int((y_test == 1).sum()),
                "n_down": int((y_test == 0).sum()),
                "pct_up": float(round((y_test == 1).mean() * 100, 2)),
            },
        },
        "models_trained": [m["name"] for m in model_defs],
        "best_model": {
            "name": best_model_name,
            "selected_by": "validation_f1_score",
            "val_f1": best_val["f1_score"],
            "val_accuracy": best_val["accuracy"],
            "val_roc_auc": best_val["roc_auc"],
        },
        "validation_results": val_results,
        "test_results": test_results,
    }

    # ── 12. Save artifacts ───────────────────────────────────────────────────
    if save_artifacts:
        _save_artifacts(
            experiment_results=experiment_results,
            trained_models=trained_models,
            best_model_name=best_model_name,
            val_table=val_table,
            test_table=test_table,
            feature_cols=feature_cols,
        )

    return experiment_results


def _save_artifacts(
    *,
    experiment_results: dict,
    trained_models: dict,
    best_model_name: str,
    val_table: pd.DataFrame,
    test_table: pd.DataFrame,
    feature_cols: list[str],
) -> None:
    """Save trained models, metrics, and metadata to disk."""
    # Ensure directories exist
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save best model
    best_model = trained_models[best_model_name]
    model_path = ARTIFACTS_DIR / "best_model.joblib"
    joblib.dump(best_model, model_path)
    logger.info("Saved best model (%s) to %s", best_model_name, model_path)

    # Save all models
    for name, model in trained_models.items():
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        path = ARTIFACTS_DIR / f"model_{safe_name}.joblib"
        joblib.dump(model, path)

    # Save feature column list
    feature_path = ARTIFACTS_DIR / "feature_columns.json"
    with open(feature_path, "w") as f:
        json.dump(feature_cols, f, indent=2)

    # Save experiment results (without confusion matrices as they aren't JSON-serializable nicely)
    results_path = REPORTS_DIR / "experiment_results.json"
    # Make a serializable copy
    serializable_results = _make_json_serializable(experiment_results)
    with open(results_path, "w") as f:
        json.dump(serializable_results, f, indent=2, default=str)
    logger.info("Saved experiment results to %s", results_path)

    # Save comparison tables as CSV
    val_table.to_csv(REPORTS_DIR / "validation_comparison.csv", index=False)
    test_table.to_csv(REPORTS_DIR / "test_comparison.csv", index=False)
    logger.info("Saved comparison tables to %s", REPORTS_DIR)


def _make_json_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ── CLI Entry Point ──────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point for running the training pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("=" * 70)
    logger.info("AMZN Intraday Trading Indicator — Model Training Pipeline")
    logger.info("=" * 70)

    results = run_training_pipeline()

    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 70)

    print(f"\nDataset: {results['dataset']['valid_target_candles']} valid candles, "
          f"{results['dataset']['feature_count']} features")

    print(f"\nSplit: Train={results['split']['n_train_candles']} | "
          f"Val={results['split']['n_val_candles']} | "
          f"Test={results['split']['n_test_candles']}")

    print("\n--- Validation Results ---")
    val_table = build_comparison_table(results["validation_results"])
    print(val_table.to_string(index=False))

    print("\n--- Test Results ---")
    test_table = build_comparison_table(results["test_results"])
    print(test_table.to_string(index=False))

    print(f"\nBest Model (by val F1): {results['best_model']['name']} "
          f"(F1={results['best_model']['val_f1']:.4f})")


if __name__ == "__main__":
    main()
