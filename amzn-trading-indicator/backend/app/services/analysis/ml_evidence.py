"""Read-only adapter to the frozen production 40-feature classifier."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.features.legacy_engineering import FEATURE_COLUMNS, REQUIRED_INPUT_COLUMNS, generate_features
from src.forecasts.legacy_sessions import (
    MODEL_PATH, MODEL_VERSION, _artifact_sha256, load_legacy_model,
)


def latest_ml_evidence(completed_candles: pd.DataFrame) -> dict:
    """Infer from the last supplied completed candle; never loads the research challenger."""
    try:
        model, columns, classes = load_legacy_model(MODEL_PATH)
        if columns != FEATURE_COLUMNS or len(columns) != 40:
            raise ValueError("production model feature manifest mismatch")
        missing_inputs = [name for name in REQUIRED_INPUT_COLUMNS if name not in completed_candles.columns]
        if missing_inputs:
            raise ValueError(f"Completed candles are missing legacy model input column(s): {missing_inputs}")
        if completed_candles.empty:
            raise ValueError("No completed candles are available for production inference")
        featured = generate_features(completed_candles.copy(), reset_daily=False)
        missing_features = [name for name in columns if name not in featured.columns]
        if missing_features:
            raise ValueError(f"Legacy feature pipeline did not generate required feature column(s): {missing_features}")
        row = featured.iloc[-1]
        values = row.loc[columns].to_numpy(dtype=float)
        if values.shape != (40,):
            raise ValueError(f"Legacy model input shape is {values.shape}; expected exactly 40 ordered features")
        unavailable = [name for name, value in zip(columns, values) if not np.isfinite(value)]
        if unavailable:
            raise ValueError(f"Latest legacy feature values are unavailable (warm-up or source data missing): {unavailable}")
        x = pd.DataFrame([values], columns=columns, dtype=float)
        probabilities = model.predict_proba(x)[0]
        raw_up = float(probabilities[classes.index(1.0)])
        direction = "UP" if int(model.predict(x)[0]) == 1 else "DOWN"
        return {
            "available": True, "direction": direction, "raw_probability_up": raw_up,
            "raw_directional_score": raw_up if direction == "UP" else 1.0 - raw_up,
            "score_type": "uncalibrated_raw_model_score_not_confidence",
            "model_name": type(model).__name__, "model_version": MODEL_VERSION,
            "artifact_sha256": _artifact_sha256(MODEL_PATH), "input_feature_count": len(columns),
            "feature_manifest": columns, "candle_timestamp": pd.Timestamp(row.timestamp).isoformat(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        return {
            "available": False, "direction": None, "raw_probability_up": None,
            "raw_directional_score": None, "score_type": "unavailable",
            "model_version": MODEL_VERSION, "input_feature_count": len(FEATURE_COLUMNS),
            "reason": f"{type(exc).__name__}: {exc}",
        }
