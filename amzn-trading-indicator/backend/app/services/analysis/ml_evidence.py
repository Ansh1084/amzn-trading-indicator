"""Read-only adapter to the frozen production 40-feature classifier."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.features.legacy_engineering import FEATURE_COLUMNS, generate_features
from src.forecasts.legacy_sessions import (
    MODEL_PATH, MODEL_VERSION, _artifact_sha256, load_legacy_model,
)


def latest_ml_evidence(completed_candles: pd.DataFrame) -> dict:
    """Infer from the last supplied completed candle; never loads the research challenger."""
    try:
        model, columns, classes = load_legacy_model(MODEL_PATH)
        if columns != FEATURE_COLUMNS or len(columns) != 40:
            raise ValueError("production model feature manifest mismatch")
        featured = generate_features(completed_candles.copy(), reset_daily=False)
        row = featured.iloc[-1]
        values = row[columns].to_numpy(dtype=float)
        if values.shape != (40,) or not np.isfinite(values).all():
            raise ValueError("latest production feature row is incomplete")
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
