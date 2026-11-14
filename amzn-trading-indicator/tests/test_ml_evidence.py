import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.services.analysis import ml_evidence
from backend.app.services import market_store
from src.data.ingestion import latest_processed_path, load_local_candles
from src.features.legacy_engineering import FEATURE_COLUMNS, generate_features
from src.forecasts.legacy_sessions import load_legacy_model


ROOT = Path(__file__).resolve().parents[1]


def actual_completed_candles() -> pd.DataFrame:
    frame = load_local_candles(latest_processed_path())
    return frame


def test_frozen_manifest_and_legacy_generated_features_match_exactly():
    manifest = json.loads((ROOT / "artifacts" / "feature_columns.json").read_text(encoding="utf-8-sig"))
    candles = actual_completed_candles()
    featured = generate_features(candles.tail(100).copy(), reset_daily=False)
    model, model_columns, _ = load_legacy_model()
    assert manifest == FEATURE_COLUMNS == model_columns
    assert len(manifest) == 40 and model.n_features_in_ == 40
    assert all(column in featured.columns for column in manifest)
    assert list(featured.loc[:, manifest].columns) == manifest


def test_live_inference_passes_exact_ordered_40_features(monkeypatch):
    seen = {}

    class RecordingModel:
        def predict_proba(self, x):
            seen["columns"] = list(x.columns)
            seen["shape"] = x.shape
            return np.array([[0.4, 0.6]])

        def predict(self, x):
            return np.array([1])

    columns = json.loads((ROOT / "artifacts" / "feature_columns.json").read_text(encoding="utf-8-sig"))
    monkeypatch.setattr(ml_evidence, "load_legacy_model", lambda _path: (RecordingModel(), columns, [0.0, 1.0]))
    monkeypatch.setattr(ml_evidence, "_artifact_sha256", lambda _path: "test-sha")
    result = ml_evidence.latest_ml_evidence(actual_completed_candles().tail(100))
    assert result["available"] is True
    assert seen == {"columns": columns, "shape": (1, 40)}


def test_missing_vwap_returns_clear_unavailable_reason_without_prediction(monkeypatch):
    class MustNotPredict:
        def predict_proba(self, _x):
            raise AssertionError("model must not be called when vwap is unavailable")

        def predict(self, _x):
            raise AssertionError("model must not be called when vwap is unavailable")

    columns = json.loads((ROOT / "artifacts" / "feature_columns.json").read_text(encoding="utf-8-sig"))
    monkeypatch.setattr(ml_evidence, "load_legacy_model", lambda _path: (MustNotPredict(), columns, [0.0, 1.0]))
    candles = actual_completed_candles().tail(100).copy()
    candles.loc[candles.index[-1], "vwap"] = np.nan
    result = ml_evidence.latest_ml_evidence(candles)
    assert result["available"] is False
    assert "vwap_distance" in result["reason"]


def test_insufficient_lookback_returns_unavailable_without_prediction(monkeypatch):
    class MustNotPredict:
        def predict_proba(self, _x):
            raise AssertionError("model must not be called before feature warm-up")

        def predict(self, _x):
            raise AssertionError("model must not be called before feature warm-up")

    columns = json.loads((ROOT / "artifacts" / "feature_columns.json").read_text(encoding="utf-8-sig"))
    monkeypatch.setattr(ml_evidence, "load_legacy_model", lambda _path: (MustNotPredict(), columns, [0.0, 1.0]))
    result = ml_evidence.latest_ml_evidence(actual_completed_candles().head(10))
    assert result["available"] is False
    assert "sma_20" in result["reason"]


def test_frozen_lightgbm_produces_prediction_from_actual_amzn_candles():
    candles = actual_completed_candles()
    result = ml_evidence.latest_ml_evidence(candles)
    assert result["available"] is True, result.get("reason")
    assert result["model_name"] == "LGBMClassifier"
    assert result["input_feature_count"] == 40
    assert result["direction"] in {"UP", "DOWN"}
    assert 0 <= result["raw_probability_up"] <= 1
    assert result["feature_manifest"] == FEATURE_COLUMNS


def test_frozen_lightgbm_predicts_from_actual_candles_round_tripped_through_sqlite(tmp_path):
    # Exercise the same storage-to-inference column path used by the live analyzer.
    candles = actual_completed_candles().tail(250).copy()
    db = tmp_path / "market.sqlite3"
    market_store.seed_completed_candles(candles, db)
    stored = pd.DataFrame(market_store.get_candles(limit=1000, path=db))
    result = ml_evidence.latest_ml_evidence(stored)
    assert result["available"] is True, result.get("reason")
    assert result["model_name"] == "LGBMClassifier"
    assert result["feature_manifest"] == FEATURE_COLUMNS
