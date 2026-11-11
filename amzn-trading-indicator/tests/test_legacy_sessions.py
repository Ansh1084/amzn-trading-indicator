from datetime import date

import pandas as pd
import pytest

from src.features.legacy_engineering import FEATURE_COLUMNS as LEGACY_COLUMNS, generate_features
from src.forecasts.legacy_sessions import interval_schedule, load_legacy_model
from src.features.engineering import get_feature_columns


def test_recovered_git_pipeline_matches_artifact_feature_manifest_and_current_is_separate():
    import json
    from pathlib import Path
    artifact_columns = json.loads((Path(__file__).parents[1] / "artifacts" / "feature_columns.json").read_text(encoding="utf-8-sig"))
    assert LEGACY_COLUMNS == artifact_columns
    assert len(LEGACY_COLUMNS) == 40
    assert get_feature_columns() != LEGACY_COLUMNS


@pytest.mark.parametrize("name", ["best_model.joblib", "model_logisticregression.joblib", "model_randomforest.joblib", "model_histgradientboosting.joblib", "model_lightgbm.joblib"])
def test_every_legacy_artifact_accepts_the_exact_ordered_feature_vector(name):
    import json
    from pathlib import Path
    from src.forecasts.legacy_sessions import LEGACY_FEATURES_PATH, ROOT
    model, columns, classes = load_legacy_model(ROOT / "artifacts" / name)
    assert columns == json.loads(LEGACY_FEATURES_PATH.read_text(encoding="utf-8-sig"))
    assert model.n_features_in_ == len(columns) == 40
    assert set(classes) == {0.0, 1.0}
    values = pd.DataFrame([[0.01] * 40], columns=columns)
    assert model.predict(values).shape == (1,)
    assert model.predict_proba(values).shape == (1, 2)


def test_schedule_has_eighteen_intervals_for_each_normal_session():
    for session in ("morning", "afternoon"):
        intervals = interval_schedule("2026-11-10", session)
        assert len(intervals) == 18
        assert all((end - start).total_seconds() == 600 for start, end in intervals)
        assert intervals[-1][1].strftime("%H:%M") in {"12:30", "15:30"}


def test_early_close_afternoon_is_adjusted_to_market_close():
    intervals = interval_schedule("2026-11-27", "afternoon")
    assert len(intervals) == 3
    assert intervals[0][0].strftime("%H:%M") == "12:30"
    assert intervals[-1][1].strftime("%H:%M") == "13:00"


def test_non_trading_date_is_rejected():
    with pytest.raises(ValueError, match="not a trading session"):
        interval_schedule("2026-11-14", "morning")


def test_automatic_evaluation_waits_for_complete_bars_and_is_duplicate_safe(tmp_path, monkeypatch):
    import json
    from datetime import timezone
    from src.forecasts import legacy_sessions as sessions
    db_path = tmp_path / "evaluations.sqlite3"
    real_connect = sessions.connect
    monkeypatch.setattr(sessions, "connect", lambda: real_connect(db_path))
    start, end = interval_schedule("2026-11-06", "morning")[0]
    payload = {"prediction_id":"p1","evaluation_date":"2026-11-06","session_id":"morning","forecast_type":"ADVANCE","revision":1,
               "interval_start":start.isoformat(),"interval_end":end.isoformat(),"generated_at":"2026-11-09T22:00:00+00:00",
               "feature_timestamp":"2026-11-09T20:55:00+00:00","reference_price":100.0,"predicted_signal":"UP","raw_probability_up":0.6,
               "support":99.0,"resistance":101.0,"model_version":"legacy-test","model_name":"Test","confidence":None}
    with real_connect(db_path) as db:
        db.execute("INSERT INTO session_predictions(prediction_id,evaluation_date,session_id,forecast_type,revision,interval_start,interval_end,generated_at,feature_timestamp,model_name,model_version,payload) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                   ("p1","2026-11-06","morning","ADVANCE",1,start.isoformat(),end.isoformat(),payload["generated_at"],payload["feature_timestamp"],"Test","legacy-test",json.dumps(payload)))
    partial = pd.DataFrame([{"timestamp":pd.Timestamp(start).tz_convert("UTC"),"open":100,"high":101,"low":99,"close":100.5,"volume":10}])
    monkeypatch.setattr(sessions, "latest_processed_path", lambda: tmp_path / "fake.csv")
    monkeypatch.setattr(sessions, "load_local_candles", lambda _path: partial.copy())
    assert sessions.auto_evaluate_available("2026-11-06") == 0
    assert sessions.list_session_predictions("2026-11-06","morning")[0]["actual"] is None
    complete = pd.DataFrame([
        {"timestamp":pd.Timestamp(start).tz_convert("UTC"),"open":100,"high":101,"low":99,"close":100.5,"volume":10},
        {"timestamp":(pd.Timestamp(start)+pd.Timedelta(minutes=5)).tz_convert("UTC"),"open":100.5,"high":102,"low":100,"close":101.5,"volume":15},
        {"timestamp":pd.Timestamp(end).tz_convert("UTC"),"open":101.5,"high":103,"low":101,"close":102.5,"volume":12},
    ])
    monkeypatch.setattr(sessions, "load_local_candles", lambda _path: complete.copy())
    assert sessions.auto_evaluate_available("2026-11-06") == 1
    saved = sessions.list_session_predictions("2026-11-06","morning")[0]
    assert saved["actual"]["actual_signal"] == "UP"
    assert saved["actual"]["actual_close"] == 102.5
    assert saved["actual"]["outcome"] == "CORRECT"
    assert sessions.auto_evaluate_available("2026-11-06") == 0


def test_automatic_evaluation_waits_until_endpoint_5m_candle_is_complete(tmp_path, monkeypatch):
    import json
    from datetime import datetime as RealDateTime, timezone
    from src.forecasts import legacy_sessions as sessions
    db_path = tmp_path / "endpoint.sqlite3"
    real_connect = sessions.connect
    monkeypatch.setattr(sessions, "connect", lambda: real_connect(db_path))
    start, end = interval_schedule("2026-11-06", "morning")[0]
    payload = {"prediction_id":"p-endpoint","evaluation_date":"2026-11-06","session_id":"morning","forecast_type":"ADVANCE","revision":1,
               "interval_start":start.isoformat(),"interval_end":end.isoformat(),"generated_at":"2026-11-05T20:00:00+00:00",
               "feature_timestamp":"2026-11-05T20:55:00+00:00","reference_price":100.0,"predicted_signal":"UP","raw_probability_up":0.6,
               "support":99.0,"resistance":101.0,"model_version":"legacy-test","model_name":"Test","confidence":None}
    with real_connect(db_path) as db:
        db.execute("INSERT INTO session_predictions(prediction_id,evaluation_date,session_id,forecast_type,revision,interval_start,interval_end,generated_at,feature_timestamp,model_name,model_version,payload) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                   ("p-endpoint","2026-11-06","morning","ADVANCE",1,start.isoformat(),end.isoformat(),payload["generated_at"],payload["feature_timestamp"],"Test","legacy-test",json.dumps(payload)))
    bars = pd.DataFrame([
        {"timestamp":pd.Timestamp(start).tz_convert("UTC"),"open":100,"high":101,"low":99,"close":100.5,"volume":10},
        {"timestamp":(pd.Timestamp(start)+pd.Timedelta(minutes=5)).tz_convert("UTC"),"open":100.5,"high":102,"low":100,"close":101.5,"volume":15},
        {"timestamp":pd.Timestamp(end).tz_convert("UTC"),"open":101.5,"high":103,"low":101,"close":102.5,"volume":12},
    ])
    monkeypatch.setattr(sessions, "latest_processed_path", lambda: tmp_path / "fake.csv")
    monkeypatch.setattr(sessions, "load_local_candles", lambda _path: bars.copy())
    class FrozenDateTime(RealDateTime):
        now_at = None
        @classmethod
        def now(cls, tz=None):
            return cls.now_at.astimezone(tz) if tz else cls.now_at.replace(tzinfo=None)
    monkeypatch.setattr(sessions, "datetime", FrozenDateTime)
    endpoint = pd.Timestamp(end).tz_convert("UTC")
    FrozenDateTime.now_at = (endpoint + pd.Timedelta(minutes=4)).to_pydatetime()
    assert sessions.auto_evaluate_available("2026-11-06") == 0
    assert sessions.list_session_predictions("2026-11-06","morning")[0]["actual"] is None
    FrozenDateTime.now_at = (endpoint + pd.Timedelta(minutes=5)).to_pydatetime()
    assert sessions.auto_evaluate_available("2026-11-06") == 1
    assert sessions.list_session_predictions("2026-11-06","morning")[0]["actual"]["actual_close"] == 102.5


def test_session_candles_are_genuine_5m_aggregates(tmp_path, monkeypatch):
    from src.forecasts import legacy_sessions as sessions
    start, _end = interval_schedule("2026-11-06", "morning")[0]
    bars = pd.DataFrame([
        {"timestamp":pd.Timestamp(start).tz_convert("UTC"),"open":100,"high":102,"low":99,"close":101,"volume":10},
        {"timestamp":(pd.Timestamp(start)+pd.Timedelta(minutes=5)).tz_convert("UTC"),"open":101,"high":104,"low":100,"close":103,"volume":12},
    ])
    monkeypatch.setattr(sessions, "latest_processed_path", lambda: tmp_path / "fake.csv")
    monkeypatch.setattr(sessions, "load_local_candles", lambda _path: bars.copy())
    result = sessions.session_actual_candles("2026-11-06","morning")
    assert result[0] == {"time":start.isoformat(),"interval_end":(start+pd.Timedelta(minutes=10)).isoformat(),"open":100.0,"high":104.0,"low":99.0,"close":103.0,"volume":22.0}
