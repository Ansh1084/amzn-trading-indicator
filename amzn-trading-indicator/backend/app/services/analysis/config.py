"""Analysis-engine settings loaded from configs/analysis_engine.json."""
from __future__ import annotations
import json
from pathlib import Path

_PATH=Path(__file__).resolve().parents[4]/"configs"/"analysis_engine.json"
_DATA=json.loads(_PATH.read_text(encoding="utf-8")) if _PATH.exists() else {}
ANALYSIS_VERSION=_DATA.get("analysis_version","amzn-analysis-v1")
TIMEZONE=_DATA.get("timezone","America/New_York")
BAR_MINUTES=int(_DATA.get("bar_minutes",5))
MAX_LIVE_STALENESS_MINUTES=int(_DATA.get("max_live_staleness_minutes",15))
MIN_WARMUP_BARS=int(_DATA.get("minimum_warmup_bars",60))
SWING_LEFT=int(_DATA.get("swing_left_bars",2))
SWING_RIGHT=int(_DATA.get("swing_right_bars",2))
ZONE_TOLERANCE_ATR=float(_DATA.get("zone_tolerance_atr",.35))
MAX_LEVEL_SESSIONS=int(_DATA.get("maximum_level_sessions",15))
MIN_TOUCH_SESSIONS=int(_DATA.get("minimum_touch_sessions",2))
BREAKOUT_PENETRATION_ATR=float(_DATA.get("breakout_penetration_atr",.15))
BREAKOUT_RELATIVE_VOLUME=float(_DATA.get("breakout_relative_volume",1.2))
RETEST_DISTANCE_ATR=float(_DATA.get("retest_distance_atr",.3))
RETEST_LOOKBACK_BARS=int(_DATA.get("retest_lookback_bars",12))
WEIGHTS={k:float(v) for k,v in _DATA.get("decision_weights",{"trend":.25,"momentum":.2,"volume":.1,"structure":.25,"ml":.1,"price_action":.1}).items()}
DECISION_THRESHOLD=float(_DATA.get("decision_threshold",.35))
STRONG_DECISION_THRESHOLD=float(_DATA.get("strong_decision_threshold",.6))
MIN_TARGET_RISK_REWARD=float(_DATA.get("minimum_target2_risk_reward",1.5))
ENTRY_ZONE_ATR=float(_DATA.get("entry_zone_atr",.1))
STOP_BUFFER_ATR=float(_DATA.get("stop_buffer_atr",.25))
TARGET1_R_MULTIPLE=float(_DATA.get("target1_r_multiple",1.0))
TARGET2_R_MULTIPLE=float(_DATA.get("target2_r_multiple",2.0))
ROUND_TRIP_COST_BPS=float(_DATA.get("round_trip_cost_bps",10.0))
MAX_HOLD_MINUTES=int(_DATA.get("maximum_hold_minutes",180))

if abs(sum(WEIGHTS.values())-1.0)>1e-9:
    raise ValueError("analysis decision weights in analysis_engine.json must sum to 1.0")
