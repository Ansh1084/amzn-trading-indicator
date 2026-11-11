"""Validated, completed-candle analysis orchestration."""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from backend.app.services.analysis.config import ANALYSIS_VERSION, BAR_MINUTES, MAX_LIVE_STALENESS_MINUTES, MIN_WARMUP_BARS, TIMEZONE
from backend.app.services.analysis.technical_analysis import calculate_indicators, analyze_latest
from backend.app.services.analysis.market_structure import analyze_structure
from backend.app.services.analysis.regime_detection import detect_regime
from backend.app.services.analysis.ml_evidence import latest_ml_evidence
from backend.app.services.analysis.decision_engine import make_decision
from backend.app.services.analysis.persistence import save_signal, active_signals
from src.forecasts.calendar import is_trading_day, previous_trading_day, session_close, market_timestamp

ET=ZoneInfo(TIMEZONE)

def _expected_latest_start(now: datetime) -> datetime:
    local=now.astimezone(ET)
    trading_today=is_trading_day(local.date())
    open_at=local.replace(hour=9,minute=30,second=0,microsecond=0)
    close_time=session_close(local.date()) if trading_today else None
    close_at=local.replace(hour=close_time.hour,minute=close_time.minute,second=0,microsecond=0) if close_time else None
    if trading_today and local>=open_at and local<close_at:
        elapsed=int((local-open_at).total_seconds()//300)
        candidate=open_at+timedelta(minutes=max(0,elapsed-1)*BAR_MINUTES)
        if candidate+timedelta(minutes=BAR_MINUTES)<=local:
            return candidate
    if trading_today and close_at and local>=close_at:
        day=local.date()
    elif trading_today:
        day=previous_trading_day(local.date())
    else:
        day=previous_trading_day(local.date()+timedelta(days=1))
    close=session_close(day)
    return market_timestamp(day,(datetime.combine(day,close)-timedelta(minutes=BAR_MINUTES)).strftime("%H:%M"))

def validate_completed_candles(candles: pd.DataFrame, as_of: datetime | None=None, stale_check=True) -> pd.DataFrame:
    if candles.empty: raise ValueError("No market candles are available")
    frame=candles.copy()
    required={"timestamp","open","high","low","close","volume"}
    if required-set(frame.columns): raise ValueError(f"Missing candle columns: {sorted(required-set(frame.columns))}")
    frame["timestamp"]=pd.to_datetime(frame.timestamp,utc=True,errors="raise")
    if frame.timestamp.duplicated().any(): raise ValueError("Duplicate candle timestamps detected")
    frame=frame.sort_values("timestamp").reset_index(drop=True)
    now=pd.Timestamp(as_of or datetime.now(timezone.utc))
    if now.tzinfo is None: raise ValueError("as_of must be timezone-aware")
    now=now.tz_convert("UTC")
    frame=frame.loc[frame.timestamp+pd.Timedelta(minutes=BAR_MINUTES)<=now].copy()
    if frame.empty: raise ValueError("No completed candles are available at the requested cutoff")
    local=frame.timestamp.dt.tz_convert(ET)
    valid_session=[]
    for ts in local:
        in_rth=(ts.hour>9 or ts.hour==9 and ts.minute>=30) and (ts.hour<16)
        is_valid_day=ts.weekday()<5 and is_trading_day(ts.date())
        closes_by_session=(ts+timedelta(minutes=BAR_MINUTES)).strftime("%H:%M")<=session_close(ts.date()).strftime("%H:%M")
        valid_session.append(bool(is_valid_day and in_rth and closes_by_session))
    excluded=int(len(frame)-sum(valid_session))
    frame=frame.loc[valid_session].reset_index(drop=True)
    if frame.empty: raise ValueError("No regular-session candles remain after calendar filtering")
    local=frame.timestamp.dt.tz_convert(ET)
    for ts in local:
        if ts.minute%5: raise ValueError(f"Candle timestamp is off the 5-minute grid: {ts.isoformat()}")
    if stale_check:
        local_now=now.to_pydatetime().astimezone(ET)
        last=local.iloc[-1].to_pydatetime()
        expected=_expected_latest_start(local_now)
        lag=(expected-last).total_seconds()/60
        if lag>MAX_LIVE_STALENESS_MINUTES:
            raise ValueError(f"Latest completed candle is stale by {lag:.1f} minutes; expected a bar at {expected.isoformat()}")
    vals=frame[["open","high","low","close","volume"]].apply(pd.to_numeric,errors="coerce")
    if vals.isna().any().any() or not np.isfinite(vals.to_numpy(float)).all(): raise ValueError("Candle OHLCV values must be finite")
    if (vals.volume<0).any() or (vals.low<=0).any() or (vals.high<vals[["open","close"]].max(axis=1)).any() or (vals.low>vals[["open","close"]].min(axis=1)).any():
        raise ValueError("Invalid OHLCV candle values")
    frame[list(vals.columns)]=vals
    # Missing bars are surfaced as a warning; calculations do not bridge the gap invisibly.
    local_dates=frame.timestamp.dt.tz_convert(ET).dt.date
    gaps=[]
    for index in range(1,len(frame)):
        previous=frame.timestamp.iloc[index-1]; current=frame.timestamp.iloc[index]
        if local_dates.iloc[index]==local_dates.iloc[index-1] and current-previous>pd.Timedelta(minutes=BAR_MINUTES):
            gaps.append({"from":previous.isoformat(),"to":current.isoformat(),"minutes":(current-previous).total_seconds()/60})
    frame.attrs["gaps"]=gaps
    frame.attrs["excluded_calendar_rows"]=excluded
    return frame

def analyze_candles(candles: pd.DataFrame, as_of: datetime | None=None, source="local", persist=True, db_path=None, ml_evidence=None) -> dict:
    frame=validate_completed_candles(candles,as_of)
    indicators=calculate_indicators(frame)
    latest=analyze_latest(indicators)
    latest_stamp=pd.Timestamp(frame.timestamp.iloc[-1]).isoformat()
    if len(frame)<MIN_WARMUP_BARS or latest.get("status")!="READY":
        decision={"classification":"WAIT FOR CONFIRMATION","directional_bias":"NEUTRAL","final_decision_score":0.0,
                  "technical_score":0.0,"reasons":[f"Insufficient completed history; {len(frame)} bars available, {MIN_WARMUP_BARS} required."],
                  "risk_plan":{"valid":False,"reason":"Indicator warm-up is incomplete."},"score_type":"heuristic_weighted_evidence_not_probability"}
        structure={"status":"INSUFFICIENT_DATA","pivots":[],"support_zones":[],"resistance_zones":[],"nearest_support":None,"nearest_resistance":None}
        regime={"classification":"TRANSITIONAL","status":"INSUFFICIENT_DATA","evidence":[]}
        ml=ml_evidence or {"available":False,"reason":"Insufficient warm-up for decision."}
    else:
        if persist:
            # Resolve prior setups against this completed, cutoff-bounded snapshot
            # before deciding whether another hypothetical position may be opened.
            from backend.app.services.analysis.outcomes import resolve_signals
            resolve_signals(frame, as_of=as_of, db_path=db_path)
        atr=float(indicators.atr_14.iloc[-1])
        structure=analyze_structure(indicators,atr)
        weak_break=(structure.get("breakout",{}).get("status")=="POTENTIAL_BREAKOUT" and not structure.get("breakout",{}).get("volume_confirmed")) or (structure.get("breakdown",{}).get("status")=="POTENTIAL_BREAKDOWN" and not structure.get("breakdown",{}).get("volume_confirmed"))
        latest["weak_volume_breakout"]=bool(weak_break)
        regime=detect_regime(indicators)
        if ml_evidence is None:
            try:
                ml=latest_ml_evidence(frame)
            except Exception as exc:
                ml={"available":False,"reason":f"Production model evidence unavailable: {type(exc).__name__}: {exc}"}
        else: ml=ml_evidence
        active=active_signals(db_path)
        latest_close=float(frame.close.iloc[-1])
        active_direction=None
        for record in reversed(active):
            existing=record["payload"].get("risk_plan",{})
            direction=existing.get("direction")
            if not existing.get("valid") or direction not in {"BUY","SELL"}: continue
            stop=float(existing["stop_loss"]); target=float(existing["target_2"])
            still_valid=(latest_close>stop and latest_close<target) if direction=="BUY" else (latest_close<stop and latest_close>target)
            if still_valid:
                active_direction=direction
                break
        decision=make_decision(indicators,structure,regime,ml,active_direction)
    generated_at=as_of or datetime.now(timezone.utc)
    if generated_at.tzinfo is None: raise ValueError("analysis generation timestamp must be timezone-aware")
    generated=pd.Timestamp(generated_at).tz_convert("UTC").isoformat()
    payload={"analysis_version":ANALYSIS_VERSION,"symbol":"AMZN","source":source,"generated_at":generated,
      "as_of":pd.Timestamp(generated_at).tz_convert("UTC").isoformat(),"candle_timestamp":latest_stamp,
      "candle_close_timestamp":(pd.Timestamp(frame.timestamp.iloc[-1])+pd.Timedelta(minutes=BAR_MINUTES)).isoformat(),
      "market_timezone":TIMEZONE,"bars_used":len(frame),"history_start":pd.Timestamp(frame.timestamp.iloc[0]).isoformat(),
      "data_gaps":frame.attrs.get("gaps",[]),"excluded_calendar_rows":frame.attrs.get("excluded_calendar_rows",0),"indicators":latest,"structure":structure,"market_regime":regime,
      "ml_evidence":ml,"hypothetical_only":True,**decision}
    stored=save_signal(payload,db_path) if persist else None
    return {"analysis":payload,"signal_record":stored}
