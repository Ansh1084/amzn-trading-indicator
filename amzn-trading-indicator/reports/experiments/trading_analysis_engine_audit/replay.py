"""Reproducible, chronological, local-data replay of the analysis decision engine.

The full indicators and legacy model feature rows are computed in batch because
each row is causal. Every decision only receives the prefix's current row,
confirmed pivots whose confirmation timestamp has completed, and structure from
the then-available trailing sessions. No production database/artifact is written.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from backend.app.services.analysis.config import (ANALYSIS_VERSION, BAR_MINUTES, MAX_LEVEL_SESSIONS,
    MAX_HOLD_MINUTES, MIN_WARMUP_BARS, ROUND_TRIP_COST_BPS)
from backend.app.services.analysis.technical_analysis import calculate_indicators, analyze_latest
from backend.app.services.analysis.market_structure import analyze_structure, confirmed_pivots
from backend.app.services.analysis.regime_detection import calculate_regime_series, detect_regime
from backend.app.services.analysis.decision_engine import make_decision
from backend.app.services.analysis.risk_management import build_risk_plan
from backend.app.services.analysis.service import validate_completed_candles
from src.data.ingestion import latest_processed_path
from src.features.legacy_engineering import FEATURE_COLUMNS, generate_features
from src.forecasts.legacy_sessions import MODEL_PATH, MODEL_VERSION, _artifact_sha256, load_legacy_model
from src.forecasts.calendar import session_close
from src.forecasts.store import connect

ET=ZoneInfo("America/New_York")
OUT=Path(__file__).resolve().parent

def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def finite_rows(features: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return np.isfinite(features[columns].to_numpy(dtype=float)).all(axis=1)

def _risk(pos: dict, row, ts: pd.Timestamp, prev_ts: pd.Timestamp | None) -> tuple[dict | None, bool]:
    """Advance one independent, one-position-at-a-time hypothetical strategy."""
    if pos is None: return None,False
    candle_local=ts.to_pydatetime().astimezone(ET)
    sig_local=pd.Timestamp(pos["candle_timestamp"]).to_pydatetime().astimezone(ET)
    if pos["entry"] is None:
        if candle_local.date()!=sig_local.date():
            pos["status"]="EXPIRED"; pos["reason"]="No next same-session candle was available for entry."; return pos,True
        if prev_ts is not None and candle_local.date()==prev_ts.to_pydatetime().astimezone(ET).date() and ts-prev_ts>pd.Timedelta(minutes=BAR_MINUTES):
            pos["gap_blocked"]=True
        if pos.get("gap_blocked"):
            return pos,False
        zone=pos["plan"].get("entry_zone",{})
        opened=float(row.open)
        if not (float(zone.get("lower",pos["reference"]))<=opened<=float(zone.get("upper",pos["reference"]))):
            pos["status"]="EXPIRED"; pos["reason"]="Next candle open was outside the planned entry zone."; pos["entry_open"] = opened; return pos,True
        pos["entry"]=opened; pos["entry_timestamp"]=ts.isoformat()
    elif candle_local.date()!=sig_local.date():
        pos["gap_blocked"]=True
        return pos,False
    if prev_ts is not None and candle_local.date()==prev_ts.to_pydatetime().astimezone(ET).date() and ts-prev_ts>pd.Timedelta(minutes=BAR_MINUTES):
        pos["gap_blocked"]=True
    if pos.get("gap_blocked"):
        return pos,False
    long=pos["direction"]=="BUY"; stop=float(pos["plan"]["stop_loss"]); t1=float(pos["plan"]["target_1"]); t2=float(pos["plan"]["target_2"])
    hit_stop=(float(row.low)<=stop) if long else (float(row.high)>=stop)
    hit_t1=(float(row.high)>=t1) if long else (float(row.low)<=t1)
    hit_t2=(float(row.high)>=t2) if long else (float(row.low)<=t2)
    if hit_stop: pos["status"]="STOP_LOSS_HIT"; exit_price=stop
    elif hit_t2: pos["status"]="TARGET_2_HIT"; exit_price=t2
    else: exit_price=None
    if hit_t1 and not hit_stop and not pos["target1_hit"]:
        pos["target1_hit"]=True; pos["target1_timestamp"]=ts.isoformat()
    local_end=candle_local+timedelta(minutes=BAR_MINUTES)
    close_at=candle_local.replace(hour=session_close(candle_local.date()).hour,minute=session_close(candle_local.date()).minute,second=0,microsecond=0)
    age=(ts-pd.Timestamp(pos["entry_timestamp"])).total_seconds()/60
    if pos["status"]=="ACTIVE" and local_end>=close_at:
        pos["status"]="CLOSED"; pos["reason"]="SESSION_END"; exit_price=float(row.close)
    elif pos["status"]=="ACTIVE" and age>=MAX_HOLD_MINUTES:
        pos["status"]="CLOSED"; pos["reason"]="TIME_LIMIT"; exit_price=float(row.close)
    if pos["status"] in {"STOP_LOSS_HIT","TARGET_2_HIT","CLOSED"}:
        pos["exit_timestamp"]=ts.isoformat(); pos["exit_price"]=float(exit_price)
        gross=(float(exit_price)-float(pos["entry"]))/float(pos["entry"])*100*(1 if long else -1)
        pos["gross_return_pct"]=gross; pos["net_return_pct"]=gross-ROUND_TRIP_COST_BPS/100
        pos["barrier_order"]="stop_first_if_stop_and_target2_touched_in_same_5m_bar"
        return pos,True
    return pos,False

def _metrics(trades: list[dict]) -> dict:
    executed=[t for t in trades if t.get("gross_return_pct") is not None]
    if not executed:
        return {"trades":0,"gross_return_pct_sum":0.0,"net_return_pct_sum":0.0,"max_drawdown_pct_points":0.0,
            "profit_factor_net":None,"profit_factor_gross":None,"average_net_return_pct":None,"average_win_pct":None,
            "average_loss_pct":None,"planned_target2_rr_mean":None,"planned_target2_rr_median":None,
            "planned_target2_rr_min":None,"planned_target2_rr_max":None}
    gross=np.array([float(t["gross_return_pct"]) for t in executed]); net=np.array([float(t["net_return_pct"]) for t in executed])
    cumulative=np.cumsum(net); high_water=np.maximum.accumulate(np.r_[0.0,cumulative])[1:]
    drawdown=high_water-cumulative
    wins=net[net>0]; losses=net[net<0]
    loss_abs=abs(float(losses.sum()))
    return {"trades":len(executed),"gross_return_pct_sum":float(gross.sum()),"net_return_pct_sum":float(net.sum()),
        "max_drawdown_pct_points":float(drawdown.max()) if len(drawdown) else 0.0,
        "profit_factor_net":float(wins.sum()/loss_abs) if loss_abs else None,
        "profit_factor_gross":float(gross[gross>0].sum()/abs(float(gross[gross<0].sum()))) if np.any(gross<0) else None,
        "average_net_return_pct":float(net.mean()),"average_win_pct":float(wins.mean()) if len(wins) else None,
        "average_loss_pct":float(losses.mean()) if len(losses) else None,
        "planned_target2_rr_mean":float(np.mean([t["rr_target2"] for t in executed if t.get("rr_target2") is not None])) if any(t.get("rr_target2") is not None for t in executed) else None,
        "planned_target2_rr_median":float(np.median([t["rr_target2"] for t in executed if t.get("rr_target2") is not None])) if any(t.get("rr_target2") is not None for t in executed) else None,
        "planned_target2_rr_min":float(np.min([t["rr_target2"] for t in executed if t.get("rr_target2") is not None])) if any(t.get("rr_target2") is not None for t in executed) else None,
        "planned_target2_rr_max":float(np.max([t["rr_target2"] for t in executed if t.get("rr_target2") is not None])) if any(t.get("rr_target2") is not None for t in executed) else None}

def _position_record(strategy: str, candle: pd.Series, decision: dict, regime: str, plan: dict, sid: str | None=None) -> dict:
    return {"signal_id":sid or str(uuid4()),"strategy":strategy,"candle_timestamp":candle.timestamp.isoformat(),
        "generation_timestamp":(candle.timestamp+pd.Timedelta(minutes=BAR_MINUTES)).isoformat(),
        "direction":plan["direction"],"reference":float(candle.close),"plan":plan,"regime":regime,
        "decision":decision.get("classification",strategy),"reason":json.dumps(decision.get("reasons",[])),
        "rr_target2":plan.get("target_2_risk_reward"),"entry":None,"entry_timestamp":None,
        "target1_hit":False,"status":"ACTIVE","gap_blocked":False}

def run(source: Path, out: Path, start_date: str | None=None, end_date: str | None=None) -> dict:
    t0=time.perf_counter(); raw=pd.read_csv(source)
    if "timestamp" not in raw: raise ValueError("Market snapshot has no timestamp field")
    raw["timestamp"]=pd.to_datetime(raw.timestamp,utc=True,errors="raise")
    max_ts=raw.timestamp.max()
    data=validate_completed_candles(raw,max_ts.to_pydatetime()+timedelta(minutes=BAR_MINUTES),stale_check=False)
    local=data.timestamp.dt.tz_convert(ET)
    if start_date: data=data.loc[local.dt.date>=pd.Timestamp(start_date).date()].reset_index(drop=True); local=data.timestamp.dt.tz_convert(ET)
    if end_date: data=data.loc[local.dt.date<=pd.Timestamp(end_date).date()].reset_index(drop=True); local=data.timestamp.dt.tz_convert(ET)
    if data.empty: raise ValueError("No candles remain inside the replay date range")
    print(f"Preparing {len(data):,} completed RTH candles from {local.iloc[0].date()} through {local.iloc[-1].date()}.",flush=True)
    indicators=calculate_indicators(data)
    regime_rows=calculate_regime_series(indicators)
    full_pivots=confirmed_pivots(data)
    pivot_times=[pd.Timestamp(p["confirmed_at"]) for p in full_pivots]
    pivot_ix=0; recent_pivots=[]; highs=deque(maxlen=2); lows=deque(maxlen=2)
    unique_days=list(dict.fromkeys(local.dt.date.tolist()))
    day_rank={day:ix for ix,day in enumerate(unique_days)}
    first_index={}
    for ix,day in enumerate(local.dt.date): first_index.setdefault(day,ix)
    day_index=local.dt.date.tolist()
    features=generate_features(data.copy(),reset_daily=False)
    model,manifest,classes=load_legacy_model(MODEL_PATH)
    if manifest!=FEATURE_COLUMNS or len(manifest)!=40: raise ValueError("Production model feature manifest mismatch")
    valid_ml=finite_rows(features,manifest)
    ml_probability=np.full(len(data),np.nan); ml_direction=np.full(len(data),None,dtype=object)
    if valid_ml.any():
        X=features.loc[valid_ml,manifest].astype(float)
        probabilities=model.predict_proba(X)
        predictions=model.predict(X)
        up_col=classes.index(1.0)
        ml_probability[valid_ml]=probabilities[:,up_col]
        ml_direction[valid_ml]=np.where(predictions==1,"UP","DOWN")
    ml_sha=_artifact_sha256(MODEL_PATH)
    data_hash=digest(source); config_path=ROOT/"configs"/"analysis_engine.json"

    # Prove batch calculations match their causal prefix calculations at fixed samples.
    sample_ix=sorted(set([min(len(data)-1,k) for k in [max(59,int(len(data)*.25)),max(60,int(len(data)*.5)),max(61,int(len(data)*.75)),len(data)-1]]))
    parity=[]
    for ix in sample_ix:
        prefix_features=generate_features(data.iloc[:ix+1].copy(),reset_daily=False)
        f_equal=np.allclose(features.loc[ix,manifest].to_numpy(float),prefix_features.iloc[-1][manifest].to_numpy(float),equal_nan=True,rtol=1e-10,atol=1e-10)
        prefix_ind=calculate_indicators(data.iloc[:ix+1])
        numeric=[c for c in indicators.columns if pd.api.types.is_numeric_dtype(indicators[c])]
        i_equal=np.allclose(indicators.loc[ix,numeric].to_numpy(float),prefix_ind.iloc[-1][numeric].to_numpy(float),equal_nan=True,rtol=1e-10,atol=1e-10)
        t=pd.Timestamp(data.timestamp.iloc[ix])
        known=[p for p,pt in zip(full_pivots,pivot_times) if pt<=t]
        s1=analyze_structure(indicators.iloc[:ix+1],float(indicators.atr_14.iloc[ix]))
        s2=analyze_structure(indicators.iloc[:ix+1],float(indicators.atr_14.iloc[ix]),known_pivots=known)
        s_equal=(s1["sequence"]==s2["sequence"] and s1["breakout"]["status"]==s2["breakout"]["status"] and s1["breakdown"]["status"]==s2["breakdown"]["status"] and s1["nearest_support"]==s2["nearest_support"] and s1["nearest_resistance"]==s2["nearest_resistance"])
        parity.append({"index":ix,"timestamp":t.isoformat(),"legacy_feature_prefix_equal":bool(f_equal),"indicator_prefix_equal":bool(i_equal),"structure_prefix_equal":bool(s_equal),"future_pivots_excluded":all(pt<=t for pt in [pd.Timestamp(p["confirmed_at"]) for p in known])})
    if not all(all(v for k,v in p.items() if k not in {"index","timestamp"}) for p in parity): raise AssertionError(f"Causal prefix parity failed: {parity}")

    # Build a second isolated replay DB. One active position per strategy; the
    # three baselines and combined engine are separate hypothetical portfolios.
    out.mkdir(parents=True,exist_ok=True); db_path=out/"replay.sqlite3"
    if db_path.exists(): db_path.unlink()
    decisions=Counter(); signal_counts_by_regime=Counter(); valid_count=0; insufficient=0; raw_excluded=int(data.attrs.get("excluded_calendar_rows",0))
    record_rows=[]; event_rows=[]; all_trades={name:[] for name in ["combined","ema_trend","rsi_14","ml_only"]}
    positions={name:None for name in all_trades}; prev_ts=None; confirmed_cursor=0
    kept_recent=[]; high_pivots=deque(maxlen=2); low_pivots=deque(maxlen=2)
    first_by_day=first_index
    for ix in range(len(data)):
        candle=data.iloc[ix]; ts=pd.Timestamp(candle.timestamp); current_day=local.iloc[ix].date()
        # Feed only pivots whose right-side confirmation candle is completed now.
        while confirmed_cursor<len(full_pivots) and pivot_times[confirmed_cursor]<=ts:
            p=full_pivots[confirmed_cursor]
            (high_pivots if p["type"]=="HIGH" else low_pivots).append(p)
            kept_recent.append(p); confirmed_cursor+=1
        cutoff_day=unique_days[max(0,day_rank[current_day]-MAX_LEVEL_SESSIONS+1)]
        kept_recent=[p for p in kept_recent if pd.Timestamp(p["confirmed_at"]).tz_convert(ET).date()>=cutoff_day]
        pivots=kept_recent+list(high_pivots)+list(low_pivots)
        # De-duplicate historical last-two pivots also present in the 40-session set.
        pivots=list({(p["type"],p["index"]):p for p in pivots}.values())
        pivots.sort(key=lambda p:(p["index"],p["type"]))
        window_start=first_by_day[cutoff_day]
        history=indicators.iloc[window_start:ix+1]

        # Resolve the prior setup at this completed candle before considering a
        # new signal. A position that stays unresolved blocks overlap.
        for name,pos in list(positions.items()):
            if pos is None: continue
            had_target1=pos["target1_hit"]
            result,terminal=_risk(pos,candle,ts,prev_ts)
            if name=="combined" and not had_target1 and result.get("target1_hit"):
                event_rows.append((result["signal_id"],result["target1_timestamp"],"TARGET_1_HIT",{"level":result["plan"]["target_1"],"informational_only":True}))
            if terminal:
                all_trades[name].append(result)
                if name=="combined":
                    event_rows.append((result["signal_id"],result.get("exit_timestamp",result["generation_timestamp"]),result["status"],result))
                else: positions[name]=None
            if terminal and name=="combined": positions[name]=None

        row=indicators.iloc[[ix]]
        latest=analyze_latest(row)
        ready=(len(indicators.iloc[:ix+1])>=MIN_WARMUP_BARS and latest.get("status")=="READY")
        if ready:
            valid_count+=1
            atr=float(indicators.atr_14.iloc[ix])
            structure=analyze_structure(history,atr,known_pivots=pivots,index_offset=window_start)
            weak=(structure["breakout"]["status"]=="POTENTIAL_BREAKOUT" and not structure["breakout"]["volume_confirmed"]) or (structure["breakdown"]["status"]=="POTENTIAL_BREAKDOWN" and not structure["breakdown"]["volume_confirmed"])
            latest["weak_volume_breakout"]=bool(weak)
            regime=detect_regime(indicators.iloc[:ix+1],computed_series=regime_rows.iloc[:ix+1])
            signal_counts_by_regime[regime["classification"]]+=1
            rawp=float(ml_probability[ix]) if np.isfinite(ml_probability[ix]) else None
            mld={"available":rawp is not None,"direction":ml_direction[ix],"raw_probability_up":rawp,
                 "raw_directional_score":(rawp if ml_direction[ix]=="UP" else 1-rawp) if rawp is not None else None,
                 "score_type":"uncalibrated_raw_model_score_not_confidence" if rawp is not None else "unavailable",
                 "model_name":type(model).__name__,"model_version":MODEL_VERSION,"artifact_sha256":ml_sha,
                 "input_feature_count":len(manifest),"feature_manifest":manifest,"candle_timestamp":ts.isoformat(),
                 "generated_at":(ts+pd.Timedelta(minutes=BAR_MINUTES)).isoformat()}
            active_dir=positions["combined"]["direction"] if positions["combined"] is not None else None
            decision=make_decision(row,structure,regime,mld,active_dir)
            label=decision["classification"]
            decisions[label]+=1
            generation=(ts+pd.Timedelta(minutes=BAR_MINUTES)).isoformat()
            sid=str(uuid4())
            payload={"analysis_version":ANALYSIS_VERSION,"symbol":"AMZN","source":"chronological_local_replay",
                "generated_at":generation,"as_of":generation,"candle_timestamp":ts.isoformat(),
                "candle_close_timestamp":generation,"market_timezone":"America/New_York","bars_used":ix+1,
                "indicators":latest,"structure":{"nearest_support":structure["nearest_support"],"nearest_resistance":structure["nearest_resistance"],
                  "sequence":structure["sequence"],"breakout":structure["breakout"],"breakdown":structure["breakdown"]},
                "market_regime":regime,"ml_evidence":mld,"hypothetical_only":True,**decision}
            payload["ml_evidence"].pop("feature_manifest",None)
            record_status="ACTIVE" if label in {"BUY","SELL"} else "NOT_ACTIONABLE"
            record_rows.append((sid,ts.isoformat(),generation,ANALYSIS_VERSION,label,decision["directional_bias"],decision.get("final_decision_score"),json.dumps(payload,default=str),record_status))
            if label in {"BUY","SELL"} and positions["combined"] is None and decision.get("risk_plan",{}).get("valid"):
                positions["combined"]=_position_record("combined",candle,decision,regime["classification"],decision["risk_plan"],sid)
            for name,direction in (("ema_trend","BUY" if latest.get("ema_20") is not None and latest["ema_20"]>latest["ema_50"] else "SELL" if latest.get("ema_20") is not None and latest["ema_20"]<latest["ema_50"] else None),
                                   ("rsi_14","BUY" if latest.get("rsi_14") is not None and latest["rsi_14"]<30 else "SELL" if latest.get("rsi_14") is not None and latest["rsi_14"]>70 else None),
                                   ("ml_only","BUY" if ml_direction[ix]=="UP" else "SELL" if ml_direction[ix]=="DOWN" else None)):
                if direction and positions[name] is None:
                    plan=build_risk_plan(direction,float(candle.close),atr,structure)
                    if plan.get("valid"):
                        baseline_decision={"classification":direction,"reasons":[f"Fixed baseline: {name} rule."],"risk_plan":plan}
                        positions[name]=_position_record(name,candle,baseline_decision,regime["classification"],plan)
        else:
            insufficient+=1
            decisions["INSUFFICIENT_HISTORY"]+=1
            generation=(ts+pd.Timedelta(minutes=BAR_MINUTES)).isoformat()
            sid=str(uuid4())
            payload={"analysis_version":ANALYSIS_VERSION,"symbol":"AMZN","source":"chronological_local_replay",
                "generated_at":generation,"as_of":generation,"candle_timestamp":ts.isoformat(),
                "classification":"WAIT FOR CONFIRMATION","directional_bias":"NEUTRAL","reasons":["Insufficient completed history or required indicator warm-up."],
                "risk_plan":{"valid":False,"reason":"Indicator warm-up incomplete."},"hypothetical_only":True}
            record_rows.append((sid,ts.isoformat(),generation,ANALYSIS_VERSION,"WAIT FOR CONFIRMATION","NEUTRAL",0.0,json.dumps(payload,default=str),"NOT_ACTIONABLE"))
        prev_ts=ts
        if (ix+1)%1000==0: print(f"Replayed {ix+1:,}/{len(data):,} candles",flush=True)

    # Remain-open positions are persisted but have no realized result.
    for name,pos in positions.items():
        if pos is not None:
            pos["status"]="UNRESOLVED"; pos["gross_return_pct"]=None; pos["net_return_pct"]=None
            all_trades[name].append(pos)
    with connect(db_path) as db:
        db.executemany("INSERT INTO analysis_signals(signal_id,candle_timestamp,generated_at,analysis_version,classification,directional_bias,final_score,payload,status) VALUES(?,?,?,?,?,?,?,?,?)",record_rows)
        db.executemany("INSERT INTO analysis_signal_events(signal_id,timestamp,event_type,details) VALUES(?,?,?,?)",
                       [(sid,tstamp,kind,json.dumps(details,default=str)) for sid,tstamp,kind,details in event_rows])
        for trade in all_trades["combined"]:
            if trade["status"]=="UNRESOLVED": continue
            outcome={"status":trade["status"],"reason":trade.get("reason"),"entry_price":trade.get("entry"),
                "entry_timestamp":trade.get("entry_timestamp"),"exit_price":trade.get("exit_price"),
                "exit_timestamp":trade.get("exit_timestamp"),"gross_return_pct":trade.get("gross_return_pct"),
                "net_return_pct_assumption":trade.get("net_return_pct"),"target1_hit":trade.get("target1_hit"),
                "hypothetical_only":True}
            db.execute("UPDATE analysis_signals SET status=?,outcome_payload=?,gross_return_pct=?,net_return_pct=? WHERE signal_id=?",
                (trade["status"],json.dumps(outcome,default=str),trade.get("gross_return_pct"),trade.get("net_return_pct"),trade["signal_id"]))

    distribution=[]
    for classification,count in sorted(decisions.items()):
        distribution.append({"classification":classification,"count":count,"total_candles":len(data),
            "valid_analysis_results":valid_count,"share_of_valid_pct":count/valid_count*100 if valid_count and classification!="INSUFFICIENT_HISTORY" else None})
    pd.DataFrame(distribution).to_csv(out/"signal_distribution.csv",index=False)
    trade_rows=[]
    for strategy,trades in all_trades.items():
        for trade in trades:
            trade_rows.append({"strategy":strategy,"signal_id":trade["signal_id"],"signal_timestamp":trade["candle_timestamp"],
                "generation_timestamp":trade["generation_timestamp"],"direction":trade["direction"],"regime":trade["regime"],
                "decision":trade["decision"],"status":trade["status"],"entry_timestamp":trade.get("entry_timestamp"),
                "entry_price":trade.get("entry"),"entry_open_if_expired":trade.get("entry_open"),"exit_timestamp":trade.get("exit_timestamp"),
                "exit_price":trade.get("exit_price"),"gross_return_pct":trade.get("gross_return_pct"),"net_return_pct":trade.get("net_return_pct"),
                "target1_hit":trade.get("target1_hit",False),"planned_target2_rr":trade.get("rr_target2"),"resolution_reason":trade.get("reason"),
                "barrier_order":trade.get("barrier_order")})
    pd.DataFrame(trade_rows).to_csv(out/"trade_outcomes.csv",index=False)

    regime_rows_out=[]
    regimes=["TRENDING_UP","TRENDING_DOWN","SIDEWAYS","HIGH_VOLATILITY","LOW_VOLATILITY","TRANSITIONAL"]
    for strategy,trades in all_trades.items():
        for reg in regimes:
            selected=[t for t in trades if t["regime"]==reg]
            m=_metrics(selected)
            m.update({"strategy":strategy,"regime":reg,"signal_count":signal_counts_by_regime[reg] if strategy=="combined" else None,
                      "entered_or_closed_setups":len(selected),"target1_hits":sum(bool(t.get("target1_hit")) for t in selected),
                      "target2_hits":sum(t["status"]=="TARGET_2_HIT" for t in selected),"stop_hits":sum(t["status"]=="STOP_LOSS_HIT" for t in selected)})
            regime_rows_out.append(m)
    pd.DataFrame(regime_rows_out).to_csv(out/"regime_performance.csv",index=False)

    overall={name:{**_metrics(trades),"setups":len(trades),"target1_hits":sum(bool(t.get("target1_hit")) for t in trades),
       "target2_hits":sum(t["status"]=="TARGET_2_HIT" for t in trades),"stop_loss_hits":sum(t["status"]=="STOP_LOSS_HIT" for t in trades),
       "expired":sum(t["status"]=="EXPIRED" for t in trades),"unresolved":sum(t["status"]=="UNRESOLVED" for t in trades),
       "invalidated":sum(t["status"]=="STOP_LOSS_HIT" for t in trades)} for name,trades in all_trades.items()}
    actionable=decisions["BUY"]+decisions["SELL"]
    metadata={"analysis_version":ANALYSIS_VERSION,"source_file":source.name,"source_sha256":data_hash,
      "model_version":MODEL_VERSION,"model_class":type(model).__name__,"model_artifact_sha256":ml_sha,
      "feature_count":len(manifest),"feature_manifest":manifest,"class_order":[str(c) for c in classes],
      "config_sha256":digest(config_path),"start_timestamp_utc":data.timestamp.iloc[0].isoformat(),"end_timestamp_utc":data.timestamp.iloc[-1].isoformat(),
      "start_date_et":local.iloc[0].date().isoformat(),"end_date_et":local.iloc[-1].date().isoformat(),
      "raw_rows":len(raw),"completed_valid_rth_candles":len(data),"calendar_rows_excluded":raw_excluded,
      "valid_analysis_results":valid_count,"insufficient_history_rows":insufficient,"signal_distribution":dict(decisions),
      "actionable_signal_count":actionable,"actionable_pct_valid":actionable/valid_count*100 if valid_count else None,
      "baseline_definitions":{"ema_trend":"BUY if EMA20>EMA50, SELL if EMA20<EMA50","rsi_14":"BUY if RSI14<30, SELL if RSI14>70",
        "ml_only":"BUY on production model UP, SELL on DOWN"},
      "execution_assumptions":{"one_position_at_a_time_per_strategy":True,"separate_strategy_ledgers":True,
        "entry":"next same-session 5m candle open only if within the signal's entry zone; otherwise EXPIRED without P&L",
        "exit":"stop/target2 intrabar touch; stop takes priority if both touch; otherwise session end or 180-minute time limit",
        "target1":"recorded milestone only; no partial exit","round_trip_cost_bps":ROUND_TRIP_COST_BPS,
        "position_size":"one share, fixed size; returns summed additively, not compounded","max_drawdown":"peak-to-trough on additive cumulative net percent points"},
      "lookahead_prefix_parity":parity,"replay_seconds":time.perf_counter()-t0,
      "strategies":overall,"no_training_or_tuning":True,"no_live_alpaca_request":True,
      "unresolved_positions_at_end":{name:sum(t["status"]=="UNRESOLVED" for t in trades) for name,trades in all_trades.items()},
      "replay_database":"replay.sqlite3"}
    (out/"replay_metadata.json").write_text(json.dumps(metadata,indent=2,default=str),encoding="utf-8")
    return metadata

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,default=latest_processed_path())
    parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    args=parser.parse_args()
    result=run(args.source,args.out,args.start_date,args.end_date)
    print(json.dumps({"candles":result["completed_valid_rth_candles"],"valid":result["valid_analysis_results"],
      "distribution":result["signal_distribution"],"strategies":result["strategies"],"seconds":round(result["replay_seconds"],1)},indent=2))

if __name__=="__main__": main()
