"""Idempotent hypothetical setup resolution from completed genuine candles."""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pandas as pd
from backend.app.services.analysis.config import BAR_MINUTES, MAX_HOLD_MINUTES, ROUND_TRIP_COST_BPS, TIMEZONE
from backend.app.services.analysis.persistence import active_signals, append_event, set_signal_status, signal_events
from backend.app.services.analysis.service import validate_completed_candles
from src.forecasts.calendar import session_close, market_timestamp

def resolve_signals(candles: pd.DataFrame, as_of: datetime | None=None, db_path=None) -> dict:
    frame=validate_completed_candles(candles,as_of,stale_check=False)
    resolved=[]; pending=[]
    for record in active_signals(db_path):
        signal=record["payload"]; plan=signal.get("risk_plan",{})
        if not plan.get("valid"):
            pending.append(record["signal_id"]); continue
        stamp=pd.Timestamp(record["candle_timestamp"])
        future=frame.loc[frame.timestamp>stamp]
        if future.empty:
            pending.append(record["signal_id"]); continue
        direction=plan["direction"]
        target1_already=any(e["event_type"]=="TARGET_1_HIT" for e in signal_events(record["signal_id"],db_path))
        reference=float(plan["trigger_price"]); entry=None; stop=float(plan["stop_loss"]); target1=float(plan["target_1"]); target2=float(plan["target_2"])
        entry_zone=plan.get("entry_zone",{"lower":reference,"upper":reference})
        prior=stamp
        entry_timestamp=None
        closed=False
        for row in future.itertuples(index=False):
            ts=pd.Timestamp(row.timestamp)
            # Do not jump over missing bars; allow a normal overnight boundary.
            previous_local=prior.to_pydatetime().astimezone(ZoneInfo(TIMEZONE))
            current_local=ts.to_pydatetime().astimezone(ZoneInfo(TIMEZONE))
            if current_local.date()==previous_local.date() and ts-prior>pd.Timedelta(minutes=BAR_MINUTES):
                pending.append(record["signal_id"]); closed=True; break
            if current_local.date()!=previous_local.date() and entry is not None:
                # Intraday setups do not cross sessions. If the prior session's
                # final completed bar was missing, leave the result pending rather
                # than using next-session prices as an invented exit.
                pending.append(record["signal_id"]); closed=True; break
            if entry is None:
                if current_local.date()!=previous_local.date():
                    details={"status":"EXPIRED","reason":"No next same-session candle was available for entry.",
                             "signal_candle_timestamp":stamp.isoformat(),"hypothetical_only":True,
                             "entry_assumption":"next same-session candle open"}
                    if set_signal_status(record["signal_id"],"EXPIRED",details,None,None,db_path): resolved.append(record["signal_id"])
                    closed=True; break
                candidate=float(row.open)
                if not (float(entry_zone["lower"])<=candidate<=float(entry_zone["upper"])):
                    details={"status":"EXPIRED","reason":"Next candle open was outside the planned entry zone; no hypothetical fill assumed.",
                             "signal_candle_timestamp":stamp.isoformat(),"entry_candle_timestamp":ts.isoformat(),
                             "entry_open":candidate,"entry_zone":entry_zone,"hypothetical_only":True}
                    if set_signal_status(record["signal_id"],"EXPIRED",details,None,None,db_path): resolved.append(record["signal_id"])
                    closed=True; break
                entry=candidate; entry_timestamp=ts
            is_long=direction=="BUY"
            hit_stop=(row.low<=stop) if is_long else (row.high>=stop)
            hit_t1=(row.high>=target1) if is_long else (row.low<=target1)
            hit_t2=(row.high>=target2) if is_long else (row.low<=target2)
            event=None; exit_price=None; close_reason=None
            if hit_stop: event,exit_price="STOP_LOSS_HIT",stop
            elif hit_t2: event,exit_price="TARGET_2_HIT",target2
            if hit_t1 and not hit_stop and not target1_already:
                append_event(record["signal_id"],"TARGET_1_HIT",{"timestamp":ts.isoformat(),"level":target1,"note":"Informational milestone; position is not partially exited."},db_path)
                target1_already=True
            local_end=ts.to_pydatetime().astimezone(ZoneInfo(TIMEZONE))+timedelta(minutes=BAR_MINUTES)
            close_dt=market_timestamp(current_local.date(),session_close(current_local.date()).strftime("%H:%M"))
            age=(ts-entry_timestamp).total_seconds()/60
            if event is None and local_end>=close_dt:
                event,exit_price,close_reason="CLOSED",float(row.close),"SESSION_END"
            elif event is None and age>=MAX_HOLD_MINUTES:
                event,exit_price,close_reason="CLOSED",float(row.close),"TIME_LIMIT"
            if event:
                raw=((exit_price-entry)/entry*100)*(1 if is_long else -1)
                net=raw-ROUND_TRIP_COST_BPS/100
                outcome={"status":event,"exit_timestamp":ts.isoformat(),"exit_price":float(exit_price),"entry_price":entry,
                         "direction":direction,"gross_return_pct":raw,"assumed_cost_bps_round_trip":ROUND_TRIP_COST_BPS,
                         "signal_reference_close":reference,"entry_assumption":"next same-session candle open within planned entry zone",
                         "net_return_pct_assumption":net,"hypothetical_only":True,
                         "close_reason":close_reason,
                         "barrier_order":"stop_first_if_stop_and_target2_touched_in_same_5m_bar"}
                if set_signal_status(record["signal_id"],event,outcome,raw,net,db_path): resolved.append(record["signal_id"])
                closed=True; break
            prior=ts
        if not closed: pending.append(record["signal_id"])
    return {"resolved_signal_ids":resolved,"pending_signal_ids":sorted(set(pending)),"evaluated_through":pd.Timestamp(frame.timestamp.iloc[-1]).isoformat(),"hypothetical_only":True}
