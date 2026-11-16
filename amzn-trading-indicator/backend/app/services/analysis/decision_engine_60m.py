"""60-Minute Trading Decision Engine for AMZN AI Trading Terminal.

Evaluates market conditions after every completed 5-minute candle to determine
the most appropriate trading decision for the NEXT 60 MINUTES (12 bars).

Decisions:
  - BUY: Strong positive directional evidence, confirmed trend/structure, clear runway, valid R:R >= 1.5.
  - SELL: Strong negative directional evidence, confirmed trend/structure, clear runway, valid R:R >= 1.5.
  - WAIT FOR CONFIRMATION: Strong directional score, but blocked by near resistance/support,
    lacking multi-timeframe confirmation, or awaiting valid risk parameters.
  - HOLD: Mixed, neutral, rangebound, session-close, or opening-discovery conditions (DEFAULT).

Evidence Groups (Normalized -1.0 to +1.0):
  1. Trend (25%): EMA 9/20/50, SMA 20/50, 15M and 1H multi-timeframe trend alignment.
  2. Market Structure (25%): Support/Resistance/Demand/Supply zones, breakouts/breakdowns, prior day levels.
  3. Momentum (20%): RSI 14/7, MACD histogram & acceleration, ROC 5/20.
  4. Volume (10%): Session rolling volume ratio, historical same-slot RVOL, breakout volume confirmation.
  5. VWAP & Price Action (20%): Session VWAP position/distance, candle body/wicks, rejection wicks.

Zero lookahead bias. All evaluations use completed historical 5-minute candles only.
"""
from __future__ import annotations

import logging
from datetime import datetime, time, date, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from backend.app.services.analysis.config import BAR_MINUTES, TIMEZONE
from src.forecasts.calendar import is_trading_day, session_close, market_timestamp, next_trading_day

logger = logging.getLogger(__name__)

ET = ZoneInfo(TIMEZONE)

# ── Configurable Weights & Thresholds ─────────────────────────────────────────
DEFAULT_WEIGHTS: dict[str, float] = {
    "trend": 0.25,
    "structure": 0.25,
    "momentum": 0.20,
    "volume": 0.10,
    "vwap_price_action": 0.20,
}

DECISION_THRESHOLD: float = 0.35
STRONG_DECISION_THRESHOLD: float = 0.60
MIN_TARGET_RISK_REWARD: float = 1.5
PROXIMITY_VETO_ATR: float = 0.35
HORIZON_MINUTES: int = 60
HORIZON_BARS: int = 12

OPENING_CURFEW_ET: time = time(9, 45)      # 09:30-09:45 is opening discovery
CLOSING_CURFEW_ET: time = time(15, 15)      # Within 45m of close, do not issue new 60m trades


def compute_prediction_schedule(
    candle_ts: datetime,
    as_of: datetime | None = None,
    is_stale: bool = False,
) -> dict[str, Any]:
    """Calculate exact next prediction timing and calendar status in America/New_York timezone."""
    if candle_ts.tzinfo is None:
        candle_local = candle_ts.replace(tzinfo=ZoneInfo("UTC")).astimezone(ET)
    else:
        candle_local = candle_ts.astimezone(ET)

    now_et = as_of.astimezone(ET) if as_of else datetime.now(ET)
    candle_date = candle_local.date()
    candle_close_local = candle_local + timedelta(minutes=BAR_MINUTES)

    # 1. Evaluate market session state
    is_trading = is_trading_day(now_et.date())
    sess_close_time = session_close(now_et.date()) if is_trading else time(16, 0)
    sess_open_dt = market_timestamp(now_et.date(), "09:30") if is_trading else None
    sess_close_dt = market_timestamp(now_et.date(), sess_close_time.strftime("%H:%M")) if is_trading else None

    if not is_trading:
        market_state = "CLOSED"
        is_market_open = False
    elif sess_open_dt and now_et < sess_open_dt:
        market_state = "PRE_MARKET"
        is_market_open = False
    elif sess_close_dt and now_et >= sess_close_dt:
        market_state = "CLOSED"
        is_market_open = False
    else:
        market_state = "OPEN"
        is_market_open = True

    # 2. Check for feed or processing delays (stale data)
    # If market is open and elapsed time since candle close > 7 minutes (420s), data is stale
    elapsed_since_candle_close = (now_et - candle_close_local).total_seconds()
    if is_market_open and elapsed_since_candle_close > 420:
        is_stale = True

    # 3. Evaluate next prediction time based on completed candle
    candle_sess_close = session_close(candle_date) if is_trading_day(candle_date) else time(16, 0)
    candle_sess_close_dt = market_timestamp(candle_date, candle_sess_close.strftime("%H:%M"))

    if candle_close_local < candle_sess_close_dt:
        # Next 5-minute candle will complete 5 minutes after current candle close
        next_pred_dt = candle_close_local + timedelta(minutes=BAR_MINUTES)
        next_display = next_pred_dt.strftime("%I:%M %p ET").lstrip("0")
    else:
        # Session ended with this candle; next prediction is at first 5m completion of next session
        next_day = next_trading_day(candle_date)
        next_pred_dt = market_timestamp(next_day, "09:35")
        next_display = f"{next_day.strftime('%b %d')} 9:35 AM ET"

    last_candle_display = candle_local.strftime("%I:%M %p ET").lstrip("0")
    last_candle_close_display = candle_close_local.strftime("%I:%M %p ET").lstrip("0")
    last_analysis_completed_display = now_et.strftime("%I:%M %p ET").lstrip("0")

    # 4. Construct headline display
    if is_stale:
        headline = "WAITING FOR MARKET DATA"
    elif not is_market_open:
        if not is_trading:
            next_day = next_trading_day(now_et.date())
            headline = f"MARKET CLOSED — Next prediction: {next_day.strftime('%b %d')} 9:35 AM ET"
        elif sess_open_dt and now_et < sess_open_dt:
            headline = f"PRE-MARKET — Next prediction: {now_et.strftime('%b %d')} 9:35 AM ET"
        else:
            next_day = next_trading_day(now_et.date())
            headline = f"MARKET CLOSED — Next prediction: {next_day.strftime('%b %d')} 9:35 AM ET"
    else:
        headline = f"NEXT PREDICTION: {next_display}"

    return {
        "candle_timestamp": candle_local.isoformat(),
        "candle_close_timestamp": candle_close_local.isoformat(),
        "next_prediction_timestamp": next_pred_dt.isoformat(),
        "next_prediction_display": next_display,
        "last_candle_display": last_candle_display,
        "last_candle_close_display": last_candle_close_display,
        "last_analysis_completed_display": last_analysis_completed_display,
        "last_prediction_display": last_candle_close_display,
        "is_market_open": is_market_open,
        "market_state": market_state,
        "headline_display": headline,
        "is_stale": is_stale,
    }


def evaluate_60m_decision(
    candles: pd.DataFrame,
    indicators: pd.DataFrame,
    structure: dict[str, Any],
    regime: dict[str, Any],
    market_intelligence: dict[str, Any] | None = None,
    ml_evidence: dict[str, Any] | None = None,
    active_60m_signal: dict[str, Any] | None = None,
    weights: dict[str, float] | None = None,
    as_of: datetime | None = None,
    is_stale: bool = False,
) -> dict[str, Any]:
    """Evaluate completed candles and produce a 60-minute trading decision."""
    cfg_weights = weights or DEFAULT_WEIGHTS
    last_row = indicators.iloc[-1]
    last_candle = candles.iloc[-1]
    price = float(last_candle["close"])
    atr = float(last_row.get("atr_14", 1.0))
    if not np.isfinite(atr) or atr <= 0:
        atr = max(0.5, price * 0.005)

    candle_ts = pd.Timestamp(last_candle["timestamp"]).tz_convert("UTC")
    candle_local = candle_ts.astimezone(ET)

    # ── 1. Evaluate Evidence Groups ───────────────────────────────────────────
    trend_score, trend_evidence, trend_meta = _evaluate_trend(indicators, market_intelligence)
    struct_score, struct_evidence, struct_meta = _evaluate_structure(structure, price, atr, market_intelligence)
    mom_score, mom_evidence, mom_meta = _evaluate_momentum(indicators)
    vol_score, vol_evidence, vol_meta = _evaluate_volume(indicators, market_intelligence, last_candle)
    vwap_score, vwap_evidence, vwap_meta = _evaluate_vwap_price_action(indicators, last_candle, atr)

    scores: dict[str, float | None] = {
        "trend": trend_score,
        "structure": struct_score,
        "momentum": mom_score,
        "volume": vol_score,
        "vwap_price_action": vwap_score,
    }

    # Calculate weighted directional score across available evidence
    active_weights: dict[str, float] = {}
    weighted_sum = 0.0
    for key, weight in cfg_weights.items():
        val = scores.get(key)
        if val is not None and np.isfinite(val):
            active_weights[key] = weight
            weighted_sum += weight * val

    total_weight = sum(active_weights.values())
    directional_score = round(weighted_sum / total_weight, 4) if total_weight > 0 else 0.0

    # Determine directional bias
    if directional_score >= 0.15:
        directional_bias = "BULLISH"
    elif directional_score <= -0.15:
        directional_bias = "BEARISH"
    else:
        directional_bias = "NEUTRAL"

    # ── 2. Risk Management Plan ───────────────────────────────────────────────
    candidate_direction = "BUY" if directional_score > 0 else "SELL"
    risk_plan = _build_60m_risk_plan(candidate_direction, price, atr, structure)

    # ── 3. Decision Logic & Gating Rules ─────────────────────────────────────
    reasons: list[str] = []
    invalidation_conditions: list[str] = []
    gating_status = "PASS"

    # Rule A: Opening Discovery Curfew (09:30–09:45 ET)
    if candle_local.time() < OPENING_CURFEW_ET:
        decision = "HOLD"
        gating_status = "OPENING_DISCOVERY"
        reasons.append("Opening price discovery phase (09:30–09:45 ET). 15-minute Opening Range is still forming.")
        invalidation_conditions.append("Opening Range High/Low established after 09:45 ET.")

    # Rule B: Session Close Curfew (>= 15:15 ET)
    elif candle_local.time() >= CLOSING_CURFEW_ET:
        decision = "HOLD"
        gating_status = "SESSION_CLOSE_PROXIMITY"
        reasons.append(f"Within 45 minutes of market close ({candle_local.strftime('%H:%M')} ET). Insufficient time remaining for intended 60-minute holding horizon.")
        invalidation_conditions.append("Market closes at 16:00 ET.")

    # Rule C: Volatility / Transitional Regime Gate
    elif regime.get("classification") == "TRANSITIONAL":
        decision = "HOLD"
        gating_status = "TRANSITIONAL_REGIME"
        reasons.append("Market regime is TRANSITIONAL; moving averages and trend slopes lack consistent alignment.")
        invalidation_conditions.append("ADX must break above 20 with clear moving average directional slope.")

    # Rule D: Sub-threshold Directional Score
    elif abs(directional_score) < DECISION_THRESHOLD:
        decision = "HOLD"
        gating_status = "SUB_THRESHOLD"
        reasons.append(f"Directional strength score {directional_score:+.3f} is within neutral zone [-{DECISION_THRESHOLD}, +{DECISION_THRESHOLD}].")
        reasons.extend(trend_evidence[:1])
        invalidation_conditions.append(f"Directional score magnitude must reach >= {DECISION_THRESHOLD}.")

    # Rule E: Bullish Candidate Evaluation (Score >= +0.35)
    elif directional_score >= DECISION_THRESHOLD:
        reasons.extend(trend_evidence[:2])
        reasons.extend(struct_evidence[:1])
        reasons.extend(vwap_evidence[:1])

        # Confirmation 1: Trend or Structure confirmation
        trend_or_struct_ok = (trend_score is not None and trend_score > 0) or (struct_score is not None and struct_score > 0)
        # Confirmation 2: Overhead Resistance Clearance
        dist_res = struct_meta.get("nearest_resistance_distance_atr")
        breakout_status = structure.get("breakout", {}).get("status")
        blocked_by_res = (dist_res is not None and dist_res < PROXIMITY_VETO_ATR and breakout_status != "CONFIRMED_BREAKOUT")

        # Confirmation 3: Risk Plan Validity & R:R
        rr_ok = bool(risk_plan.get("valid") and (risk_plan.get("risk_reward_ratio") or 0) >= MIN_TARGET_RISK_REWARD)

        if not trend_or_struct_ok:
            decision = "WAIT FOR CONFIRMATION"
            gating_status = "AWAITING_TREND_STRUCTURE_ALIGNMENT"
            reasons.append("Bullish momentum present, but moving average structure or market geometry has not confirmed trend continuation.")
            invalidation_conditions.append("5M/15M EMA alignment and higher-high/higher-low pivot confirmation required.")
        elif blocked_by_res:
            decision = "WAIT FOR CONFIRMATION"
            gating_status = "RESISTANCE_PROXIMITY"
            reasons.append(f"Price is within {dist_res:.2f} ATR of overhead resistance ({struct_meta.get('nearest_resistance_price')}). Awaiting confirmed breakout.")
            invalidation_conditions.append(f"Confirmed breakout above {struct_meta.get('nearest_resistance_price')} with volume confirmation.")
        elif not rr_ok:
            decision = "WAIT FOR CONFIRMATION"
            gating_status = "UNFAVORABLE_RISK_REWARD"
            reasons.append(f"Reward-to-risk ratio ({risk_plan.get('risk_reward_ratio', 0):.2f}) is below the required {MIN_TARGET_RISK_REWARD}:1 minimum.")
            invalidation_conditions.append("Favorable structural pullback required to establish viable stop-loss geometry.")
        else:
            # Check active signal hysteresis
            act_sig_dir = (active_60m_signal.get("decision") or active_60m_signal.get("direction") or "").upper() if active_60m_signal else ""
            if active_60m_signal and act_sig_dir == "BUY" and active_60m_signal.get("status") in ("ACTIVE", "NEW", "TARGET_1_HIT"):
                decision = "HOLD"
                gating_status = "ALREADY_ACTIVE_BUY"
                reasons.append(f"A 60-minute BUY setup is already active (entered at ${float(active_60m_signal.get('trigger_price', 0)):.2f}). Maintaining position.")
            elif active_60m_signal and act_sig_dir == "SELL" and active_60m_signal.get("status") in ("ACTIVE", "NEW"):
                decision = "WAIT FOR CONFIRMATION"
                gating_status = "OPPOSING_ACTIVE_SIGNAL"
                reasons.append("Prior 60-minute SELL thesis has not been formally stopped out or invalidated. Awaiting clean transition.")
            else:
                decision = "BUY"
                gating_status = "CONFIRMED_BUY"
                invalidation_conditions.append(f"Price close below stop-loss at ${risk_plan.get('stop_loss', 0):.2f}.")
                invalidation_conditions.append("Confirmed breakdown below session VWAP.")

    # Rule F: Bearish Candidate Evaluation (Score <= -0.35)
    else:
        reasons.extend(trend_evidence[:2])
        reasons.extend(struct_evidence[:1])
        reasons.extend(vwap_evidence[:1])

        trend_or_struct_ok = (trend_score is not None and trend_score < 0) or (struct_score is not None and struct_score < 0)
        dist_sup = struct_meta.get("nearest_support_distance_atr")
        breakdown_status = structure.get("breakdown", {}).get("status")
        blocked_by_sup = (dist_sup is not None and dist_sup < PROXIMITY_VETO_ATR and breakdown_status != "CONFIRMED_BREAKDOWN")
        rr_ok = bool(risk_plan.get("valid") and (risk_plan.get("risk_reward_ratio") or 0) >= MIN_TARGET_RISK_REWARD)

        if not trend_or_struct_ok:
            decision = "WAIT FOR CONFIRMATION"
            gating_status = "AWAITING_TREND_STRUCTURE_ALIGNMENT"
            reasons.append("Bearish momentum present, but moving average structure or market geometry has not confirmed breakdown continuation.")
            invalidation_conditions.append("5M/15M EMA alignment and lower-high/lower-low pivot confirmation required.")
        elif blocked_by_sup:
            decision = "WAIT FOR CONFIRMATION"
            gating_status = "SUPPORT_PROXIMITY"
            reasons.append(f"Price is within {dist_sup:.2f} ATR of major support ({struct_meta.get('nearest_support_price')}). Awaiting confirmed breakdown.")
            invalidation_conditions.append(f"Confirmed breakdown below {struct_meta.get('nearest_support_price')} with volume confirmation.")
        elif not rr_ok:
            decision = "WAIT FOR CONFIRMATION"
            gating_status = "UNFAVORABLE_RISK_REWARD"
            reasons.append(f"Reward-to-risk ratio ({risk_plan.get('risk_reward_ratio', 0):.2f}) is below the required {MIN_TARGET_RISK_REWARD}:1 minimum.")
            invalidation_conditions.append("Favorable structural bounce required to establish viable stop-loss geometry.")
        else:
            act_sig_dir = (active_60m_signal.get("decision") or active_60m_signal.get("direction") or "").upper() if active_60m_signal else ""
            if active_60m_signal and act_sig_dir == "SELL" and active_60m_signal.get("status") in ("ACTIVE", "NEW", "TARGET_1_HIT"):
                decision = "HOLD"
                gating_status = "ALREADY_ACTIVE_SELL"
                reasons.append(f"A 60-minute SELL setup is already active (entered at ${float(active_60m_signal.get('trigger_price', 0)):.2f}). Maintaining position.")
            elif active_60m_signal and act_sig_dir == "BUY" and active_60m_signal.get("status") in ("ACTIVE", "NEW"):
                decision = "WAIT FOR CONFIRMATION"
                gating_status = "OPPOSING_ACTIVE_SIGNAL"
                reasons.append("Prior 60-minute BUY thesis has not been formally stopped out or invalidated. Awaiting clean transition.")
            else:
                decision = "SELL"
                gating_status = "CONFIRMED_SELL"
                invalidation_conditions.append(f"Price close above stop-loss at ${risk_plan.get('stop_loss', 0):.2f}.")
                invalidation_conditions.append("Confirmed breakout above session VWAP.")

    # ── 4. Short-Term 10-Minute ML Supporting Context ─────────────────────────
    # Kept strictly separate as short-term supporting context; never confused with 60m horizon
    ml_context = {
        "model_name": ml_evidence.get("model_name", "LGBMClassifier") if ml_evidence else "LGBMClassifier",
        "model_version": ml_evidence.get("model_version", "legacy-lightgbm-10m-e1b5cf0") if ml_evidence else "legacy-lightgbm-10m-e1b5cf0",
        "prediction_horizon_minutes": 10,
        "available": bool(ml_evidence and ml_evidence.get("available")),
        "direction": ml_evidence.get("direction") if ml_evidence else None,
        "raw_probability_up": ml_evidence.get("raw_probability_up") if ml_evidence else None,
        "raw_directional_score": ml_evidence.get("raw_directional_score") if ml_evidence else None,
        "score_type": "uncalibrated_raw_model_score_not_confidence",
        "role": "supporting_micro_momentum_context_only",
    }

    # ── Final Risk Plan Validation & Active Signal Resolution ─────────────────
    # Rules:
    # 1. BUY: valid BUY execution plan only when decision is BUY and passes validation
    # 2. SELL: valid SELL execution plan only when decision is SELL and passes validation
    # 3. HOLD / WAIT FOR CONFIRMATION: do not display an approved execution plan
    if decision == "BUY":
        is_ok, err = validate_trade_plan(risk_plan, "BUY", price)
        if is_ok:
            final_risk_plan = dict(risk_plan)
            final_risk_plan["valid"] = True
            final_risk_plan["approved"] = True
        else:
            decision = "WAIT FOR CONFIRMATION"
            gating_status = "INVALID_RISK_PLAN"
            reasons.append(f"Trade plan validation failed: {err}")
            final_risk_plan = {"valid": False, "approved": False, "reason": err, "direction": "BUY"}
    elif decision == "SELL":
        is_ok, err = validate_trade_plan(risk_plan, "SELL", price)
        if is_ok:
            final_risk_plan = dict(risk_plan)
            final_risk_plan["valid"] = True
            final_risk_plan["approved"] = True
        else:
            decision = "WAIT FOR CONFIRMATION"
            gating_status = "INVALID_RISK_PLAN"
            reasons.append(f"Trade plan validation failed: {err}")
            final_risk_plan = {"valid": False, "approved": False, "reason": err, "direction": "SELL"}
    else:
        # HOLD or WAIT FOR CONFIRMATION -> never display approved setup
        final_risk_plan = {
            "valid": False,
            "approved": False,
            "direction": candidate_direction,
            "reason": f"No new trade setup approved. Current recommendation is {decision}.",
        }

    # Format existing active signal summary (if one is active)
    active_signal_summary = None
    if active_60m_signal and active_60m_signal.get("status") in ("ACTIVE", "NEW", "TARGET_1_HIT"):
        act_dir = (active_60m_signal.get("decision") or active_60m_signal.get("direction") or "").upper()
        act_entry = float(active_60m_signal.get("trigger_price", price))
        act_stop = float(active_60m_signal.get("stop_loss", 0.0))
        act_t1 = float(active_60m_signal.get("target_1", 0.0))
        act_t2 = float(active_60m_signal.get("target_2", 0.0))
        act_valid, act_err = validate_trade_plan({
            "valid": True,
            "direction": act_dir,
            "trigger_price": act_entry,
            "stop_loss": act_stop,
            "target_1": act_t1,
            "target_2": act_t2,
        }, act_dir, act_entry)

        unrealized_pct = None
        if act_entry > 0:
            if act_dir == "BUY":
                unrealized_pct = round(((price - act_entry) / act_entry) * 100.0, 3)
            elif act_dir == "SELL":
                unrealized_pct = round(((act_entry - price) / act_entry) * 100.0, 3)

        active_signal_summary = {
            "signal_id": active_60m_signal.get("signal_id"),
            "direction": act_dir,
            "status": active_60m_signal.get("status"),
            "trigger_price": act_entry,
            "entry_zone": {
                "lower": float(active_60m_signal.get("entry_zone_lower", act_entry)),
                "upper": float(active_60m_signal.get("entry_zone_upper", act_entry)),
            },
            "stop_loss": act_stop,
            "target_1": act_t1,
            "target_2": act_t2,
            "risk_reward_ratio": active_60m_signal.get("risk_reward_ratio"),
            "candle_timestamp": active_60m_signal.get("candle_timestamp"),
            "is_existing_position": True,
            "valid": act_valid,
            "validation_error": act_err,
            "unrealized_pnl_pct": unrealized_pct,
        }

    # ── 5. Assemble Payload ───────────────────────────────────────────────────
    return {
        "intended_horizon_minutes": HORIZON_MINUTES,
        "intended_horizon_bars": HORIZON_BARS,
        "decision": decision,
        "directional_bias": directional_bias,
        "directional_score": directional_score,
        "score_type": "heuristic_weighted_evidence_not_probability",
        "gating_status": gating_status,
        "reasons": reasons,
        "invalidation_conditions": invalidation_conditions,
        "components": {
            "trend": {
                "score": trend_score,
                "weight": cfg_weights["trend"],
                "weighted_contribution": round((trend_score or 0) * cfg_weights["trend"], 4),
                "summary": trend_meta.get("classification", "NEUTRAL"),
                "evidence": trend_evidence,
                "meta": trend_meta,
            },
            "structure": {
                "score": struct_score,
                "weight": cfg_weights["structure"],
                "weighted_contribution": round((struct_score or 0) * cfg_weights["structure"], 4),
                "summary": struct_meta.get("classification", "RANGEBOUND"),
                "evidence": struct_evidence,
                "meta": struct_meta,
            },
            "momentum": {
                "score": mom_score,
                "weight": cfg_weights["momentum"],
                "weighted_contribution": round((mom_score or 0) * cfg_weights["momentum"], 4),
                "summary": mom_meta.get("classification", "NEUTRAL"),
                "evidence": mom_evidence,
                "meta": mom_meta,
            },
            "volume": {
                "score": vol_score,
                "weight": cfg_weights["volume"],
                "weighted_contribution": round((vol_score or 0) * cfg_weights["volume"], 4),
                "summary": vol_meta.get("classification", "NORMAL"),
                "evidence": vol_evidence,
                "meta": vol_meta,
            },
            "vwap_price_action": {
                "score": vwap_score,
                "weight": cfg_weights["vwap_price_action"],
                "weighted_contribution": round((vwap_score or 0) * cfg_weights["vwap_price_action"], 4),
                "summary": vwap_meta.get("classification", "AT_VWAP"),
                "evidence": vwap_evidence,
                "meta": vwap_meta,
            },
        },
        "risk_plan": final_risk_plan,
        "active_signal": active_signal_summary,
        "dedicated_60m_ml_model": {
            "status": "NOT_AVAILABLE",
            "message": "60-minute ML model intentionally deferred to avoid unvalidated training shortcuts. Heuristic multi-evidence engine active.",
        },
        "ml_10m_supporting": ml_context,
        "timing": compute_prediction_schedule(candle_local, as_of=as_of, is_stale=is_stale),
        "evaluation_candle_timestamp": candle_ts.isoformat(),
        "decision_timestamp": datetime.now(ET).isoformat(),
        "market_timezone": TIMEZONE,
    }


# ── Evidence Evaluators ───────────────────────────────────────────────────────

def _evaluate_trend(
    indicators: pd.DataFrame,
    intel: dict[str, Any] | None,
) -> tuple[float, list[str], dict[str, Any]]:
    """Evaluate 5m EMA/SMA structure and 15m/1h trend alignment."""
    row = indicators.iloc[-1]
    evidence: list[str] = []
    points = 0.0
    max_points = 5.0

    # 1. 5-minute Moving Average Alignment (EMA 9/20/50)
    ema9 = row.get("ema_9")
    ema20 = row.get("ema_20")
    ema50 = row.get("ema_50")

    if ema9 is not None and ema20 is not None and ema50 is not None:
        if ema9 > ema20 > ema50:
            points += 1.5
            evidence.append(f"5M EMA alignment is bullish (EMA9 > EMA20 > EMA50).")
        elif ema9 < ema20 < ema50:
            points -= 1.5
            evidence.append(f"5M EMA alignment is bearish (EMA9 < EMA20 < EMA50).")
        else:
            evidence.append(f"5M EMA structure is mixed/crossing.")

    # 2. Moving Average Slopes (EMA 20 & 50)
    s20 = row.get("ema_20_slope_5")
    s50 = row.get("ema_50_slope_10")
    if s20 is not None and s50 is not None:
        if s20 > 0 and s50 > 0:
            points += 1.0
            evidence.append("EMA 20/50 slopes are positive (trend expanding upward).")
        elif s20 < 0 and s50 < 0:
            points -= 1.0
            evidence.append("EMA 20/50 slopes are negative (trend expanding downward).")

    # 3. Multi-Timeframe Alignment (15M and 1H from Market Intelligence)
    trend_align = intel.get("trend_alignment", {}) if intel else {}
    overall_mtf = trend_align.get("overall", "NEUTRAL")
    timeframes = {t.get("timeframe"): t.get("trend") for t in trend_align.get("timeframes", [])}

    t15 = timeframes.get("15M", "NEUTRAL")
    t1h = timeframes.get("1H", "NEUTRAL")

    if t1h == "BULLISH":
        points += 1.5
        evidence.append("1H higher-timeframe trend is BULLISH.")
    elif t1h == "BEARISH":
        points -= 1.5
        evidence.append("1H higher-timeframe trend is BEARISH.")

    if t15 == "BULLISH":
        points += 1.0
        evidence.append("15M intermediate trend is BULLISH.")
    elif t15 == "BEARISH":
        points -= 1.0
        evidence.append("15M intermediate trend is BEARISH.")

    norm_score = float(np.clip(points / max_points, -1.0, 1.0))
    classification = (
        "STRONG_BULLISH" if norm_score >= 0.6
        else "BULLISH" if norm_score >= 0.2
        else "STRONG_BEARISH" if norm_score <= -0.6
        else "BEARISH" if norm_score <= -0.2
        else "NEUTRAL"
    )

    meta = {
        "classification": classification,
        "ema_5m_alignment": "BULLISH" if (ema9 and ema20 and ema50 and ema9 > ema20 > ema50) else ("BEARISH" if (ema9 and ema20 and ema50 and ema9 < ema20 < ema50) else "MIXED"),
        "trend_15m": t15,
        "trend_1h": t1h,
        "overall_mtf": overall_mtf,
    }
    return round(norm_score, 4), evidence, meta


def _evaluate_structure(
    structure: dict[str, Any],
    price: float,
    atr: float,
    intel: dict[str, Any] | None,
) -> tuple[float, list[str], dict[str, Any]]:
    """Evaluate support/resistance zones, breakouts, sequence, and prior day levels."""
    evidence: list[str] = []
    points = 0.0
    max_points = 4.0

    # 1. Swing Sequence (Higher Highs / Lower Lows)
    seq = structure.get("sequence", {})
    if seq.get("higher_high") and seq.get("higher_low"):
        points += 1.5
        evidence.append("Market structure sequence: Higher Highs and Higher Lows (bullish order flow).")
    elif seq.get("lower_high") and seq.get("lower_low"):
        points -= 1.5
        evidence.append("Market structure sequence: Lower Highs and Lower Lows (bearish order flow).")

    # 2. Breakouts / Breakdowns
    bk_out = structure.get("breakout", {})
    bk_dn = structure.get("breakdown", {})

    if bk_out.get("status") == "CONFIRMED_BREAKOUT":
        points += 2.0
        evidence.append("CONFIRMED BREAKOUT above structural resistance with volume.")
    elif bk_out.get("status") == "FAILED_BREAKOUT":
        points -= 1.0
        evidence.append("FAILED BREAKOUT above resistance (bull trap).")

    if bk_dn.get("status") == "CONFIRMED_BREAKDOWN":
        points -= 2.0
        evidence.append("CONFIRMED BREAKDOWN below structural support with volume.")
    elif bk_dn.get("status") == "FAILED_BREAKDOWN":
        points += 1.0
        evidence.append("FAILED BREAKDOWN below support (bear trap).")

    # 3. Proximity to nearest support / resistance
    near_sup = structure.get("nearest_support")
    near_res = structure.get("nearest_resistance")

    sup_dist_atr = None
    res_dist_atr = None

    if near_sup:
        sup_price = float(near_sup.get("upper", 0.0))
        sup_dist_atr = round((price - sup_price) / atr, 2)
        if sup_dist_atr < 0.35 and bk_dn.get("status") != "CONFIRMED_BREAKDOWN":
            evidence.append(f"Price is testing Support ({sup_price:.2f}, {sup_dist_atr:.2f} ATR away).")

    if near_res:
        res_price = float(near_res.get("lower", 0.0))
        res_dist_atr = round((res_price - price) / atr, 2)
        if res_dist_atr < 0.35 and bk_out.get("status") != "CONFIRMED_BREAKOUT":
            evidence.append(f"Price is testing Resistance ({res_price:.2f}, {res_dist_atr:.2f} ATR away).")

    # 4. Opening Range & Previous Day Context
    if intel:
        orb = intel.get("opening_range", {})
        if orb.get("classification") == "BULLISH_BREAKOUT":
            points += 1.0
            evidence.append("Price is trading above the 15-minute Opening Range High.")
        elif orb.get("classification") == "BEARISH_BREAKDOWN":
            points -= 1.0
            evidence.append("Price is trading below the 15-minute Opening Range Low.")

        p_levels = intel.get("previous_day_levels", {})
        if p_levels.get("position", {}).get("high") == "ABOVE":
            points += 0.5
            evidence.append("Trading above previous session's high.")
        elif p_levels.get("position", {}).get("low") == "BELOW":
            points -= 0.5
            evidence.append("Trading below previous session's low.")

    norm_score = float(np.clip(points / max_points, -1.0, 1.0))
    classification = (
        "BULLISH_BREAKOUT" if bk_out.get("status") == "CONFIRMED_BREAKOUT"
        else "BEARISH_BREAKDOWN" if bk_dn.get("status") == "CONFIRMED_BREAKDOWN"
        else "BULLISH_STRUCTURE" if norm_score >= 0.3
        else "BEARISH_STRUCTURE" if norm_score <= -0.3
        else "RANGEBOUND"
    )

    meta = {
        "classification": classification,
        "nearest_support_distance_atr": sup_dist_atr,
        "nearest_support_price": near_sup.get("upper") if near_sup else None,
        "nearest_resistance_distance_atr": res_dist_atr,
        "nearest_resistance_price": near_res.get("lower") if near_res else None,
        "breakout_status": bk_out.get("status", "NONE"),
        "breakdown_status": bk_dn.get("status", "NONE"),
    }
    return round(norm_score, 4), evidence, meta


def _evaluate_momentum(indicators: pd.DataFrame) -> tuple[float, list[str], dict[str, Any]]:
    """Evaluate RSI 14/7, MACD histogram & acceleration, ROC."""
    row = indicators.iloc[-1]
    evidence: list[str] = []
    points = 0.0
    max_points = 4.0

    rsi14 = row.get("rsi_14")
    rsi7 = row.get("rsi_7")
    macd_hist = row.get("macd_histogram")
    macd_accel = row.get("macd_histogram_change")
    roc5 = row.get("roc_5_pct")
    roc20 = row.get("roc_20_pct")

    # 1. RSI-14 Levels & Bias
    if rsi14 is not None and np.isfinite(rsi14):
        if rsi14 >= 55:
            points += 1.0
            evidence.append(f"RSI-14 ({rsi14:.1f}) in bullish territory (>55).")
        elif rsi14 <= 45:
            points -= 1.0
            evidence.append(f"RSI-14 ({rsi14:.1f}) in bearish territory (<45).")
        else:
            evidence.append(f"RSI-14 ({rsi14:.1f}) is neutral (45–55).")

    # 2. MACD Histogram & Acceleration
    if macd_hist is not None and np.isfinite(macd_hist):
        if macd_hist > 0:
            points += 1.0
            accel_text = "accelerating" if (macd_accel and macd_accel > 0) else "decelerating"
            evidence.append(f"MACD histogram positive ({macd_hist:+.3f}, {accel_text}).")
        else:
            points -= 1.0
            accel_text = "expanding downward" if (macd_accel and macd_accel < 0) else "contracting"
            evidence.append(f"MACD histogram negative ({macd_hist:+.3f}, {accel_text}).")

        if macd_accel is not None and np.isfinite(macd_accel):
            if macd_accel > 0:
                points += 0.5
            elif macd_accel < 0:
                points -= 0.5

    # 3. Rate of Change (ROC-5 & ROC-20)
    if roc5 is not None and roc20 is not None:
        if roc5 > 0 and roc20 > 0:
            points += 1.0
            evidence.append(f"Multi-horizon rate of change is positive (ROC-5: {roc5:+.2f}%, ROC-20: {roc20:+.2f}%).")
        elif roc5 < 0 and roc20 < 0:
            points -= 1.0
            evidence.append(f"Multi-horizon rate of change is negative (ROC-5: {roc5:+.2f}%, ROC-20: {roc20:+.2f}%).")

    # 4. Divergence Overrides
    if row.get("bullish_divergence"):
        points += 1.5
        evidence.append("Bullish momentum divergence detected at recent swing lows.")
    elif row.get("bearish_divergence"):
        points -= 1.5
        evidence.append("Bearish momentum divergence detected at recent swing highs.")

    norm_score = float(np.clip(points / max_points, -1.0, 1.0))
    classification = (
        "STRONG_POSITIVE" if norm_score >= 0.5
        else "POSITIVE" if norm_score >= 0.2
        else "STRONG_NEGATIVE" if norm_score <= -0.5
        else "NEGATIVE" if norm_score <= -0.2
        else "NEUTRAL"
    )

    meta = {
        "classification": classification,
        "rsi_14": round(rsi14, 1) if (rsi14 is not None and np.isfinite(rsi14)) else None,
        "macd_histogram": round(macd_hist, 4) if (macd_hist is not None and np.isfinite(macd_hist)) else None,
        "roc_5_pct": round(roc5, 2) if (roc5 is not None and np.isfinite(roc5)) else None,
    }
    return round(norm_score, 4), evidence, meta


def _evaluate_volume(
    indicators: pd.DataFrame,
    intel: dict[str, Any] | None,
    last_candle: pd.Series,
) -> tuple[float, list[str], dict[str, Any]]:
    """Evaluate session rolling volume ratio, same-slot RVOL, and directional conviction."""
    row = indicators.iloc[-1]
    evidence: list[str] = []
    points = 0.0
    max_points = 3.0

    c_open = float(last_candle["open"])
    c_close = float(last_candle["close"])
    bar_sign = 1 if c_close >= c_open else -1

    # 1. Session Rolling Volume Ratio (Volume / Trailing 20-bar SMA)
    rvol_session = row.get("relative_volume")
    if rvol_session is not None and np.isfinite(rvol_session):
        if rvol_session >= 1.2:
            points += 1.5 * bar_sign
            evidence.append(f"Session Rolling Volume Ratio is elevated at {rvol_session:.2f}x (confirming {('bullish' if bar_sign > 0 else 'bearish')} bar).")
        elif rvol_session < 0.75:
            points -= 0.5 * bar_sign
            evidence.append(f"Session Rolling Volume Ratio is low at {rvol_session:.2f}x (light participation).")
        else:
            evidence.append(f"Session Rolling Volume Ratio is normal at {rvol_session:.2f}x.")

    # 2. Historical Same-Slot RVOL (from Market Intelligence)
    intel_rvol = intel.get("rvol", {}) if intel else {}
    slot_rvol = intel_rvol.get("rvol")
    slot_class = intel_rvol.get("classification", "UNAVAILABLE")

    if slot_rvol is not None and np.isfinite(slot_rvol):
        if slot_rvol >= 1.2:
            points += 1.5 * bar_sign
            evidence.append(f"Historical Same-Slot RVOL is {slot_rvol:.2f}x ({slot_class}).")
        elif slot_rvol < 0.75:
            evidence.append(f"Historical Same-Slot RVOL is below average at {slot_rvol:.2f}x ({slot_class}).")

    norm_score = float(np.clip(points / max_points, -1.0, 1.0))
    classification = (
        "HIGH_VOLUME_CONFIRMED" if abs(norm_score) >= 0.5
        else "MODERATE_VOLUME" if abs(norm_score) >= 0.2
        else "LOW_VOLUME"
    )

    meta = {
        "classification": classification,
        "session_rolling_volume_ratio": round(rvol_session, 2) if (rvol_session is not None and np.isfinite(rvol_session)) else None,
        "historical_same_slot_rvol": round(slot_rvol, 2) if (slot_rvol is not None and np.isfinite(slot_rvol)) else None,
        "historical_same_slot_classification": slot_class,
    }
    return round(norm_score, 4), evidence, meta


def _evaluate_vwap_price_action(
    indicators: pd.DataFrame,
    last_candle: pd.Series,
    atr: float,
) -> tuple[float, list[str], dict[str, Any]]:
    """Evaluate Session VWAP position/distance and candle body/wick dynamics."""
    row = indicators.iloc[-1]
    evidence: list[str] = []
    points = 0.0
    max_points = 3.0

    c_open = float(last_candle["open"])
    c_high = float(last_candle["high"])
    c_low = float(last_candle["low"])
    c_close = float(last_candle["close"])
    vwap = row.get("vwap")

    # 1. Price relative to Session VWAP
    vwap_dist_atr = None
    if vwap is not None and np.isfinite(vwap) and vwap > 0:
        vwap_dist = c_close - vwap
        vwap_dist_atr = round(vwap_dist / atr, 2)
        if vwap_dist > 0:
            points += 1.5
            evidence.append(f"Price is trading ABOVE Session VWAP (+${vwap_dist:.2f}, +{vwap_dist_atr:.2f} ATR).")
        else:
            points -= 1.5
            evidence.append(f"Price is trading BELOW Session VWAP (-${abs(vwap_dist):.2f}, {vwap_dist_atr:.2f} ATR).")

    # 2. Candle Body and Wick Analysis
    c_range = max(0.01, c_high - c_low)
    body = abs(c_close - c_open)
    upper_wick = c_high - max(c_open, c_close)
    lower_wick = min(c_open, c_close) - c_low

    if c_close > c_open:
        if upper_wick < body * 0.3:
            points += 1.0
            evidence.append("Bullish candle closed near bar high with minimal upper wick.")
        if lower_wick > body * 1.5:
            points += 0.5
            evidence.append("Bullish lower wick indicates buying rejection at lower prices.")
    elif c_close < c_open:
        if lower_wick < body * 0.3:
            points -= 1.0
            evidence.append("Bearish candle closed near bar low with minimal lower wick.")
        if upper_wick > body * 1.5:
            points -= 0.5
            evidence.append("Bearish upper wick indicates selling rejection at higher prices.")

    norm_score = float(np.clip(points / max_points, -1.0, 1.0))
    classification = (
        "BULLISH_ABOVE_VWAP" if (vwap_dist_atr is not None and vwap_dist_atr > 0.2)
        else "BEARISH_BELOW_VWAP" if (vwap_dist_atr is not None and vwap_dist_atr < -0.2)
        else "AT_VWAP"
    )

    meta = {
        "classification": classification,
        "session_vwap": round(vwap, 2) if (vwap is not None and np.isfinite(vwap)) else None,
        "vwap_distance_atr": vwap_dist_atr,
        "candle_body_ratio": round(body / c_range, 2),
    }
    return round(norm_score, 4), evidence, meta


# ── Risk Management Helper ───────────────────────────────────────────────────

def validate_trade_plan(
    plan: dict[str, Any] | None,
    expected_direction: str,
    trigger_price: float,
) -> tuple[bool, str | None]:
    """Validate trade-plan direction, structure, and minimum risk/reward ratio.

    Rules:
      - BUY: stop_loss < trigger_price < target_1 < target_2
      - SELL: target_2 < target_1 < trigger_price < stop_loss
      - Risk/Reward ratio >= MIN_TARGET_RISK_REWARD (1.5:1)
    """
    if not plan or not isinstance(plan, dict):
        return False, "Trade plan is missing or invalid structure."

    if not plan.get("valid", True):
        return False, plan.get("reason", "Trade plan marked invalid.")

    plan_dir = (plan.get("direction") or expected_direction).upper()
    if plan_dir != expected_direction.upper():
        return False, f"Plan direction '{plan_dir}' does not match expected direction '{expected_direction}'."

    stop_loss = plan.get("stop_loss")
    target_1 = plan.get("target_1")
    target_2 = plan.get("target_2")

    if stop_loss is None or target_1 is None or target_2 is None:
        return False, "Incomplete price levels in trade plan (missing stop or targets)."

    try:
        stop = float(stop_loss)
        t1 = float(target_1)
        t2 = float(target_2)
        entry = float(plan.get("trigger_price", trigger_price))
    except (ValueError, TypeError):
        return False, "Non-numeric price levels in trade plan."

    if expected_direction.upper() == "BUY":
        if not (stop < entry):
            return False, f"BUY stop-loss (${stop:.2f}) must be strictly below entry price (${entry:.2f})."
        if not (entry < t1):
            return False, f"BUY target 1 (${t1:.2f}) must be strictly above entry price (${entry:.2f})."
        if not (t1 < t2):
            return False, f"BUY target 2 (${t2:.2f}) must be strictly above target 1 (${t1:.2f})."
        risk = entry - stop
        reward = t2 - entry
    elif expected_direction.upper() == "SELL":
        if not (stop > entry):
            return False, f"SELL stop-loss (${stop:.2f}) must be strictly above entry price (${entry:.2f})."
        if not (entry > t1):
            return False, f"SELL target 1 (${t1:.2f}) must be strictly below entry price (${entry:.2f})."
        if not (t1 > t2):
            return False, f"SELL target 2 (${t2:.2f}) must be strictly below target 1 (${t1:.2f})."
        risk = stop - entry
        reward = entry - t2
    else:
        return False, f"Unknown trade plan direction '{expected_direction}'."

    if risk <= 0:
        return False, f"Invalid risk per share (${risk:.2f} <= 0)."
    if reward <= 0:
        return False, f"Invalid reward per share (${reward:.2f} <= 0)."

    calculated_rr = round(reward / risk, 2)
    if calculated_rr < MIN_TARGET_RISK_REWARD:
        return False, f"Risk/reward ratio ({calculated_rr:.2f}:1) is below required minimum {MIN_TARGET_RISK_REWARD}:1."

    return True, None


def _build_60m_risk_plan(
    direction: str,
    price: float,
    atr: float,
    structure: dict[str, Any],
) -> dict[str, Any]:
    """Calculate structural entry zone, stop loss, Target 1 (1.0R), and Target 2 (1.5-2.0R)."""
    entry_band_half = round(0.1 * atr, 2)
    entry_zone = {
        "lower": round(price - entry_band_half, 2),
        "upper": round(price + entry_band_half, 2),
    }

    if direction == "BUY":
        near_sup = structure.get("nearest_support")
        if near_sup and near_sup.get("lower"):
            raw_stop = float(near_sup["lower"]) - (0.25 * atr)
        else:
            raw_stop = price - (1.25 * atr)

        # Ensure stop is at least 0.40 ATR below entry to prevent premature stopouts in noise
        stop_loss = round(min(raw_stop, price - (0.40 * atr)), 2)
        risk = round(price - stop_loss, 2)

        if risk <= 0:
            return {"valid": False, "reason": "Calculated stop loss is above entry price.", "direction": "BUY"}

        target_1 = round(price + (1.0 * risk), 2)
        near_res = structure.get("nearest_resistance")
        if near_res and float(near_res.get("lower", 0)) >= price + (1.5 * risk):
            target_2 = round(float(near_res["lower"]), 2)
        else:
            target_2 = round(price + (2.0 * risk), 2)

        rr_ratio = round((target_2 - price) / risk, 2)
        candidate_plan = {
            "valid": True,
            "direction": "BUY",
            "trigger_price": round(price, 2),
            "entry_zone": entry_zone,
            "stop_loss": stop_loss,
            "target_1": target_1,
            "target_2": target_2,
            "risk_per_share": risk,
            "risk_reward_ratio": rr_ratio,
            "stop_method": "structural_support_with_0.25_atr_buffer",
            "target_method": "1.0R_milestone_and_2.0R_structural_expansion",
        }
        is_ok, err = validate_trade_plan(candidate_plan, "BUY", price)
        if not is_ok:
            candidate_plan["valid"] = False
            candidate_plan["reason"] = err
        return candidate_plan
    else:
        near_res = structure.get("nearest_resistance")
        if near_res and near_res.get("upper"):
            raw_stop = float(near_res["upper"]) + (0.25 * atr)
        else:
            raw_stop = price + (1.25 * atr)

        stop_loss = round(max(raw_stop, price + (0.40 * atr)), 2)
        risk = round(stop_loss - price, 2)

        if risk <= 0:
            return {"valid": False, "reason": "Calculated stop loss is below entry price.", "direction": "SELL"}

        target_1 = round(price - (1.0 * risk), 2)
        near_sup = structure.get("nearest_support")
        if near_sup and float(near_sup.get("upper", 0)) <= price - (1.5 * risk):
            target_2 = round(float(near_sup["upper"]), 2)
        else:
            target_2 = round(price - (2.0 * risk), 2)

        rr_ratio = round((price - target_2) / risk, 2)
        candidate_plan = {
            "valid": True,
            "direction": "SELL",
            "trigger_price": round(price, 2),
            "entry_zone": entry_zone,
            "stop_loss": stop_loss,
            "target_1": target_1,
            "target_2": target_2,
            "risk_per_share": risk,
            "risk_reward_ratio": rr_ratio,
            "stop_method": "structural_resistance_with_0.25_atr_buffer",
            "target_method": "1.0R_milestone_and_2.0R_structural_expansion",
        }
        is_ok, err = validate_trade_plan(candidate_plan, "SELL", price)
        if not is_ok:
            candidate_plan["valid"] = False
            candidate_plan["reason"] = err
        return candidate_plan
