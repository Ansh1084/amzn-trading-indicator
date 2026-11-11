"""Transparent multi-source decision scoring; scores are not probabilities."""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.services.analysis.config import (
    DECISION_THRESHOLD, MIN_TARGET_RISK_REWARD, STRONG_DECISION_THRESHOLD, WEIGHTS,
)
from backend.app.services.analysis.risk_management import build_risk_plan


def _evidence(indicators: pd.DataFrame, structure: dict, regime: dict, ml: dict) -> tuple[dict, list[str], list[str], list[str]]:
    last = indicators.iloc[-1]
    trend = float(np.clip(float(last.trend_alignment) / 2, -1, 1))
    hist = float(last.macd_histogram) if pd.notna(last.macd_histogram) else 0.0
    rsi = float(last.rsi_7) if pd.notna(last.rsi_7) else 50.0
    roc = float(last.roc_5_pct) if pd.notna(last.roc_5_pct) else 0.0
    positive = (rsi > 52 and hist > 0) or (rsi <= 30 and hist > 0)
    negative = (rsi < 48 and hist < 0) or (rsi >= 70 and hist < 0)
    momentum = 1.0 if positive and roc > 0 else .5 if positive else -1.0 if negative and roc < 0 else -.5 if negative else 0.0
    bullish_divergence=bool(last.get("bullish_divergence",False))
    bearish_divergence=bool(last.get("bearish_divergence",False))
    if bullish_divergence and momentum<=0: momentum=.5
    if bearish_divergence and momentum>=0: momentum=-.5
    acceleration=float(last.momentum_acceleration) if pd.notna(last.get("momentum_acceleration")) else 0.0
    rel_vol = float(last.relative_volume) if pd.notna(last.relative_volume) else 1.0
    bar_sign = float(np.sign(float(last.close) - float(last.open)))
    volume = float(np.clip(bar_sign * max(0.0, rel_vol - 1.0) / 1.5, -1, 1))
    sequence = structure.get("sequence", {})
    struct = 0.0
    if sequence.get("higher_high") and sequence.get("higher_low"):
        struct += .7
    if sequence.get("lower_high") and sequence.get("lower_low"):
        struct -= .7
    bo = structure.get("breakout", {}).get("status")
    bd = structure.get("breakdown", {}).get("status")
    if bo == "CONFIRMED_BREAKOUT":
        struct = 1.0
    elif bo == "FAILED_BREAKOUT":
        struct = min(struct, -.6)
    if bd == "CONFIRMED_BREAKDOWN":
        struct = -1.0
    elif bd == "FAILED_BREAKDOWN":
        struct = max(struct, .6)
    # Weak-volume breakouts add no positive structure score.
    if bo == "POTENTIAL_BREAKOUT" and not structure.get("breakout", {}).get("volume_confirmed"):
        struct = min(struct, .25)
    if bd == "POTENTIAL_BREAKDOWN" and not structure.get("breakdown", {}).get("volume_confirmed"):
        struct = max(struct, -.25)
    atr_value = float(last.atr_14) if pd.notna(last.atr_14) and float(last.atr_14) > 0 else 0.0
    price_action = float(np.clip((float(last.close) - float(last.open)) / max(atr_value, 1e-9), -1, 1))
    ml_score = None
    if ml.get("available") and ml.get("raw_probability_up") is not None:
        # A raw probability is used only as a small directional evidence input.
        # It is explicitly not treated as calibrated confidence.
        ml_score = float(np.clip((float(ml["raw_probability_up"]) - .5) / .5, -1, 1))
    components = {"trend": trend, "momentum": momentum, "volume": volume,
                  "structure": struct, "price_action": price_action, "ml": ml_score}
    bullish, bearish, neutral = [], [], []
    reasons = {
        "trend": f"Trend alignment is {last.trend_classification.lower()} (EMA 9/20/50 and slopes).",
        "momentum": f"RSI-7={rsi:.1f}, MACD histogram={hist:.4f}, histogram change={acceleration:+.4f}, ROC-5={roc:.3f}%; bullish divergence={bullish_divergence}, bearish divergence={bearish_divergence}.",
        "volume": f"Relative volume={rel_vol:.2f}x; candle signed volume evidence={volume:+.2f}.",
        "structure": f"Structure sequence={sequence}; breakout={bo}; breakdown={bd}.",
        "price_action": f"Latest completed candle body is {price_action:+.2f} ATR.",
        "ml": (f"Frozen {ml.get('model_version')} raw UP score={float(ml['raw_probability_up']):.3f} (uncalibrated)."
               if ml_score is not None else "Production ML evidence unavailable; technical analysis continues without it."),
    }
    for name, score in components.items():
        text = reasons[name]
        if score is None or abs(score) < .15:
            neutral.append(text)
        elif score > 0:
            bullish.append(text)
        else:
            bearish.append(text)
    return components, bullish, bearish, neutral


def make_decision(indicators: pd.DataFrame, structure: dict, regime: dict,
                  ml: dict, active_direction: str | None = None) -> dict:
    last = indicators.iloc[-1]
    components, bullish, bearish, neutral = _evidence(indicators, structure, regime, ml)
    active_weights = {key: weight for key, weight in WEIGHTS.items() if components[key] is not None}
    denominator = sum(active_weights.values())
    score = float(sum(active_weights[key] * components[key] for key in active_weights) / denominator) if denominator else 0.0
    technical_keys = [key for key in active_weights if key != "ml"]
    technical_weights = {key: WEIGHTS[key] for key in technical_keys}
    technical_score = float(sum(technical_weights[k] * components[k] for k in technical_keys) /
                             sum(technical_weights.values())) if technical_weights else 0.0
    normalized_weights={k:active_weights[k]/denominator for k in active_weights} if denominator else {}
    contributions={k:normalized_weights[k]*components[k] for k in active_weights}
    dominant=sorted(contributions,key=lambda k:abs(contributions[k]),reverse=True)[:3]
    bias = "BULLISH" if score >= .15 else "BEARISH" if score <= -.15 else "NEUTRAL"
    direction = "BUY" if score > 0 else "SELL"
    atr = float(last.atr_14) if pd.notna(last.atr_14) else np.nan
    price = float(last.close)
    plan = build_risk_plan(direction, price, atr, structure)
    conflicts = []
    for a, b in (("trend", "momentum"), ("trend", "ml"), ("structure", "trend")):
        if components.get(a) is not None and components.get(b) is not None and components[a] * components[b] < -.05:
            conflicts.append(f"{a} and {b} evidence disagree.")
    decision, reasons = "WAIT FOR CONFIRMATION", []
    if abs(score) < DECISION_THRESHOLD:
        reasons.append(f"Combined directional score {score:+.3f} is below the {DECISION_THRESHOLD:.2f} action threshold.")
    if conflicts and abs(score) < STRONG_DECISION_THRESHOLD:
        reasons.extend(conflicts)
    near_resistance = structure.get("nearest_resistance")
    near_support = structure.get("nearest_support")
    if direction == "BUY" and near_resistance and near_resistance["distance_atr"] <= .30:
        if structure.get("breakout", {}).get("status") != "CONFIRMED_BREAKOUT":
            reasons.append("Price is within 0.30 ATR of resistance without a close- and volume-confirmed breakout.")
    if direction == "SELL" and near_support and near_support["distance_atr"] <= .30:
        if structure.get("breakdown", {}).get("status") != "CONFIRMED_BREAKDOWN":
            reasons.append("Price is within 0.30 ATR of support without a close- and volume-confirmed breakdown.")
    if not plan.get("valid"):
        reasons.append(plan.get("reason", "Risk levels unavailable."))
    elif plan["risk_reward_ratio"] < MIN_TARGET_RISK_REWARD:
        reasons.append(f"Target 2 risk/reward {plan['risk_reward_ratio']:.2f} is below {MIN_TARGET_RISK_REWARD:.2f}.")
    if regime.get("classification") == "TRANSITIONAL":
        reasons.append("Market regime is transitional; directional entry is withheld.")
    gate_failed = bool(reasons)
    if not gate_failed and abs(score) >= DECISION_THRESHOLD:
        decision = direction
        if active_direction in {"BUY", "SELL"} and active_direction == direction:
            decision = "HOLD"
            reasons.append("A same-direction hypothetical setup is already active; no duplicate entry is proposed.")
        elif active_direction in {"BUY", "SELL"}:
            decision = "WAIT FOR CONFIRMATION"
            reasons.append(f"An active hypothetical {active_direction} setup remains open; overlapping opposite positions are not proposed.")
        else:
            reasons.append(f"Directional score {score:+.3f}, regime evidence and risk/reward gates support a hypothetical {direction}.")
    elif not reasons:
        reasons.append("Evidence is mixed or does not meet the conservative action threshold.")
    if decision in {"WAIT FOR CONFIRMATION", "HOLD"}:
        # Do not expose a trade plan as actionable when a decision gate failed.
        if decision == "WAIT FOR CONFIRMATION":
            plan = {"valid": False, "reason": "No actionable setup is approved at this candle."}
    condition = ("A completed candle closes above resistance with relative volume >= 1.20x."
                 if bias == "BULLISH" else "A completed candle closes below support with relative volume >= 1.20x."
                 if bias == "BEARISH" else "Trend and momentum align beyond the configured score threshold.")
    invalidation = (f"Close beyond stop/invalidation level {plan.get('setup_invalidation_level'):.2f}."
                    if plan.get("valid") else "No active setup exists to invalidate.")
    return {
        "classification": decision, "directional_bias": bias,
        "technical_score": technical_score, "final_decision_score": score,
        "score_type": "heuristic_weighted_evidence_not_probability",
        "weights": {**WEIGHTS, "ml": WEIGHTS["ml"] if components["ml"] is not None else 0.0},
        "components": components, "bullish_evidence": bullish, "bearish_evidence": bearish,
        "normalized_component_weights": normalized_weights,"component_contributions": contributions,
        "dominant_contributors": [{"component":k,"contribution":contributions[k]} for k in dominant],
        "neutral_observations": neutral, "conflicting_signals": conflicts,
        "reasons": reasons, "required_confirmation": condition,
        "setup_invalidation": invalidation, "risk_plan": plan,
        "ml_evidence": ml, "market_regime": regime,
        "method": {"decision_threshold": DECISION_THRESHOLD, "strong_threshold": STRONG_DECISION_THRESHOLD,
                   "minimum_target2_rr": MIN_TARGET_RISK_REWARD, "conservative_gates": True},
    }
