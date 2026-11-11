"""Hypothetical, non-executable entry/stop/target calculations."""
from __future__ import annotations

import math

from backend.app.services.analysis.config import (
    ENTRY_ZONE_ATR, STOP_BUFFER_ATR, TARGET1_R_MULTIPLE, TARGET2_R_MULTIPLE,
)


def build_risk_plan(direction: str, price: float, atr: float, structure: dict) -> dict:
    if direction not in {"BUY", "SELL"}:
        return {"valid": False, "reason": "No directional setup; risk levels are not generated."}
    if not math.isfinite(price) or not math.isfinite(atr) or price <= 0 or atr <= 0:
        return {"valid": False, "reason": "A finite positive price and ATR are required."}
    long = direction == "BUY"
    entry_lower = price - atr * ENTRY_ZONE_ATR
    entry_upper = price + atr * ENTRY_ZONE_ATR
    support = structure.get("nearest_support")
    resistance = structure.get("nearest_resistance")
    if long:
        support_price = float(support["lower"]) if support and float(support["midpoint"]) < price else None
        swings = [z for z in structure.get("support_zones", []) if float(z["midpoint"]) < price]
        fallback = max([float(z["lower"]) for z in swings], default=price - atr * 1.25)
        stop = min(support_price if support_price is not None else fallback, price - atr * 0.25) - atr * STOP_BUFFER_ATR
        risk = price - stop
        targets = sorted(float(z["midpoint"]) for z in structure.get("resistance_zones", [])
                         if float(z["midpoint"]) > price)
        target1 = next((x for x in targets if x - price >= risk * TARGET1_R_MULTIPLE), price + risk * TARGET1_R_MULTIPLE)
        target2 = next((x for x in targets if x - price >= risk * TARGET2_R_MULTIPLE and x > target1),
                       max(target1, price + risk * TARGET2_R_MULTIPLE))
        invalidation = stop
    else:
        resistance_price = float(resistance["upper"]) if resistance and float(resistance["midpoint"]) > price else None
        swings = [z for z in structure.get("resistance_zones", []) if float(z["midpoint"]) > price]
        fallback = min([float(z["upper"]) for z in swings], default=price + atr * 1.25)
        stop = max(resistance_price if resistance_price is not None else fallback, price + atr * 0.25) + atr * STOP_BUFFER_ATR
        risk = stop - price
        targets = sorted((float(z["midpoint"]) for z in structure.get("support_zones", [])
                          if float(z["midpoint"]) < price), reverse=True)
        target1 = next((x for x in targets if price - x >= risk * TARGET1_R_MULTIPLE), price - risk * TARGET1_R_MULTIPLE)
        target2 = next((x for x in targets if price - x >= risk * TARGET2_R_MULTIPLE and x < target1),
                       min(target1, price - risk * TARGET2_R_MULTIPLE))
        invalidation = stop
    if risk <= 0 or not math.isfinite(risk):
        return {"valid": False, "reason": "No logically placed stop-loss can be calculated."}
    return {
        "valid": True, "hypothetical_only": True, "direction": direction,
        "trigger_price": price, "entry_zone": {"lower": entry_lower, "upper": entry_upper},
        "stop_loss": float(stop), "target_1": float(target1), "target_2": float(target2),
        "risk_per_share": float(risk), "distance_to_stop": float(risk),
        "target_1_risk_reward": float(abs(target1-price) / risk),
        "target_2_risk_reward": float(abs(target2-price) / risk),
        "risk_reward_ratio": float(abs(target2-price) / risk),
        "atr_reference": float(atr), "setup_invalidation_level": float(invalidation),
        "stop_method": "nearest_confirmed_zone_with_atr_buffer" if swings else "1.25_atr_fallback_with_atr_buffer",
        "target_method": "next_confirmed_opposite_zone_or_configured_R_multiple",
        "assumptions": {"signal_reference": "current completed candle close; not assumed to be an executable fill",
                        "historical_entry_evaluation": "next same-session candle open, only when it is inside the planned entry zone",
                        "stop_buffer_atr": STOP_BUFFER_ATR,
                        "target1_r": TARGET1_R_MULTIPLE, "target2_r": TARGET2_R_MULTIPLE},
    }
