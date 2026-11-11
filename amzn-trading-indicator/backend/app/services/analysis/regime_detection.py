"""Causal trend/volatility regime estimates with explicit evidence."""
from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_regime_series(indicators: pd.DataFrame) -> pd.DataFrame:
    """Compute each bar's causal ADX/regime once; useful for chronological replay."""
    x=indicators
    prev_close=x.close.shift()
    up=x.high.diff(); down=-x.low.diff()
    plus_dm=up.where((up>down)&(up>0),0.0)
    minus_dm=down.where((down>up)&(down>0),0.0)
    tr=pd.concat([(x.high-x.low),(x.high-prev_close).abs(),(x.low-prev_close).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/14,min_periods=14,adjust=False).mean()
    plus_di=100*plus_dm.ewm(alpha=1/14,min_periods=14,adjust=False).mean()/atr.replace(0,np.nan)
    minus_di=100*minus_dm.ewm(alpha=1/14,min_periods=14,adjust=False).mean()/atr.replace(0,np.nan)
    dx=100*(plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,np.nan)
    adx=dx.ewm(alpha=1/14,min_periods=14,adjust=False).mean()
    up_aligned=(x.ema_9>x.ema_20)&(x.ema_20>x.ema_50)
    down_aligned=(x.ema_9<x.ema_20)&(x.ema_20<x.ema_50)
    up_slope=(x.ema_20_slope_5>0)&(x.ema_50_slope_10>0)
    down_slope=(x.ema_20_slope_5<0)&(x.ema_50_slope_10<0)
    high_vol=x.volatility_classification.isin(["HIGH","EXTREME"])
    low_vol=x.volatility_classification.eq("LOW")
    ready=adx.notna()&x.volatility_percentile.notna()
    classes=np.select([~ready,high_vol,low_vol,(adx>=25)&up_aligned&up_slope,
                       (adx>=25)&down_aligned&down_slope,adx<20],
                      ["TRANSITIONAL","HIGH_VOLATILITY","LOW_VOLATILITY","TRENDING_UP","TRENDING_DOWN","SIDEWAYS"],
                      default="TRANSITIONAL")
    return pd.DataFrame({"adx_14":adx,"plus_di_14":plus_di,"minus_di_14":minus_di,
                         "classification":classes,"volatility_classification":x.volatility_classification,
                         "volatility_percentile":x.volatility_percentile,"trend_alignment":x.trend_alignment},index=x.index)


def detect_regime(indicators: pd.DataFrame, computed_series: pd.DataFrame | None = None) -> dict:
    if len(indicators) < 30:
        return {"classification": "TRANSITIONAL", "status": "INSUFFICIENT_DATA",
                "evidence": ["At least 30 completed candles are required for regime context."]}
    x = indicators
    series=computed_series if computed_series is not None else calculate_regime_series(x)
    row=series.loc[x.index[-1]]
    adx=row.adx_14
    last = x.iloc[-1]
    aligned_up = bool(last.ema_9 > last.ema_20 > last.ema_50)
    aligned_down = bool(last.ema_9 < last.ema_20 < last.ema_50)
    slope_up = bool(last.ema_20_slope_5 > 0 and last.ema_50_slope_10 > 0)
    slope_down = bool(last.ema_20_slope_5 < 0 and last.ema_50_slope_10 < 0)
    high_vol = bool(row.volatility_classification in {"HIGH", "EXTREME"})
    low_vol = bool(row.volatility_classification == "LOW")
    evidence = [f"ADX(14)={float(adx):.2f}" if pd.notna(adx) else "ADX is warming up.",
                f"EMA alignment: {'up' if aligned_up else 'down' if aligned_down else 'mixed'}.",
                f"EMA 20/50 slope: {'up' if slope_up else 'down' if slope_down else 'mixed'}.",
                f"ATR volatility percentile: {float(row.volatility_percentile):.2f} ({row.volatility_classification})."
                if pd.notna(row.volatility_percentile) else "Volatility percentile is warming up."]
    if pd.isna(adx) or pd.isna(row.volatility_percentile):
        classification = "TRANSITIONAL"
    elif high_vol:
        classification = "HIGH_VOLATILITY"
    elif low_vol:
        classification = "LOW_VOLATILITY"
    elif adx >= 25 and aligned_up and slope_up:
        classification = "TRENDING_UP"
    elif adx >= 25 and aligned_down and slope_down:
        classification = "TRENDING_DOWN"
    elif adx < 20:
        classification = "SIDEWAYS"
    else:
        classification = "TRANSITIONAL"
    return {"classification": classification, "status": "READY",
            "evidence": evidence, "adx_14": float(adx) if pd.notna(adx) else None,
            "plus_di_14": float(row.plus_di_14) if pd.notna(row.plus_di_14) else None,
            "minus_di_14": float(row.minus_di_14) if pd.notna(row.minus_di_14) else None,
            "trend_alignment": int(last.trend_alignment),
            "volatility_classification": str(row.volatility_classification),
            "volatility_percentile": float(row.volatility_percentile) if pd.notna(row.volatility_percentile) else None,
            "rules": {"trend_adx_min": 25, "sideways_adx_max": 20,
                      "volatility_percentile_high": .80, "low": .20, "extreme": .95}}
