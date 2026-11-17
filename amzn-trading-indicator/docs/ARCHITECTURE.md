# System Architecture

## Overview

The **AMZN AI Trading Decision Terminal** is an institutional-grade, full-stack market analysis and decision-support system designed specifically for Amazon.com, Inc. (`NASDAQ: AMZN`). It bridges machine learning, multi-timeframe technical structure, and algorithmic risk management into an interactive desktop terminal.

```
                                  ┌────────────────────────┐
                                  │   Alpaca Market Data   │
                                  │   (IEX 1M/5M Stream)   │
                                  └───────────┬────────────┘
                                              │
                                              ▼
                                  ┌────────────────────────┐
                                  │    Market Ingestion    │
                                  │  & Reconciliation Loop │
                                  └───────────┬────────────┘
                                              │
                       ┌──────────────────────┴──────────────────────┐
                       ▼                                             ▼
        ┌─────────────────────────────┐               ┌─────────────────────────────┐
        │   SQLite Ingestion Buffer   │               │    Feature Engineering      │
        │  (market_stream.sqlite3)    │               │  40 Causal Intraday Features│
        └─────────────────────────────┘               └──────────────┬──────────────┘
                                                                     │
                       ┌─────────────────────────────────────────────┴────────────────────────┐
                       ▼                                                                      ▼
        ┌─────────────────────────────┐                                        ┌─────────────────────────────┐
        │  10-Minute LightGBM Model   │                                        │   Technical Indicator Suite │
        │  P(UP) / P(DOWN) Classifier │                                        │  EMA, SMA, RSI, MACD, VWAP  │
        └──────────────┬──────────────┘                                        └──────────────┬──────────────┘
                       │                                                                      │
                       └──────────────────────┬───────────────────────────────────────────────┘
                                              ▼
                               ┌─────────────────────────────┐
                               │   Multi-Timeframe Pivots    │
                               │  5M, 15M, 1H Anchor Swings  │
                               └──────────────┬──────────────┘
                                              ▼
                               ┌─────────────────────────────┐
                               │   Supply & Demand Engine    │
                               │  3 Major Supply / 5 Demand  │
                               └──────────────┬──────────────┘
                                              ▼
                               ┌─────────────────────────────┐
                               │ 60-Minute Decision Engine   │
                               │ Trend, Structure, Momentum, │
                               │ Volume, VWAP & Curfews      │
                               └──────────────┬──────────────┘
                                              ▼
                               ┌─────────────────────────────┐
                               │    Risk Management Plan     │
                               │  Trigger, Stop, Targets     │
                               └──────────────┬──────────────┘
                                              ▼
                               ┌─────────────────────────────┐
                               │  Persistent Database Layer  │
                               │   (forecast_log.sqlite3)    │
                               └──────────────┬──────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────┐
                               │      FastAPI REST & SSE     │
                               │    Uvicorn Port 8000        │
                               └──────────────┬──────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────┐
                               │    React 19 + TypeScript    │
                               │  Lightweight Charts UI      │
                               └─────────────────────────────┘
```

---

## Component Deep Dive

### 1. Ingestion & Streaming Layer
- **Provider**: Alpaca Data API (IEX free tier or SIP live feed).
- **Format**: 1-minute streaming bars aggregated into completed regular trading hours (RTH: 09:30–16:00 ET) 5-minute OHLCV candles.
- **Buffer Storage**: Persisted locally in `artifacts/market_stream.sqlite3`.
- **Reconciliation Loop**: Runs every 60 seconds via `backend/app/services/market_stream.py` to identify missing bars, backfill completed historical candles, and maintain a seamless chronological sequence without gaps.
- **Server-Sent Events (SSE)**: The endpoint `GET /api/market/events` streams live candle formations and market status events to the frontend in real time.

### 2. Feature Engineering & Signal Computation
- **Causal Calculation**: Features are derived exclusively using completed candles up to bar $t$. No future data, target leaks, or repainting indicators are permitted.
- **Indicator Suite**:
  - Exponential Moving Averages: EMA 9, EMA 20, EMA 50
  - Simple Moving Averages: SMA 20, SMA 50
  - Relative Strength Index: RSI 14 and RSI 7
  - Moving Average Convergence Divergence: MACD (12, 26, 9) line, signal line, histogram, and histogram acceleration
  - Volatility: ATR 14, Bollinger Bands (20, 2.0 std dev) with width and %B metrics
  - Volume Profiling: Session rolling volume, 20-period volume SMA, Relative Volume (RVOL), volume breakout confirmation
  - Anchored VWAP: Regular trading hours session VWAP with distance metrics and price rejection wick analysis.

### 3. Machine Learning Micro-Momentum Model
- **Artifact**: `artifacts/model_lightgbm.joblib`.
- **Input Dimension**: 40 ordered numerical features (`artifacts/feature_columns.json`).
- **Prediction Target**: Direction of the upcoming 10-minute price bar (`UP` vs `DOWN`).
- **Score Type**: Uncalibrated raw probability scores `P(UP)` and `P(DOWN)`. Clearly presented as micro-momentum confirmation rather than guaranteed execution confidence.

### 4. Supply & Demand Zone Engine
- **Multi-Timeframe Pivots**: 5M, 15M, and 1H pivots detected using swing confirmation (`SWING_LEFT = 2`, `SWING_RIGHT = 2`).
- **Wick-and-Body Anchoring**:
  - Supply (Resistance): Ceiling anchored at the highest swing wick; floor anchored at the real body top (`max(open, close)`).
  - Demand (Support): Floor anchored at the lowest swing wick; ceiling anchored at the real body bottom (`min(open, close)`).
- **Separation & Touch Filter**: Tracks price departures beyond an ATR threshold (`separation_buf = max(0.20, 0.40 * atr)`). Only distinct departures and returns or cross-session retests count toward independent touch totals.
- **Display Governance**: Displays strictly the **3 most significant supply zones** and **up to 5 demand zones** to preserve clean chart clarity.

### 5. 60-Minute Trading Decision Engine
- **Horizon**: Evaluates the upcoming 60-minute window (12 x 5-minute bars).
- **Multi-Evidence Synthesis**:
  - Trend Alignment (25% weight)
  - Market Structure & Zones (25% weight)
  - Momentum & Acceleration (20% weight)
  - Volume & RVOL (10% weight)
  - VWAP & Price Action (20% weight)
- **Decision Outputs**:
  - `BUY`: Strong positive confluence, confirmed structure, clear overhead runway, valid R:R $\ge 1.5$.
  - `SELL`: Strong negative confluence, confirmed structure, clear support runway, valid R:R $\ge 1.5$.
  - `WAIT FOR CONFIRMATION`: Directional bias established but blocked by immediate overhead/underfoot zone proximity or awaiting risk parameters.
  - `HOLD`: Rangebound, balanced, or active curfew conditions.
- **Session Curfews**:
  - Opening Discovery Curfew: 09:30–09:45 AM ET (holds new signals during opening volatility).
  - Closing Curfew: 15:15–16:00 PM ET (holds new 60m setups within 45 minutes of market close).

### 6. Persistence & Audit Layer
- Database: `artifacts/forecast_log.sqlite3`.
  - `analysis_signals`: Stores complete snapshot payloads, indicator dictionaries, and structure metadata.
  - `analysis_60m_signals`: Manages 60-minute signal lifecycles (`ACTIVE`, `RESOLVED_TARGET_1`, `RESOLVED_TARGET_2`, `RESOLVED_STOP_LOSS`, `EXPIRED`, `INVALIDATED`).
  - `ml_predictions`: Logs 10-minute ML predictions and evaluates realized 10-minute directional accuracy.

### 7. Frontend User Interface
- **Framework**: React 19 with TypeScript, bundled by Vite 6.
- **Charting**: TradingView Lightweight Charts v5.0 with custom HTML5 Canvas primitives for finite rectangular supply/demand zones and crosshair hover hitboxes.
- **Panels**:
  - Candlestick Chart with multi-timeframe overlays and finite zone boxes
  - Market Overview with multi-timeframe trend classifications
  - 60-Minute Trading Decision Panel with dynamic next-prediction countdown
  - 10-Minute ML Model Prediction card with raw probability bars
  - Algorithmic Signal Stream displaying historical audit trails and outcomes.
