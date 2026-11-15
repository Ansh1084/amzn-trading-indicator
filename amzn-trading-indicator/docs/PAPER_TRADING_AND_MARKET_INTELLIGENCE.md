# Paper Trading and Market Intelligence

This feature set is analysis-only. It does not connect to a broker or submit orders. Paper trading starts **disabled** and stores simulated trades in `artifacts/paper_trading.sqlite3`, separate from the analysis outcome and historical research stores.

## Paper account

The account starts with a configurable $100,000 balance, one fixed share per signal, 1 basis point of transaction costs per side and 2 basis points of slippage per side. These are simulation defaults, not measured Alpaca or broker fees. Settings are frozen onto each trade so later settings changes do not rewrite its costs.

An entry is queued only for a fresh, persisted BUY/SELL analysis with a valid directional risk plan. HOLD, WAIT, stale or invalid analyses are ignored. The signal candle is never an entry candle. The first candle immediately after the signal must be a completed candle from the same session, and its open must be inside the risk plan's entry zone; otherwise the entry expires without a fill. Missing immediate-next candles are not replaced with later candles.

The entry reference is the next candle's actual open. The simulated fill applies adverse configured slippage. `entry_price` and `exit_price` hold raw observed reference prices; `entry_fill_price` and `exit_fill_price` hold the slippage-adjusted simulation fills. This keeps accounting explicit:

```text
LONG gross P&L  = (raw exit reference - raw entry reference) × quantity
SHORT gross P&L = (raw entry reference - raw exit reference) × quantity
Net P&L         = gross P&L - transaction costs - slippage costs
```

Transaction costs are charged on simulated fill notional. Slippage cost is the absolute difference between each raw reference and its simulated fill, multiplied by the executed quantity. Position sizing supports fixed shares, fixed capital or a percentage of account equity, capped by available account capital. Invalid quantity, entry zone or stop/target ordering is rejected.

On each genuine completed 5-minute candle, the engine checks stop before targets. If the stop and target are both touched in one candle, the stop wins. A gap through a stop fills at the worse candle open; target gaps fill at the target level without favorable price improvement. Target 1 closes half the original quantity once; the remainder exits at stop, target 2, the regular-session close or the 180-minute time limit. Open positions continue to be managed while new entries are paused. Queued, unfilled entries are cancelled when pausing.

Manual close uses the last completed stored candle close and timestamp. The interface labels this as the last-known mark and shows freshness/market state; it is not represented as a live quote. Unrealized P&L subtracts paid entry costs and estimated exit costs. The equity curve and maximum drawdown use realized net P&L at completed trade exits; they do not model intrabar account drawdown. Profit factor and win rate are based on fully closed trades.

## Historical replay

`backend.app.services.paper_trading.replay_historical()` can replay persisted analysis records against supplied completed candles. It uses an isolated temporary SQLite file and cannot mix its results into the paper account. It does not manufacture signals. The current local ledger contains 22 stored analysis records and no actionable BUY/SELL signals; therefore the genuine local-history replay has zero trades and no meaningful win rate or profit factor. Do not interpret zero-trade totals as evidence of strategy performance.

## Market intelligence

The `GET /api/analysis/market-intelligence` endpoint uses locally persisted completed regular-session candles, bounded by the request-time cutoff. No incomplete or future candle is included.

- **5M, 15M and 1H trends:** 15-minute and 1-hour bars are derived from complete groups of 5-minute candles anchored at 09:30 ET. Partial groups are discarded. Trend is BULLISH when close > EMA(9) > EMA(21) and EMA(21) rose over three completed bars; BEARISH is the inverse; otherwise it is NEUTRAL. At least 21 aggregate bars are required.
- **Opening range:** high, low and midpoint of the three completed 5-minute bars from 09:30–09:45 ET. Subsequent completed closes describe breakout, breakdown, retest, failed break or inside-range state. This context does not change the decision or its weights.
- **RVOL:** current completed 5-minute volume divided by the average volume in the same ET 5-minute slot over up to 20 previous trading sessions. At least three matching historical sessions are required. Lookback and classification thresholds are in `configs/market_intelligence.json`. RVOL is shown as breakout context, never as an independent trade signal.
- **Previous-session references:** high, low and last completed close from the exact previous scheduled trading session. Only regular-session candles are included. If that session is absent, the levels are explicitly unavailable.

Chart toggles show opening-range high/low/midpoint, previous-session high/low and previous-session close. The new evidence is explanatory only; production model features, decision weights and thresholds are unchanged.

## Start the application

Backend (one Windows CMD window):

```bat
cd /d "C:\Users\anshg\Desktop\SEM5\Projects\ML2 - Copy\amzn-trading-indicator"
call .venv\Scripts\activate.bat
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Frontend (second CMD window):

```bat
cd /d "C:\Users\anshg\Desktop\SEM5\Projects\ML2 - Copy\amzn-trading-indicator\frontend"
npm run dev -- --host 127.0.0.1 --port 5173
```

Paper-account settings and history are available under **Paper trading**. Enable the simulator explicitly after checking sizing and cost assumptions. Market-intelligence output appears in the terminal sidebar; the chart line toggles are in its overlay toolbar.
