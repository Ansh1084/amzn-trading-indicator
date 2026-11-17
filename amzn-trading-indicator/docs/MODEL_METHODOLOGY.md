# Machine Learning & Decision Methodology

## 1. 10-Minute LightGBM Directional Classifier

### Overview
The production machine learning model is a LightGBM gradient-boosted decision tree binary classifier (`artifacts/model_lightgbm.joblib`) designed to forecast short-term price direction over the next 10 minutes (2 completed 5-minute bars).

### Feature Schema (40 Ordered Features)
Features are computed causally from completed 5-minute OHLCV candles without lookahead bias:

| Index | Feature Name | Description | Category |
|:---:|---|---|---|
| 0 | `return_1` | 1-bar percentage return ($t$ vs $t-1$) | Return Dynamics |
| 1 | `return_2` | 2-bar percentage return ($t$ vs $t-2$) | Return Dynamics |
| 2 | `return_3` | 3-bar percentage return ($t$ vs $t-3$) | Return Dynamics |
| 3 | `log_return_1` | 1-bar log return $\ln(C_t / C_{t-1})$ | Return Dynamics |
| 4 | `price_change` | Dollar price difference $(C_t - C_{t-1})$ | Return Dynamics |
| 5 | `candle_body` | Absolute dollar size of candle body $\|C_t - O_t\|$ | Candle Anatomy |
| 6 | `candle_range` | High-to-low candle span $(H_t - L_t)$ | Candle Anatomy |
| 7 | `upper_wick` | Length of top shadow $H_t - \max(O_t, C_t)$ | Candle Anatomy |
| 8 | `lower_wick` | Length of bottom shadow $\min(O_t, C_t) - L_t$ | Candle Anatomy |
| 9 | `body_to_range` | Ratio of candle body to range | Candle Anatomy |
| 10 | `sma_5` | 5-period Simple Moving Average of close | Moving Averages |
| 11 | `sma_10` | 10-period Simple Moving Average of close | Moving Averages |
| 12 | `sma_20` | 20-period Simple Moving Average of close | Moving Averages |
| 13 | `ema_5` | 5-period Exponential Moving Average | Moving Averages |
| 14 | `ema_10` | 10-period Exponential Moving Average | Moving Averages |
| 15 | `ema_20` | 20-period Exponential Moving Average | Moving Averages |
| 16 | `macd` | MACD Line ($12\text{ EMA} - 26\text{ EMA}$) | Momentum |
| 17 | `macd_signal` | MACD Signal Line ($9\text{ EMA of MACD}$) | Momentum |
| 18 | `macd_hist` | MACD Histogram ($\text{MACD} - \text{Signal}$) | Momentum |
| 19 | `rsi_14` | 14-period Relative Strength Index | Momentum |
| 20 | `roc_5` | 5-period Rate of Change percentage | Momentum |
| 21 | `roc_10` | 10-period Rate of Change percentage | Momentum |
| 22 | `atr_14` | 14-period Average True Range | Volatility |
| 23 | `bb_upper` | Bollinger Band Upper (20, 2.0 std dev) | Volatility |
| 24 | `bb_middle` | Bollinger Band Middle (20 SMA) | Volatility |
| 25 | `bb_lower` | Bollinger Band Lower (20, 2.0 std dev) | Volatility |
| 26 | `bb_width` | Normalized Bollinger Band Width | Volatility |
| 27 | `bb_position` | Percent b oscillator ($(C - \text{Lower}) / (\text{Upper} - \text{Lower})$) | Volatility |
| 28 | `volume_change` | 1-bar volume percentage change | Volume Dynamics |
| 29 | `volume_sma_5` | 5-period Simple Moving Average of volume | Volume Dynamics |
| 30 | `volume_sma_20` | 20-period Simple Moving Average of volume | Volume Dynamics |
| 31 | `volume_ratio_5` | Volume relative to 5-period volume SMA | Volume Dynamics |
| 32 | `volume_ratio_20` | Volume relative to 20-period volume SMA | Volume Dynamics |
| 33 | `volume_ratio` | Current volume relative to session mean volume | Volume Dynamics |
| 34 | `vwap_distance` | Percentage distance of price from Session VWAP | Price & Volume |
| 35 | `hour` | Hour of day (Eastern Time) | Temporal |
| 36 | `minute` | Minute of hour | Temporal |
| 37 | `day_of_week` | Day of week integer (0 = Monday, 4 = Friday) | Temporal |
| 38 | `time_of_day_sin`| Cyclical intraday time encoding ($\sin$) | Temporal |
| 39 | `time_of_day_cos`| Cyclical intraday time encoding ($\cos$) | Temporal |

### Uncalibrated Probability Scores
The model outputs raw sigmoid probability scores:
$$P(\text{UP}) \in [0, 1], \quad P(\text{DOWN}) = 1 - P(\text{UP})$$
- **Important**: These scores are raw, uncalibrated model outputs. They do not represent true empirical probabilities, Bayesian confidence intervals, or trade win rates.
- In the terminal, they function as **micro-momentum indicators**, providing short-term directional context to validate or question broader multi-timeframe setups.

---

## 2. 60-Minute Trading Decision Engine

### Objective
The 60-Minute Decision Engine operates on a 12-bar prediction horizon ($12 \times 5\text{-minute bars} = 60\text{ minutes}$). It evaluates multi-timeframe market evidence to answer the trader's core question:

> *"Considering current market structure, trend alignment, momentum, and risk parameters, should I BUY, SELL, or HOLD for the upcoming one-hour period?"*

### Synthesis Model (Weighted Confluence)
The system calculates a continuous composite score $S \in [-1.0, +1.0]$ across five evidence groups:

$$S = w_{\text{trend}} E_{\text{trend}} + w_{\text{struct}} E_{\text{struct}} + w_{\text{mom}} E_{\text{mom}} + w_{\text{vol}} E_{\text{vol}} + w_{\text{vwap}} E_{\text{vwap}}$$

Default weights (summing to 1.0):
- **Trend Alignment ($w = 0.25$)**: Confluence across 5M, 15M, and 1H trends (EMA 9/20/50 and SMA 20/50 slope and stacking).
- **Market Structure ($w = 0.25$)**: Proximity to 3 major supply zones and 5 demand zones, break of structure (BOS), and prior session highs/lows.
- **Momentum ($w = 0.20$)**: RSI 14/7 positioning, MACD histogram magnitude, and ROC acceleration.
- **Volume Confirmation ($w = 0.10$)**: Session RVOL, 20-period volume confirmation, and breakout volume verification.
- **VWAP & Price Action ($w = 0.20$)**: Distance from Session VWAP, candle body-to-range ratios, and upper/lower rejection wicks.

### Gating & Decision Rules
1. **`BUY` Recommendation**:
   - $S \ge +0.35$ (or $\ge +0.60$ for Strong Buy).
   - Price above Session VWAP or demonstrating clear bullish rejection.
   - Overhead runway to the nearest supply zone $\ge 0.35\text{ ATR}$.
   - Valid execution risk plan with $\text{Reward/Risk Ratio} \ge 1.5$.
2. **`SELL` Recommendation**:
   - $S \le -0.35$ (or $\le -0.60$ for Strong Sell).
   - Price below Session VWAP or demonstrating clear bearish rejection.
   - Downward runway to the nearest demand zone $\ge 0.35\text{ ATR}$.
   - Valid execution risk plan with $\text{Reward/Risk Ratio} \ge 1.5$.
3. **`WAIT FOR CONFIRMATION`**:
   - $|S| \ge 0.35$, but an invalidating constraint is triggered:
     - Proximity veto: Price is too close to a major opposing supply or demand zone ($< 0.35\text{ ATR}$).
     - Missing multi-timeframe trend alignment.
     - Insufficient reward-to-risk ratio ($< 1.5$).
4. **`HOLD` (Default)**:
   - $|S| < 0.35$ (neutral or conflicting market conditions).
   - Opening Discovery Curfew (09:30–09:45 AM ET).
   - Closing Curfew (15:15–16:00 PM ET).
