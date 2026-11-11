# Architecture
Training: Alpaca historical IEX bars -> existing validation/preprocessing -> existing causal features -> window-level outcomes -> expanding chronological folds -> per-evaluation-date frozen model artifact and metadata.

Issuance: Completed Alpaca 5-minute bars -> stale/gap/cutoff checks -> shared causal feature pipeline -> exact evaluation-date model -> window prediction -> causal swing levels and conditional historical support frequency -> immutable SQLite forecast -> structured pipeline logs -> FastAPI -> Lightweight Charts UI.

After a window ends, the comparison route verifies that the complete set of actual 5-minute bars exists and appends outcome metrics once. The original forecast payload is retained unchanged. Existing 10-minute experiments remain a separate legacy path.

The market-data API polls Alpaca historical bars and discards a still-forming candle. It currently does not keep a streaming connection open. The NYSE holiday helper covers standard holidays and recurring early closes, but not emergency calendar changes. Forecast windows are in `configs/forecast_windows.json`.
