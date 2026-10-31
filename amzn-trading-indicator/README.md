# AMZN Trading Indicator
ML-based 10-minute direction indicator for Amazon (AMZN).

## Architecture
- `data/`: local raw and processed market data (CSV files excluded from Git).
- `src/data/`: ingestion, validation, preprocessing and chronological splitting.
- `src/features/`: shared causal feature engineering for training and inference.
- `src/targets/`: target construction.
- `src/models/`: training and model selection (LightGBM first; LSTM/Lorentzian KNN experiments).
- `src/evaluation/`: metrics, leakage-safe evaluation and paper-trading simulation.
- `src/inference/`: shared live prediction pipeline.
- `backend/`: FastAPI service.
- `frontend/`: React + TypeScript + TradingView Lightweight Charts.
- `artifacts/`: local model artifacts (not committed by default).
- `configs/`, `tests/`, `reports/`, `docs/`.

## Rules
Never use `future_close_10m` or the target as model inputs. Split chronologically and purge boundary labels. Fit transformations on training data only. Share feature logic between training and inference. Finalized signals must never repaint. Keep credentials in `.env`.
See `docs/SETUP.md`. This is a scaffold; model and application logic will be implemented phase by phase.
