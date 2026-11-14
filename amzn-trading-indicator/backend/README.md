FastAPI service for the dashboard, completed IEX candles, frozen window forecast issuance, post-window comparison, model metadata, and structured pipeline logs. Start it from the repository root with `python -m uvicorn backend.app.main:app --reload`. Alpaca credentials remain backend-only in `.env`.


FastAPI startup seeds the separate `artifacts/market_stream.sqlite3` store from the latest processed snapshot, starts one AMZN Alpaca IEX stream when backend credentials are configured, and reconciles actual completed 5-minute bars every minute. One-minute data can update a forming chart candle but never enters analysis. Market events are replayed through `/api/market/events`; stream health is exposed at `/api/market/stream/status`. The forecast scheduler remains a separate command.
