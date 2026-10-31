# Architecture
Offline: Alpaca historical data -> validation -> feature/target generation -> chronological split + purge -> experiments -> evaluation -> model artifact.
Online: Alpaca completed candles -> validation -> shared feature pipeline -> saved model -> confidence/signal policy -> FastAPI -> frontend chart.
Keep feature logic shared. Only completed candles generate finalized signals. Store timestamp, model version, prediction, confidence and data status. DBSCAN is optional regime analysis, not the main classifier.
