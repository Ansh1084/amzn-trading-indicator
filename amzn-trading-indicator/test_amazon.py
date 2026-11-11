
import os
import pandas as pd
from datetime import datetime, timedelta, timezone
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed


def main() -> None:
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise ValueError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY first.")

    client = StockHistoricalDataClient(api_key, secret_key)

    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=365)
    cursor = start
    all_bars = []

    while cursor < end:
        chunk_end = min(cursor + timedelta(days=30), end)
        print(f"Downloading {cursor.date()} to {chunk_end.date()}...")

        request = StockBarsRequest(
            symbol_or_symbols=["AMZN"],
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=cursor,
            end=chunk_end,
            limit=10000,
            feed=DataFeed.IEX
        )

        bars = client.get_stock_bars(request).df

        if not bars.empty:
            all_bars.append(bars)

        cursor = chunk_end

    if not all_bars:
        raise ValueError("No data returned. Check your API credentials and data access.")

    df = pd.concat(all_bars).reset_index()
    df = df.drop_duplicates(subset=["timestamp", "symbol"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.to_csv("AMZN_data_1year.csv", index=False)

    print("\nDownload completed!")
    print("Total candles:", len(df))
    print("Start:", df["timestamp"].min())
    print("End:", df["timestamp"].max())
    print("Missing values:\n", df.isnull().sum())
    print("Saved to AMZN_data_1year.csv")


if __name__ == "__main__":
    main()
