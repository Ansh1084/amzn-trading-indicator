
import pandas as pd

def load_and_filter_rth(filepath):
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['timestamp'] = df['timestamp'].dt.tz_convert('America/New_York')
    df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    df = df.set_index('timestamp')
    df = df.between_time('09:30', '15:55').reset_index()
    return df

def engineer_features(df):
    df = df.copy()
    df['trading_date'] = df['timestamp'].dt.date

    delta = df.groupby('trading_date')['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.groupby(df['trading_date']).transform(lambda x: x.rolling(14).mean())
    avg_loss = loss.groupby(df['trading_date']).transform(lambda x: x.rolling(14).mean())
    rs = avg_gain / avg_loss.replace(0, float('nan'))
    df['RSI_14'] = 100 - (100 / (1 + rs))

    ema12 = df.groupby('trading_date')['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = df.groupby('trading_date')['close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df['MACD_12_26_9'] = ema12 - ema26
    df['MACDs_12_26_9'] = df.groupby('trading_date')['MACD_12_26_9'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    df['MACDh_12_26_9'] = df['MACD_12_26_9'] - df['MACDs_12_26_9']

    sma20 = df.groupby('trading_date')['close'].transform(lambda x: x.rolling(20).mean())
    std20 = df.groupby('trading_date')['close'].transform(lambda x: x.rolling(20).std())
    df['BBL_20_2.0'] = sma20 - 2 * std20
    df['BBM_20_2.0'] = sma20
    df['BBU_20_2.0'] = sma20 + 2 * std20

    df['returns_1_period'] = df.groupby('trading_date')['close'].pct_change()
    df['returns_2_period'] = df.groupby('trading_date')['close'].pct_change(2)
    df['returns_3_period'] = df.groupby('trading_date')['close'].pct_change(3)
    df['volatility_10'] = df.groupby('trading_date')['returns_1_period'].transform(lambda x: x.rolling(10).std())
    df['volume_change'] = df.groupby('trading_date')['volume'].pct_change()

    return df


def generate_10min_target(df):
    df = df.copy()
    df['future_timestamp'] = df['timestamp'] + pd.Timedelta(minutes=10)
    future = df[['timestamp', 'close']].rename(columns={'timestamp': 'future_timestamp', 'close': 'future_close_10m'})
    df = df.merge(future, on='future_timestamp', how='left')
    same_day = df['timestamp'].dt.date == df['future_timestamp'].dt.date
    df.loc[~same_day, 'future_close_10m'] = float('nan')
    df['target_10m_UP'] = pd.Series(pd.NA, index=df.index, dtype='Int64')
    valid = df['future_close_10m'].notna()
    df.loc[valid, 'target_10m_UP'] = (df.loc[valid, 'future_close_10m'] > df.loc[valid, 'close']).astype(int)
    df = df.drop(columns=['future_timestamp'])
    return df

def chronological_split(df, train_ratio=0.7, val_ratio=0.15):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test


def main() -> None:
    df_raw = load_and_filter_rth("AMZN_data_1year.csv")
    df_features = engineer_features(df_raw)
    df_labeled = generate_10min_target(df_features)

    df_clean = df_labeled.dropna().reset_index(drop=True)

    train_set, val_set, test_set = chronological_split(df_clean)

    print("Total rows after cleaning:", len(df_clean))
    print("Training rows:", len(train_set))
    print("Validation rows:", len(val_set))
    print("Testing rows:", len(test_set))
    print("\nTarget distribution:")
    print(df_clean['target_10m_UP'].value_counts())
    print("\nTarget proportions:")
    print(df_clean['target_10m_UP'].value_counts(normalize=True).round(3))
    print("\nDate range:", df_clean['timestamp'].min(), "to", df_clean['timestamp'].max())

    df_clean.to_csv("AMZN_prepared.csv", index=False)
    train_set.to_csv("AMZN_train.csv", index=False)
    val_set.to_csv("AMZN_val.csv", index=False)
    test_set.to_csv("AMZN_test.csv", index=False)

    print("\nSaved AMZN_prepared.csv, AMZN_train.csv, AMZN_val.csv and AMZN_test.csv")


if __name__ == "__main__":
    main()
