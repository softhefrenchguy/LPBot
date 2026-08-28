import pandas as pd
import numpy as np

INPUT = "data/eth_usdc_1m.csv"

OUTPUT = "data/eth_usdc_features_1m.csv"


def compute_features(df):
    df = df.copy()

    # Basic returns
    df["log_return_1m"] = np.log(df["close"] / df["close"].shift(1))

    # EMAs for trend regime
    df["ema5"] = df["close"].ewm(span=5, adjust=False).mean()
    df["ema30"] = df["close"].ewm(span=30, adjust=False).mean()

    # EMA gap %
    df["ema_gap_pct"] = (df["ema5"] - df["ema30"]) / df["ema30"] * 100

    # EMA5 slope (per minute)
    df["ema5_slope"] = df["ema5"].diff() / df["ema5"].shift(1) * 100

    # Volatility (EWMA)
    df["vol_ewma_60"] = (
        df["log_return_1m"].ewm(span=60, adjust=False).std() * np.sqrt(60)
    )

    # Price acceleration
    df["return_sma_15"] = df["log_return_1m"].rolling(15).mean()
    df["return_std_15"] = df["log_return_1m"].rolling(15).std()

    # Drop NaN rows
    df = df.dropna().reset_index(drop=True)

    return df


def main():
    print("=== Generating ETH Features ===")

    df = pd.read_csv(INPUT)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values("timestamp")

    df_feat = compute_features(df)

    df_feat.to_csv(OUTPUT, index=False)
    print(f"[OK] Saved features → {OUTPUT}")
    print(df_feat.tail())


if __name__ == "__main__":
    main()
