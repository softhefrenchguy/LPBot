import pandas as pd

paths = [
    "data/Binance_ETHUSDT_2023_minute.csv",
    "data/Binance_ETHUSDT_2024_minute.csv",
    "data/Binance_ETHUSDT_2025_minute.csv",
]

for path in paths:
    print("\n==============================")
    print("FILE:", path)
    print("==============================")
    df = pd.read_csv(path, nrows=5)
    print("COLUMNS:", df.columns.tolist())
    print(df.head())
