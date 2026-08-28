import pandas as pd
import glob
import os

DATA_FOLDER = "data"

def load_crypto_csv(path):
    print(f"Loading {path} ...")

    # Skip first line (CryptoDataDownload's banner)
    raw = pd.read_csv(path, skiprows=1)

    # Normalize column names (lowercase, no spaces)
    raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]

    # Detect timestamp column
    if "date" in raw.columns:
        raw["timestamp"] = pd.to_datetime(raw["date"])
    else:
        raise ValueError(f"❌ No 'date' column found in {path}")

    # Detect volume column
    volume_col = None
    if "volume_eth" in raw.columns:
        volume_col = "volume_eth"
    elif "volume" in raw.columns:
        volume_col = "volume"
    else:
        raise ValueError(f"❌ No usable volume column in {path}")

    df = raw[[
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        volume_col
    ]].copy()

    df = df.rename(columns={volume_col: "volume"})
    return df


def main():
    print("=== Merging ETHUSDT minute CSV files ===")

    files = sorted(glob.glob(os.path.join(DATA_FOLDER, "*ETHUSDT*minute*.csv")))
    
    if not files:
        print("❌ No files found")
        return

    print("Found files:")
    for f in files:
        print(" -", f)

    dfs = []
    for f in files:
        dfs.append(load_crypto_csv(f))

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("timestamp")
    combined = combined.drop_duplicates(subset=["timestamp"])

    out_path = os.path.join(DATA_FOLDER, "eth_usdc_1m.csv")
    combined.to_csv(out_path, index=False)

    print("\n[OK] Saved merged result →", out_path)
    print("Total rows:", len(combined))


if __name__ == "__main__":
    main()

