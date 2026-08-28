import os
import csv
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def load_hourly_volumes(csv_name: str = "eth_usdc_005_hourly_volume.csv"):
    """
    Load historical hourly volume (USD) from a CSV file.
    Expected columns: timestamp, volume_usd
    """
    path = os.path.join(DATA_DIR, csv_name)
    vols = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                v = float(row["volume_usd"])
                if v > 0:
                    vols.append(v)
            except Exception:
                continue

    if not vols:
        raise RuntimeError(f"No valid volume_usd rows found in {path}")

    return np.array(vols, dtype=float)


def sample_volume_path(n_steps: int, vols_hist: np.ndarray, rng: np.random.Generator):
    """
    Bootstrap sampling: for each step, pick a random historical hourly volume.
    This keeps the real distribution of volumes.
    """
    idx = rng.integers(0, len(vols_hist), size=n_steps)
    return vols_hist[idx]
