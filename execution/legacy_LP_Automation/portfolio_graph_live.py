# portfolio_graph_live.py
import json
import time
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime

# Use a non-GUI backend (prevents Windows crashes)
matplotlib.use("Agg")

HISTORY_FILE = "portfolio_history.jsonl"
GRAPH_FILE = "portfolio_graph.png"

def load_data():
    """Load JSONL portfolio history into a pandas DataFrame."""
    try:
        with open(HISTORY_FILE, "r") as f:
            data = [json.loads(line) for line in f if line.strip()]
        if not data:
            print("⚠️ No data found in history file yet.")
            return pd.DataFrame()
        df = pd.DataFrame(data)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
        return df
    except Exception as e:
        print(f"⚠️ Failed to load history: {e}")
        return pd.DataFrame()

def plot_graph():
    """Draw and save the graph from historical portfolio data."""
    df = load_data()
    if df.empty:
        return

    plt.figure(figsize=(10, 5))

    # Always plot actual portfolio value
    if "actual_value_usd" in df.columns:
        plt.plot(df["timestamp"], df["actual_value_usd"],
                 label="Portfolio (Actual)", color="orange", linewidth=2)
    else:
        print("⚠️ No 'actual_value_usd' column found — skipping plot.")
        return

    # Plot baseline if present
    if "hold_value_usd" in df.columns:
        plt.plot(df["timestamp"], df["hold_value_usd"],
                 label="Hold Baseline", color="blue", linestyle="--")

    # Add labels and style
    plt.xlabel("Time (UTC)")
    plt.ylabel("Portfolio Value (USD)")
    plt.title("Portfolio Performance Over Time")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Save graph as PNG
    plt.savefig(GRAPH_FILE)
    plt.close()
    print(f"📊 Graph updated → {GRAPH_FILE}")

if __name__ == "__main__":
    print("📈 Live graph started (auto-refresh every 60s, headless safe).")
    while True:
        try:
            plot_graph()
            time.sleep(60)
        except Exception as e:
            print(f"⚠️ Graph update failed: {e}")
            time.sleep(60)
