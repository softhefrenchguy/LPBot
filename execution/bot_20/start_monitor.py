import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)
import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)
# start_monitor.py
import subprocess, time, json, os, pandas as pd, matplotlib
import matplotlib.pyplot as plt
from datetime import datetime

# Use non-GUI backend so nothing pops up
matplotlib.use("Agg")

HISTORY_FILE = "portfolio_history.jsonl"
GRAPH_FILE = "portfolio_graph.png"

def update_graph():
    """Generate/update the PNG graph from portfolio history."""
    try:
        if not os.path.exists(HISTORY_FILE):
            return
        data = [json.loads(line) for line in open(HISTORY_FILE) if line.strip()]
        if not data:
            return
        df = pd.DataFrame(data)
        if "timestamp" not in df.columns:
            return
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)

        plt.figure(figsize=(10, 5))
        plt.plot(df["timestamp"], df["actual_value_usd"], label="Portfolio (Actual)", color="orange", linewidth=2)
        if "hold_value_usd" in df.columns:
            plt.plot(df["timestamp"], df["hold_value_usd"], label="Hold Baseline", color="blue", linestyle="--")
        plt.xlabel("Time (UTC)")
        plt.ylabel("Portfolio Value (USD)")
        plt.title("Portfolio Performance Over Time")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(GRAPH_FILE)
        plt.close()
        print(f"ðŸ“Š Graph updated â†’ {GRAPH_FILE}")
    except Exception as e:
        print(f"âš ï¸ Graph update failed: {e}")

def run_tracker():
    """Run the portfolio tracker once."""
    try:
        subprocess.run(["python", "portfolio_tracker_plus.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"âŒ Tracker run failed: {e}")

def main():
    print("ðŸš€ Starting safe hourly monitor (no new windows).")
    print("ðŸ•’ Tracker every 10 minutes; graph auto-updates silently.\n")
    while True:
        print(f"â° Running tracker... ({datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')})")
        run_tracker()
        update_graph()
        print("âœ… Cycle complete â€” sleeping 10 minutes.\n")
        time.sleep(600)

if __name__ == "__main__":
    main()
