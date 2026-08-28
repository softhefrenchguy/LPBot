import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)
import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)
# summary_report.py
import json
from datetime import datetime

FILE = "cycle_history.jsonl"

def load_cycles(file_path):
    cycles = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                cycles.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return cycles

def main():
    cycles = load_cycles(FILE)
    if not cycles:
        print("âš ï¸ No cycle data found.")
        return

    total_fees = sum(c.get("fees_collected_usd", 0) for c in cycles)
    total_pnl = sum(c.get("pnl_usd", 0) for c in cycles)
    total_profit = sum(c.get("cycle_profit_usd", 0) for c in cycles)
    avg_profit = total_profit / len(cycles)
    start_time = cycles[0].get("timestamp_create", "N/A")
    end_time = cycles[-1].get("timestamp_withdraw", "N/A")

    print("=======================================")
    print("ðŸ“Š LP Summary Report")
    print("=======================================")
    print(f"ðŸ“† Period: {start_time} â†’ {end_time}")
    print(f"ðŸ” Total Cycles:   {len(cycles)}")
    print(f"ðŸ’° Total Fees:     +${total_fees:.2f}")
    print(f"ðŸ“‰ Total PnL:      {total_pnl:+.2f} USD")
    print(f"ðŸ’¹ Net Profit:     {total_profit:+.2f} USD")
    print(f"âš™ï¸  Avg Profit/Cycle: {avg_profit:+.2f} USD")
    print("=======================================")

if __name__ == "__main__":
    main()
