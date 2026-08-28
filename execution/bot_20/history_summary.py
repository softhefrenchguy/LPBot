# history_summary.py
import json
import os
from datetime import datetime

HISTORY_FILE = "cycle_history.jsonl"

if not os.path.exists(HISTORY_FILE):
    print("No cycle_history.jsonl found.")
    raise SystemExit

durations = []
net_pnls = []
rows = []

with open(HISTORY_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except Exception:
            continue

        cycle = entry.get("cycle_number")
        t0 = entry.get("timestamp_create")
        t1 = entry.get("timestamp_withdraw") or entry.get("timestamp_rebalance")
        dur = entry.get("duration_minutes")
        net = entry.get("net_pnl_usd") or entry.get("net_usd")

        durations.append(dur if dur is not None else 0)
        if net is not None:
            net_pnls.append(net)

        rows.append((cycle, t0, t1, dur, net))

print("Cycle | duration(min) | net_pnl_usd | created_at")
print("------------------------------------------------")
for cycle, t0, t1, dur, net in rows[-50:]:  # last 50 entries
    d_str = f"{dur:.2f}" if dur is not None else "-"
    n_str = f"{net:+.4f}" if net is not None else "-"
    print(f"{cycle or '-':>5} | {d_str:>12} | {n_str:>11} | {t0}")

if durations:
    avg_dur = sum(d for d in durations if d) / max(1, len([d for d in durations if d]))
    print("\n--------------------------------------")
    print(f"Total entries: {len(rows)}")
    print(f"Avg duration:  {avg_dur:.2f} min")
if net_pnls:
    total_net = sum(net_pnls)
    avg_net = total_net / len(net_pnls)
    print(f"Total net PnL: {total_net:+.4f} USDC")
    print(f"Avg net/cycle: {avg_net:+.4f} USDC")
