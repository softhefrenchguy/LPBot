import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)
import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)
import json, os

IN_FILE = "cycle_history.jsonl"
OUT_FILE = "cycle_history_deduped.jsonl"

def key(e):
    return (
        e.get("timestamp_create"),
        e.get("timestamp_withdraw"),
        e.get("lower_bound_usd"),
        e.get("upper_bound_usd"),
    )

def score(e):
    s = 0
    if e.get("net_pnl_usd") is not None: s += 3
    if e.get("pnl_cycle_usd") is not None: s += 2
    if e.get("gas_cost_usd") is not None: s += 1
    return s

entries = []
with open(IN_FILE, "r", encoding="utf-8", errors="ignore") as f:
    for ln in f:
        ln = ln.strip()
        if not ln: continue
        try:
            entries.append(json.loads(ln))
        except Exception:
            pass

best = {}
for e in entries:
    k = key(e)
    cur = best.get(k)
    if cur is None or score(e) > score(cur):
        best[k] = e

with open(OUT_FILE, "w", encoding="utf-8") as f:
    for e in best.values():
        f.write(json.dumps(e) + "\n")

print(f"âœ… Wrote deduped file â†’ {OUT_FILE} ({len(best)} unique cycles)")
