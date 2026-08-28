import json
from datetime import datetime

def log_cycle(cycle_data: dict):
    # Ensure timestamps exist
    cd = dict(cycle_data)
    cd.setdefault("logged_at", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))

    # 1) Append to JSONL history (one line per cycle)
    with open("cycle_history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(cd) + "\n")

    # 2) Update last snapshot (merged)
    try:
        with open("last_cycle_data.json", "r") as f:
            snap = json.load(f)
    except FileNotFoundError:
        snap = {}
    snap.update(cd)
    with open("last_cycle_data.json", "w") as f:
        json.dump(snap, f, indent=2)

    print("📝 Cycle recorded.")
