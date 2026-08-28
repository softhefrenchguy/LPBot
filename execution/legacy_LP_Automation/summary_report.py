# summary_report.py (reconciliation version)
import json, re
from datetime import datetime

HISTORY_FILE = "cycle_history.jsonl"

def parse_jsonl(path):
    lines = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw or not raw.startswith("{"):  # skip console art
                continue
            try:
                obj = json.loads(raw)
                lines.append((i, obj))
            except Exception:
                # ignore non-json
                pass
    return lines

def is_valid_cycle(o):
    return ("timestamp_withdraw" in o and
            isinstance(o.get("fees_collected_usd"), (int, float)) and
            isinstance(o.get("pnl_usd"), (int, float)))

def key_for_dedup(o):
    # withdraw + create within 2 minutes considered same cycle snapshot
    return (o.get("timestamp_withdraw"), o.get("timestamp_create"))

def amount(o, k): return float(o.get(k, 0.0))

def fee_anomaly(o):
    fees = amount(o, "fees_collected_usd")
    pnl  = amount(o, "pnl_usd")
    # keep small fees; drop extreme one-offs
    return fees > 10.0 or fees < -0.01 or pnl < -10 or pnl > 10

def dt(s):
    # 'YYYY-MM-DD HH:MM:SS UTC'
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S UTC")

def summarize(tag, items):
    if not items:
        return {"count":0,"from":None,"to":None,"fees":0.0,"pnl":0.0,"net":0.0,"avg":0.0}
    fees = sum(amount(o,"fees_collected_usd") for o in items)
    pnl  = sum(amount(o,"pnl_usd") for o in items)
    net  = fees + pnl
    dts  = [dt(o["timestamp_withdraw"]) for o in items if "timestamp_withdraw" in o]
    return {
        "count": len(items),
        "from": min(dts).strftime("%Y-%m-%d %H:%M:%S UTC") if dts else None,
        "to":   max(dts).strftime("%Y-%m-%d %H:%M:%S UTC") if dts else None,
        "fees": fees, "pnl": pnl, "net": net,
        "avg": (net/len(items)) if items else 0.0
    }

def print_block(title, s):
    print(f"\n📊 {title}")
    print("="*39)
    print(f"📆 Period: {s['from']} → {s['to']}")
    print(f"🔁 Cycles: {s['count']}")
    print(f"💰 Fees:   {s['fees']:+.2f} USD")
    print(f"📉 PnL:    {s['pnl']:+.2f} USD")
    print(f"💹 Net:    {s['net']:+.2f} USD")
    print(f"⚙️ Avg/Cycle: {s['avg']:+.2f} USD")

def main():
    raw = parse_jsonl(HISTORY_FILE)
    raw_objs = [o for _, o in raw]

    # 0) RAW (best-effort JSON only)
    raw_summary = summarize("RAW", raw_objs)
    print_block("RAW totals (JSON only, no filters)", raw_summary)

    # 1) VALID FORM ONLY
    valid = [o for o in raw_objs if is_valid_cycle(o)]
    print_block("Valid cycles (has withdraw, fees, pnl)", summarize("VALID", valid))

    # 2) DEDUP
    seen = set(); dedup = []
    for o in valid:
        k = key_for_dedup(o)
        if k in seen:  # keep first appearance
            continue
        seen.add(k); dedup.append(o)
    print_block("After de-duplication", summarize("DEDUP", dedup))

    # 3) FILTER OUT ANOMALIES
    filtered = [o for o in dedup if not fee_anomaly(o)]
    print_block("After anomaly filter", summarize("FILTERED", filtered))

    # 4) FINAL (optional: ensure chronological, drop strictly worse duplicates)
    # Already deduped; if you want most-recent snapshot per withdraw time, flip the logic above.

    # Show what was removed
    removed = [o for o in dedup if fee_anomaly(o)]
    if removed:
        print("\n— Removed as anomalies —")
        for o in removed:
            print(f"withdraw={o.get('timestamp_withdraw')} fees={o.get('fees_collected_usd')} pnl={o.get('pnl_usd')}")

if __name__ == "__main__":
    main()


