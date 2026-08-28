import os
import json
import statistics
from datetime import datetime

from prettytable import PrettyTable

HISTORY_FILE = "cycle_history.jsonl"

def load_cycles():
    if not os.path.exists(HISTORY_FILE):
        print("⚠️ No cycle_history.jsonl found.")
        return []
    rows = []
    with open(HISTORY_FILE, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows

def parse_dt(s):
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return None

def fmt(v, prec=4):
    if v is None:
        return "-"
    try:
        return f"{float(v):.{prec}f}"
    except Exception:
        return "-"

def main():
    data = load_cycles()
    if not data:
        print("⚠️ No data to display.")
        return

    tbl = PrettyTable()
    tbl.field_names = [
        "#",
        "Created (UTC)",
        "Withdrawn (UTC)",
        "Dur (min)",
        "Value In",
        "Value Out",
        "Pos PnL",
        "Gas",
        "Net PnL",
    ]
    tbl.align = "r"

    pnl_list = []
    gas_list = []
    net_list = []
    dur_list = []

    for row in data:
        cnum = row.get("cycle_number")
        t1 = row.get("timestamp_create")
        t2 = row.get("timestamp_withdraw")

        d1 = parse_dt(t1) if t1 else None
        d2 = parse_dt(t2) if t2 else None
        dur = None
        if d1 and d2:
            dur = (d2 - d1).total_seconds() / 60.0

        value_in = row.get("value_in_usd")
        value_out = row.get("value_out_usd")
        pnl_pos = row.get("pnl_position_usd")
        gas = row.get("gas_cost_usd")
        net = row.get("net_pnl_usd")

        if pnl_pos is not None:
            pnl_list.append(float(pnl_pos))
        if gas is not None:
            gas_list.append(float(gas))
        if net is not None:
            net_list.append(float(net))
        if dur is not None:
            dur_list.append(dur)

        tbl.add_row([
            cnum if cnum is not None else "-",
            t1.split(" ")[1] if t1 else "-",
            t2.split(" ")[1] if t2 else "-",
            fmt(dur, 2) if dur is not None else "-",
            fmt(value_in, 4),
            fmt(value_out, 4),
            fmt(pnl_pos, 4),
            fmt(gas, 4),
            fmt(net, 4),
        ])

    print("\n====================================================")
    print("📘 LP CYCLE HISTORY — Uniswap V3 (Position-only PnL)")
    print("====================================================")
    print(tbl)
    print("----------------------------------------------------")
    print(f"🏁 Total cycles: {len(data)}")
    if dur_list:
        print(f"⏱  Avg duration: {statistics.mean(dur_list):.2f} min")
    print(f"💰 Total Pos PnL: {sum(pnl_list):+.4f} USDC")
    print(f"⛽ Total Gas:     {sum(gas_list):.4f} USDC")
    print(f"🧮 Net Profit:    {sum(net_list):+.4f} USDC")
    if net_list:
        print(f"📊 Avg Net/Cycle: {statistics.mean(net_list):+.4f} USDC")
    print("====================================================\n")

if __name__ == "__main__":
    main()
