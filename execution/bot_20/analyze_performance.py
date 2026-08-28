import json
import os
from collections import defaultdict

CYCLE_HISTORY_FILE = "cycle_history.jsonl"


def load_rows(path=CYCLE_HISTORY_FILE):
    rows = []
    if not os.path.exists(path):
        print(f"❌ {path} not found.")
        return rows

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def f(d, key, default=0.0):
    v = d.get(key, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def analyze(rows):
    # Split into position cycles vs rebalances
    position_cycles = [r for r in rows if "cycle_number" in r]
    rebalances = [r for r in rows if "timestamp_rebalance" in r]

    print(f"\n📊 Loaded {len(rows)} records "
          f"({len(position_cycles)} position cycles, {len(rebalances)} rebalances)\n")

    # ---------- 1) Position cycles ----------
    if not position_cycles:
        print("No position cycles (with cycle_number) found.")
    else:
        pnls = [f(r, "net_pnl_usd") for r in position_cycles]
        total = sum(pnls)
        avg = total / len(pnls)
        wins = sum(1 for x in pnls if x > 0)
        losses = sum(1 for x in pnls if x < 0)

        print("1️⃣ Position cycles (LP in/out only)")
        print(f"   Cycles: {len(pnls)}")
        print(f"   Total net PnL: {total:.2f} USDC")
        print(f"   Avg per cycle: {avg:.2f} USDC")
        print(f"   Wins: {wins} | Losses: {losses}")
        print(f"   Best: {max(pnls):.2f} | Worst: {min(pnls):.2f}\n")

    # ---------- 2) Rebalances ----------
    if not rebalances:
        print("No rebalances (timestamp_rebalance) found.")
    else:
        by_type = defaultdict(list)
        for r in rebalances:
            t = r.get("type", "unknown")
            by_type[t].append(f(r, "net_usd"))

        print("2️⃣ Rebalances (incl. panic)")
        total_rb = 0.0
        for t, pnls in by_type.items():
            t_total = sum(pnls)
            t_avg = t_total / len(pnls)
            total_rb += t_total
            print(
                f"   {t:15s} → {len(pnls):3d} events | "
                f"total {t_total:7.2f} | avg {t_avg:6.2f} USDC"
            )
        print(f"   Total rebalance net PnL: {total_rb:.2f} USDC\n")

    # ---------- 3) Combined ----------
    total_pos = sum(f(r, "net_pnl_usd") for r in position_cycles)
    total_rb = sum(f(r, "net_usd") for r in rebalances)
    total_all = total_pos + total_rb

    print("3️⃣ Combined PnL")
    print(f"   Positions total:   {total_pos:8.2f} USDC")
    print(f"   Rebalances total:  {total_rb:8.2f} USDC")
    print(f"   ➜ Overall total:   {total_all:8.2f} USDC\n")

    # ---------- 4) Simple 'AI coach' feedback ----------
    print("🧠 Strategy Assistant — Feedback\n" + "-" * 40)

    if position_cycles:
        avg_pos = total_pos / len(position_cycles)
        if avg_pos > 0:
            print(f"• Your *core LP cycles* are profitable "
                  f"(avg {avg_pos:.2f} USDC per cycle). That’s good.")
        else:
            print(f"• Your *core LP cycles* are not profitable "
                  f"(avg {avg_pos:.2f} USDC per cycle). "
                  "The range logic itself may need work.")

    if rebalances:
        if "panic_100_usdc" in by_type:
            panic_pnls = by_type["panic_100_usdc"]
            panic_total = sum(panic_pnls)
            panic_avg = panic_total / len(panic_pnls)
            print(
                f"• Panic cycles (panic_100_usdc) total: {panic_total:.2f} "
                f"(avg {panic_avg:.2f})."
            )
            if panic_total < 0:
                print("  → Panic exits are currently a drag on PnL.")
                print("    Consider: slightly wider base range or tighter panic trigger "
                      "so you don't fire panic so often.")

        if "rebalance_50_50" in by_type:
            normal_pnls = by_type["rebalance_50_50"]
            normal_total = sum(normal_pnls)
            normal_avg = normal_total / len(normal_pnls)
            print(
                f"• Normal 50/50 rebalances total: {normal_total:.2f} "
                f"(avg {normal_avg:.2f})."
            )
            if normal_total < 0:
                print("  → The 50/50 balancing itself is costing you (fees + slippage + gas).")
                print("    Possible tweaks: lower rebalance frequency, larger drift threshold, "
                      "or slightly wider LP ranges so you don't exit as often.")

    if total_all < 0:
        print("\n⚠️ Overall you're losing money so far.")
        print("   Short-term focus:")
        print("   - Make sure panic isn't triggering on small moves.")
        print("   - Check if very short cycles (e.g. <10–15 min) are profitable; "
              "if not, slow the system down.")
    else:
        print("\n✅ Overall you're net positive.")
        print("   Focus on leaning into the parameters (durations / volatility regimes) "
              "that give you the highest avg net PnL.")

    print("\n(We can later extend this to auto-tune parameters based on these stats.)")


if __name__ == "__main__":
    rows = load_rows()
    if not rows:
        print("No data found in cycle_history.jsonl.")
    else:
        analyze(rows)
