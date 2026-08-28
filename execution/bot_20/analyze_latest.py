import json
import os
from collections import defaultdict

HISTORY_FILE = "cycle_history.jsonl"
SEEN_FILE = "seen_records.json"


def load_seen_count():
    """
    Load how many lines of cycle_history.jsonl we've already analysed.
    """
    if not os.path.exists(SEEN_FILE):
        return 0
    try:
        with open(SEEN_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return int(data.get("seen_count", 0))
    except Exception:
        return 0


def save_seen_count(count: int):
    """
    Save how many lines we've processed so far.
    """
    with open(SEEN_FILE, "w", encoding="utf-8") as f:
        json.dump({"seen_count": count}, f)


def load_new_records(seen_count: int):
    """
    Read cycle_history.jsonl and return only the new lines since `seen_count`.
    Also return total line count so we can update seen_count afterwards.
    """
    if not os.path.exists(HISTORY_FILE):
        return [], 0

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total_lines = len(lines)
    new_lines = lines[seen_count:]
    records = []

    for line in new_lines:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return records, total_lines


def f_float(d, key, default=0.0):
    v = d.get(key, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def f_int(d, key, default=0):
    v = d.get(key, default)
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def analyse_new(records):
    """
    Analyse only the new records since last run and print stats + AI coach feedback.
    """
    # Split into position cycles vs rebalances
    cycles = [r for r in records if "cycle_number" in r]
    rebalances = [r for r in records if "timestamp_rebalance" in r]

    print(f"\n📊 New records since last analysis: {len(records)}")
    print(f"   ➤ New position cycles: {len(cycles)}")
    print(f"   ➤ New rebalances:      {len(rebalances)}\n")

    # -----------------------------
    # 1) Position cycles (LP in/out)
    # -----------------------------
    total_cycles_pnl = 0.0
    wins = losses = 0
    best = None
    worst = None
    durations = []

    for c in cycles:
        pnl = f_float(c, "net_pnl_usd", 0.0)
        dur = f_float(c, "duration_minutes", 0.0)
        total_cycles_pnl += pnl
        durations.append(dur)

        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1

        if best is None or pnl > best:
            best = pnl
        if worst is None or pnl < worst:
            worst = pnl

    if cycles:
        avg_cycle_pnl = total_cycles_pnl / len(cycles)
        avg_dur = sum(durations) / len(durations) if durations else 0.0

        print("1️⃣ Position cycles (new only)")
        print(f"   Cycles:        {len(cycles)}")
        print(f"   Total net PnL: {total_cycles_pnl:+.2f} USDC")
        print(f"   Avg per cycle: {avg_cycle_pnl:+.2f} USDC")
        print(f"   Wins/Losses:   {wins}/{losses}")
        print(f"   Best/Worst:    {best:+.2f} / {worst:+.2f}")
        print(f"   Avg duration:  {avg_dur:.2f} min\n")

        # Short vs long cycles
        short = [c for c in cycles if f_float(c, "duration_minutes", 0.0) < 10.0]
        long_ = [c for c in cycles if f_float(c, "duration_minutes", 0.0) >= 30.0]

        if short:
            short_pnls = [f_float(c, "net_pnl_usd", 0.0) for c in short]
            short_avg = sum(short_pnls) / len(short_pnls)
            print(f"   • Short cycles (<10 min): {len(short)} | avg {short_avg:+.2f} USDC")
        if long_:
            long_pnls = [f_float(c, "net_pnl_usd", 0.0) for c in long_]
            long_avg = sum(long_pnls) / len(long_pnls)
            print(f"   • Long cycles (≥30 min):  {len(long_)} | avg {long_avg:+.2f} USDC")

        print()
    else:
        avg_cycle_pnl = 0.0
        short = long_ = []
        print("1️⃣ Position cycles (new only)")
        print("   No new LP cycles.\n")

    # -----------------------------
    # 2) Rebalances
    # -----------------------------
    by_type = defaultdict(list)
    for r in rebalances:
        t = r.get("type", "unknown")
        pn = f_float(r, "net_usd", 0.0)
        by_type[t].append(pn)

    total_reb_pnl = 0.0

    print("2️⃣ Rebalances (new only)")
    if not rebalances:
        print("   No new rebalances.\n")
    else:
        for t, vals in by_type.items():
            t_total = sum(vals)
            t_avg = t_total / len(vals)
            total_reb_pnl += t_total
            print(
                f"   {t:15s} → {len(vals):3d} events | "
                f"total {t_total:+7.2f} | avg {t_avg:+6.2f} USDC"
            )
        print(f"\n   Total rebalance net PnL (new): {total_reb_pnl:+.2f} USDC\n")

    # -----------------------------
    # 3) Combined (new only)
    # -----------------------------
    overall = total_cycles_pnl + total_reb_pnl
    print("3️⃣ Combined (new only)")
    print(f"   Positions total:  {total_cycles_pnl:+.2f} USDC")
    print(f"   Rebalances total: {total_reb_pnl:+.2f} USDC")
    print(f"   ➜ Overall total:  {overall:+.2f} USDC\n")

    # -----------------------------
    # 4) AI Coach feedback (new only)
    # -----------------------------
    print("🧠 Strategy Coach — Latest Phase Feedback")
    print("----------------------------------------")

    # Sample size notes
    if len(cycles) < 5:
        print("• Not many new cycles yet — treat these results as noisy.")
    else:
        print(f"• Based on {len(cycles)} new cycles and {len(rebalances)} new rebalances:")

    # 4a) Core LP assessment
    if cycles:
        if avg_cycle_pnl > 0.2:
            print(
                f"  → Core LP cycles look profitable in this phase "
                f"(avg {avg_cycle_pnl:+.2f} USDC per cycle)."
            )
        elif avg_cycle_pnl < -0.2:
            print(
                f"  → Core LP cycles are losing on average "
                f"(avg {avg_cycle_pnl:+.2f} USDC). "
                "Your range / timing may be too aggressive for the current market."
            )
        else:
            print(
                f"  → Core LP cycles are roughly breakeven "
                f"(avg {avg_cycle_pnl:+.2f} USDC). Fees are mostly offset by exits/gas."
            )

    # 4b) Short vs long behaviour
    if short:
        short_pnls = [f_float(c, "net_pnl_usd", 0.0) for c in short]
        short_avg = sum(short_pnls) / len(short_pnls)
        if short_avg < -0.1:
            print(
                f"• Short cycles (<10 min) are weak (avg {short_avg:+.2f}). "
                "This suggests you're churning too much on small moves."
            )
    if long_:
        long_pnls = [f_float(c, "net_pnl_usd", 0.0) for c in long_]
        long_avg = sum(long_pnls) / len(long_pnls)
        if long_avg > 0.1:
            print(
                f"• Long cycles (≥30 min) look healthier (avg {long_avg:+.2f}). "
                "Letting positions run longer seems to help."
            )

    # 4c) Panic vs normal rebalances
    panic_vals = by_type.get("panic_100_usdc", [])
    normal_vals = by_type.get("rebalance_50_50", [])

    if panic_vals:
        panic_total = sum(panic_vals)
        panic_avg = panic_total / len(panic_vals)
        print(
            f"• Panic events: {len(panic_vals)} | total {panic_total:+.2f} | "
            f"avg {panic_avg:+.2f} USDC."
        )
        if panic_total < 0:
            print(
                "  → Recent panics are a net drag. Options:\n"
                "    - Make panic harder to trigger (wider USD band, bigger tick buffer).\n"
                "    - Let EMA confirmation be stricter before re-entering."
            )

    if normal_vals:
        normal_total = sum(normal_vals)
        normal_avg = normal_total / len(normal_vals)
        print(
            f"• Normal 50/50 rebalances: {len(normal_vals)} | "
            f"total {normal_total:+.2f} | avg {normal_avg:+.2f} USDC."
        )
        if normal_total < 0:
            print(
                "  → Normal rebalances in this phase are costing you. Consider:\n"
                "    - A higher drift threshold before rebalancing.\n"
                "    - Slightly wider LP ranges so you rebalance less often."
            )

    # 4d) Overall verdict for this phase
    if overall < -0.1:
        print(
            f"\n⚠️ This latest phase is negative overall ({overall:+.2f} USDC). "
            "The main things to watch are:\n"
            "  - Are you exiting too early (short, losing cycles)?\n"
            "  - Are panics / rebalances eating more than LP earns?\n"
            "  - Can you widen the range or increase min cycle duration to reduce churn?"
        )
    elif overall > 0.1:
        print(
            f"\n✅ This latest phase is positive overall ({overall:+.2f} USDC). "
            "Leaning into the behaviours that generated these wins (cycle durations, "
            "range widths, calmer panic settings) is likely a good idea."
        )
    else:
        print(
            f"\n➖ This latest phase is roughly flat ({overall:+.2f} USDC). "
            "You're close to break-even — small tweaks to reduce rebalance costs "
            "or avoid unprofitable short cycles could push this positive."
        )

    print("\n(Each time you run this script, it only analyses new data since the last run.)\n")


def main():
    seen = load_seen_count()

    if not os.path.exists(HISTORY_FILE):
        print(f"❌ {HISTORY_FILE} not found.")
        return

    records, total_lines = load_new_records(seen)

    if not records:
        print("\n✅ No new data since last analysis.")
        return

    analyse_new(records)
    save_seen_count(total_lines)


if __name__ == "__main__":
    main()

