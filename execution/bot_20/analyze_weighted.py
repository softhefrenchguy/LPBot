import json
import math
import os
from collections import defaultdict

HISTORY_FILE = "cycle_history.jsonl"

# Half-life ~10 cycles -> lambda:
HALF_LIFE_CYCLES = 10.0
LAMBDA = math.log(2) / HALF_LIFE_CYCLES


def load_records():
    if not os.path.exists(HISTORY_FILE):
        print(f"❌ {HISTORY_FILE} not found.")
        return []

    records = []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def f_float(d, key, default=0.0):
    v = d.get(key, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def analyze_weighted(records):
    """
    Uses exponential weights so recent cycles/rebalances matter more.
    """
    # Split
    cycles = [r for r in records if "cycle_number" in r]
    rebalances = [r for r in records if "timestamp_rebalance" in r]

    N_cycles = len(cycles)
    N_reb = len(rebalances)

    print(f"\n📊 Records loaded: {len(records)}")
    print(f"   ➤ LP cycles:    {N_cycles}")
    print(f"   ➤ Rebalances:   {N_reb}\n")

    # -----------------------
    # 1) Cycles: raw & weighted
    # -----------------------
    total_raw_cycles = 0.0
    wins = losses = 0
    best = None
    worst = None
    durations = []

    total_w = 0.0
    total_w_pnl = 0.0
    total_w_dur = 0.0

    for idx, c in enumerate(cycles):
        pnl = f_float(c, "net_pnl_usd", 0.0)
        dur = f_float(c, "duration_minutes", 0.0)

        total_raw_cycles += pnl
        durations.append(dur)

        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1

        if best is None or pnl > best:
            best = pnl
        if worst is None or pnl < worst:
            worst = pnl

        # recency weight
        # newest cycle has age=0 → highest weight
        age = (N_cycles - 1) - idx
        w = math.exp(-LAMBDA * age)

        total_w += w
        total_w_pnl += w * pnl
        total_w_dur += w * dur

    if N_cycles > 0:
        avg_raw_pnl = total_raw_cycles / N_cycles
        avg_raw_dur = sum(durations) / N_cycles if durations else 0.0
        avg_w_pnl = total_w_pnl / total_w if total_w > 0 else 0.0
        avg_w_dur = total_w_dur / total_w if total_w > 0 else 0.0
    else:
        avg_raw_pnl = avg_raw_dur = avg_w_pnl = avg_w_dur = 0.0

    print("1️⃣ Position cycles (all history)")
    if N_cycles == 0:
        print("   No cycles found.\n")
    else:
        print(f"   Cycles:              {N_cycles}")
        print(f"   Raw total net PnL:   {total_raw_cycles:+.2f} USDC")
        print(f"   Raw avg per cycle:   {avg_raw_pnl:+.2f} USDC")
        print(f"   Raw wins/losses:     {wins}/{losses}")
        print(f"   Raw best/worst:      {best:+.2f} / {worst:+.2f}")
        print(f"   Raw avg duration:    {avg_raw_dur:.2f} min")
        print()
        print(f"   ⚖️ Recency-weighted avg PnL: {avg_w_pnl:+.2f} USDC")
        print(f"   ⚖️ Recency-weighted avg dur: {avg_w_dur:.2f} min\n")

    # Short vs long (weighted)
    short_w_pnl = short_w = 0.0
    long_w_pnl = long_w = 0.0
    for idx, c in enumerate(cycles):
        dur = f_float(c, "duration_minutes", 0.0)
        pnl = f_float(c, "net_pnl_usd", 0.0)
        age = (N_cycles - 1) - idx
        w = math.exp(-LAMBDA * age)
        if dur < 10.0:
            short_w += w
            short_w_pnl += w * pnl
        if dur >= 30.0:
            long_w += w
            long_w_pnl += w * pnl

    if short_w > 0:
        print(f"   • Recency-weighted short cycles (<10 min): avg {short_w_pnl/short_w:+.2f} USDC")
    if long_w > 0:
        print(f"   • Recency-weighted long cycles (≥30 min): avg {long_w_pnl/long_w:+.2f} USDC")
    print()

    # -----------------------
    # 2) Rebalances: raw & weighted
    # -----------------------
    by_type_raw = defaultdict(list)
    by_type_weighted = defaultdict(lambda: {"w": 0.0, "wp": 0.0})

    for idx, r in enumerate(rebalances):
        t = r.get("type", "unknown")
        pnl = f_float(r, "net_usd", 0.0)
        by_type_raw[t].append(pnl)

        age = (N_reb - 1) - idx
        w = math.exp(-LAMBDA * age)
        by_type_weighted[t]["w"] += w
        by_type_weighted[t]["wp"] += w * pnl

    print("2️⃣ Rebalances (all history)")
    total_raw_reb = 0.0
    total_w_reb = 0.0

    if N_reb == 0:
        print("   No rebalances found.\n")
    else:
        for t, vals in by_type_raw.items():
            raw_total = sum(vals)
            raw_avg = raw_total / len(vals)
            total_raw_reb += raw_total

            w_info = by_type_weighted[t]
            if w_info["w"] > 0:
                w_avg = w_info["wp"] / w_info["w"]
            else:
                w_avg = 0.0
            total_w_reb += w_info["wp"]

            print(
                f"   {t:15s} → {len(vals):3d} events | "
                f"raw total {raw_total:+7.2f} | raw avg {raw_avg:+6.2f} | "
                f"weighted avg {w_avg:+6.2f}"
            )

        print(f"\n   Raw rebalance total PnL:      {total_raw_reb:+.2f} USDC")
        if total_w > 0:
            print(f"   ⚖️ Weighted rebalance total: {total_w_reb:+.2f} USDC")
        print()

    # -----------------------
    # 3) Combined
    # -----------------------
    raw_overall = total_raw_cycles + total_raw_reb
    weighted_overall = avg_w_pnl * (total_w / total_w if total_w > 0 else 0) + total_w_reb

    print("3️⃣ Combined PnL (all history)")
    print(f"   Raw positions total:   {total_raw_cycles:+.2f} USDC")
    print(f"   Raw rebalances total:  {total_raw_reb:+.2f} USDC")
    print(f"   ➜ Raw overall total:   {raw_overall:+.2f} USDC\n")

    print(f"   ⚖️ Recency-weighted view:")
    print(f"   • Weighted positions avg PnL/cycle: {avg_w_pnl:+.2f} USDC")
    print(f"   • Weighted rebalance contribution:  {total_w_reb:+.2f} USDC (in PnL units)\n")

    # -----------------------
    # 4) AI Coach (based on weighted stats)
    # -----------------------
    print("🧠 Strategy Coach — Recency-weighted view")
    print("----------------------------------------")

    # Core LP signal
    if N_cycles == 0:
        print("• No LP cycles in history yet — nothing to analyse.")
        return

    if avg_w_pnl > 0.2:
        print(
            f"• Recently, your core LP cycles look profitable "
            f"(weighted avg {avg_w_pnl:+.2f} USDC per cycle). "
            "Your current range / timing is working better than the long-term average."
        )
    elif avg_w_pnl < -0.2:
        print(
            f"• Recently, your core LP cycles are losing on average "
            f"(weighted avg {avg_w_pnl:+.2f} USDC). "
            "The market regime has probably shifted, and your current range logic "
            "is too aggressive or too tight for this volatility."
        )
    else:
        print(
            f"• Recently, your core LP cycles are close to breakeven "
            f"(weighted avg {avg_w_pnl:+.2f} USDC). "
            "Fees are roughly cancelled by exits and gas."
        )

    # Short vs long (weighted)
    if short_w > 0:
        short_avg_w = short_w_pnl / short_w
        if short_avg_w < -0.1:
            print(
                f"• Short cycles (<10 min) are weak on a recency-weighted basis "
                f"(avg {short_avg_w:+.2f}). You're still churning on noise: "
                "consider increasing min cycle duration or widening the range so you don't exit so fast."
            )
    if long_w > 0:
        long_avg_w = long_w_pnl / long_w
        if long_avg_w > 0.1:
            print(
                f"• Long cycles (≥30 min) look healthier recently "
                f"(avg {long_avg_w:+.2f}). Letting positions run longer seems to help."
            )

    # Panic vs normal (weighted)
    panic_info = by_type_weighted.get("panic_100_usdc")
    normal_info = by_type_weighted.get("rebalance_50_50")

    if panic_info and panic_info["w"] > 0:
        panic_avg_w = panic_info["wp"] / panic_info["w"]
        print(
            f"• Panic events (weighted) average {panic_avg_w:+.2f} USDC. "
            "If this is strongly negative, your panic triggers are too sensitive "
            "or you're re-entering too soon after volatility spikes."
        )

    if normal_info and normal_info["w"] > 0:
        normal_avg_w = normal_info["wp"] / normal_info["w"]
        print(
            f"• Normal 50/50 rebalances (weighted) average {normal_avg_w:+.2f} USDC. "
            "If this is negative, you should raise the drift threshold or reduce how often you rebalance."
        )

    # Overall recent tone
    if raw_overall < 0 and avg_w_pnl > 0:
        print(
            "\n⚖️ Long-term you're down, but the recency-weighted LP PnL is improving.\n"
            "   → Interpretation: your current configuration is better than the old one.\n"
            "     Focus on reinforcing the recent behaviour (wider ranges, fewer panics, "
            "less rebalance churn)."
        )
    elif raw_overall > 0 and avg_w_pnl < 0:
        print(
            "\n⚖️ Long-term you're up, but recent LP PnL is deteriorating.\n"
            "   → Interpretation: market conditions changed, and the old parameters "
            "are no longer ideal. It's time to tune again."
        )
    else:
        print(
            "\n⚖️ Compare raw vs weighted metrics to see whether things are improving "
            "or worsening with the latest configuration."
        )

    print("\n(Recency-weighted stats emphasise your last ~10–20 cycles while still "
          "respecting older history.)\n")


def main():
    records = load_records()
    if not records:
        print("No data to analyse.")
        return
    analyze_weighted(records)


if __name__ == "__main__":
    main()
