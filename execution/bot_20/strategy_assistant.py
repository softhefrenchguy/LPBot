import os
import json
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from statistics import mean

# ==============================
# CONFIG
# ==============================
STATE_FILE = "last_cycle_data.json"
HISTORY_FILE = "cycle_history.jsonl"

# How many recent cycles / rebalances to treat as "recent"
RECENT_CYCLES_N = 20
RECENT_HOURS_FOR_MARKET_REGIME = 6  # look-back window in hours for "recent" stats


# ==============================
# FILE HELPERS
# ==============================
def load_json_safe(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            return None
        return json.loads(raw)
    except Exception as e:
        print(f"⚠️ Could not read {path}: {e}")
        return None


def load_history():
    """Load all lines from cycle_history.jsonl into Python objects."""
    entries = []
    if not os.path.exists(HISTORY_FILE):
        print(f"⚠️ No {HISTORY_FILE} file found.")
        return entries
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


# ==============================
# BASIC PARSING / SPLITTING
# ==============================
def split_entries(entries):
    """Split history entries into LP cycles vs rebalances."""
    cycles = []
    rebalances = []
    for e in entries:
        if "cycle_number" in e:
            cycles.append(e)
        elif "type" in e:
            rebalances.append(e)
    return cycles, rebalances


def parse_timestamp(ts_str):
    """Parse your UTC timestamp format 'YYYY-MM-DD HH:MM:SS UTC'."""
    if not ts_str:
        return None
    try:
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return None


# ==============================
# STATS: CYCLES
# ==============================
def compute_cycle_stats(cycles):
    """Compute aggregate stats over LP cycles."""
    if not cycles:
        return {}

    net_list = []
    durations = []
    by_cycle = []

    for c in cycles:
        net = c.get("net_pnl_usd")
        dur = c.get("duration_minutes")
        if net is None:
            continue
        net = float(net)
        if dur is not None:
            try:
                dur = float(dur)
            except Exception:
                dur = None
        net_list.append(net)
        if dur is not None:
            durations.append(dur)
        by_cycle.append({
            "cycle_number": c.get("cycle_number"),
            "net": net,
            "dur": dur,
            "timestamp_withdraw": c.get("timestamp_withdraw")
        })

    wins = sum(1 for x in net_list if x > 0)
    losses = sum(1 for x in net_list if x <= 0)
    best = max(net_list) if net_list else 0.0
    worst = min(net_list) if net_list else 0.0

    # Duration buckets
    dur_buckets = {
        "<10": [],
        "10–30": [],
        "30–120": [],
        ">=120": [],
    }
    for c in by_cycle:
        d = c["dur"]
        if d is None:
            continue
        if d < 10:
            key = "<10"
        elif d < 30:
            key = "10–30"
        elif d < 120:
            key = "30–120"
        else:
            key = ">=120"
        dur_buckets[key].append(c["net"])

    dur_bucket_summary = {
        k: {
            "count": len(v),
            "avg_net": round(mean(v), 4) if v else 0.0
        }
        for k, v in dur_buckets.items()
    }

    stats = {
        "count": len(net_list),
        "total_net": round(sum(net_list), 4),
        "avg_net": round(mean(net_list), 4) if net_list else 0.0,
        "wins": wins,
        "losses": losses,
        "best": round(best, 4),
        "worst": round(worst, 4),
        "avg_duration": round(mean(durations), 2) if durations else None,
        "dur_buckets": dur_bucket_summary,
        "by_cycle": by_cycle,
    }
    return stats


def recent_cycles_stats(cycles, n=RECENT_CYCLES_N, hours_limit=RECENT_HOURS_FOR_MARKET_REGIME):
    """Stats on last N cycles & last X hours to see recent behaviour."""
    if not cycles:
        return {}

    # Sort by withdraw time if available, else by index
    def sort_key(c):
        ts = parse_timestamp(c.get("timestamp_withdraw"))
        return ts or datetime.min

    sorted_cycles = sorted(cycles, key=sort_key)
    recent = sorted_cycles[-n:]

    now = datetime.utcnow()
    time_filtered = [
        c for c in sorted_cycles
        if (parse_timestamp(c.get("timestamp_withdraw")) or now) >= now - timedelta(hours=hours_limit)
    ]

    def calc(c_list):
        if not c_list:
            return {"count": 0}
        nets = [float(c.get("net_pnl_usd", 0.0)) for c in c_list]
        return {
            "count": len(nets),
            "total_net": round(sum(nets), 4),
            "avg_net": round(mean(nets), 4),
            "wins": sum(1 for x in nets if x > 0),
            "losses": sum(1 for x in nets if x <= 0),
            "best": round(max(nets), 4),
            "worst": round(min(nets), 4),
        }

    return {
        "last_n": calc(recent),
        "last_hours": calc(time_filtered),
    }


# ==============================
# STATS: REBALANCES
# ==============================
def compute_rebalance_stats(rebalances):
    if not rebalances:
        return {}

    by_type = defaultdict(list)
    times = []

    for r in rebalances:
        ttype = r.get("type", "unknown")
        net = r.get("net_usd")
        if net is None:
            net = r.get("pnl_usd", 0.0)
        try:
            net = float(net)
        except Exception:
            net = 0.0

        by_type[ttype].append(net)

        ts = r.get("timestamp_rebalance")
        dt = parse_timestamp(ts)
        if dt:
            times.append((dt, net, ttype))

    type_summary = {}
    for ttype, vals in by_type.items():
        type_summary[ttype] = {
            "count": len(vals),
            "total_net": round(sum(vals), 4),
            "avg_net": round(mean(vals), 4),
        }

    total_reb = sum(sum(v) for v in by_type.values())

    # recent rebalances last X hours
    now = datetime.utcnow()
    recent_reb = [x for x in times if x[0] >= now - timedelta(hours=RECENT_HOURS_FOR_MARKET_REGIME)]
    recent_by_type = defaultdict(list)
    for dt, net, ttype in recent_reb:
        recent_by_type[ttype].append(net)
    recent_summary = {
        ttype: {
            "count": len(vals),
            "total_net": round(sum(vals), 4),
            "avg_net": round(mean(vals), 4),
        }
        for ttype, vals in recent_by_type.items()
    }

    return {
        "by_type": type_summary,
        "total_rebalance_net": round(total_reb, 4),
        "recent_hours": recent_summary,
    }


# ==============================
# OVERALL SNAPSHOT
# ==============================
def build_snapshot():
    entries = load_history()
    state = load_json_safe(STATE_FILE) or {}

    cycles, rebalances = split_entries(entries)
    cycle_stats = compute_cycle_stats(cycles)
    cycle_recent = recent_cycles_stats(cycles)
    reb_stats = compute_rebalance_stats(rebalances)

    current_width = state.get("range_width")
    current_lower = state.get("lower_bound_usd")
    current_upper = state.get("upper_bound_usd")
    panic_flag = bool(state.get("panic_active", False))

    snapshot = {
        "cycle_stats": cycle_stats,
        "cycle_recent": cycle_recent,
        "rebalance_stats": reb_stats,
        "current_width": current_width,
        "current_band": (current_lower, current_upper),
        "panic_active": panic_flag,
        "raw_cycles": cycles,
        "raw_rebalances": rebalances,
    }
    return snapshot


def format_snapshot_text(snap):
    cs = snap["cycle_stats"]
    cr = snap["cycle_recent"]
    rs = snap["rebalance_stats"]

    lines = []
    lines.append("=== LP Strategy Snapshot ===")
    # --- cycles overall
    if cs:
        lines.append(
            f"Cycles: {cs['count']} | total net {cs['total_net']:+.2f} USDC "
            f"| avg {cs['avg_net']:+.2f} USDC"
        )
        lines.append(
            f"Wins/Losses: {cs['wins']}/{cs['losses']} | best {cs['best']:+.2f} | worst {cs['worst']:+.2f}"
        )
        if cs["avg_duration"] is not None:
            lines.append(f"Avg duration: {cs['avg_duration']:.1f} min")
        lines.append("By duration bucket (avg net, count):")
        for bucket, info in cs["dur_buckets"].items():
            lines.append(
                f"  {bucket} min → avg {info['avg_net']:+.2f} USDC over {info['count']} cycles"
            )
    else:
        lines.append("No cycle data found.")

    lines.append("")

    # --- recent cycles
    if cr and cr.get("last_n", {}).get("count", 0) > 0:
        ln = cr["last_n"]
        lines.append(f"Last {ln['count']} cycles → avg {ln['avg_net']:+.2f} USDC "
                     f"(total {ln['total_net']:+.2f}, best {ln['best']:+.2f}, worst {ln['worst']:+.2f})")
    if cr and cr.get("last_hours", {}).get("count", 0) > 0:
        lh = cr["last_hours"]
        lines.append(
            f"Last {RECENT_HOURS_FOR_MARKET_REGIME}h cycles → avg {lh['avg_net']:+.2f} "
            f"(total {lh['total_net']:+.2f}, wins/losses {lh['wins']}/{lh['losses']})"
        )

    lines.append("")

    # --- rebalances
    if rs and rs["by_type"]:
        lines.append("Rebalances by type (net PnL):")
        for ttype, info in rs["by_type"].items():
            lines.append(
                f"  {ttype:<15} → {info['count']:3d} events | total {info['total_net']:+.2f} "
                f"| avg {info['avg_net']:+.2f} USDC"
            )
        lines.append(f"Total rebalance net: {rs['total_rebalance_net']:+.2f} USDC")

        if rs["recent_hours"]:
            lines.append(f"Last {RECENT_HOURS_FOR_MARKET_REGIME}h rebalances:")
            for ttype, info in rs["recent_hours"].items():
                lines.append(
                    f"  {ttype:<15} → {info['count']:3d} events | total {info['total_net']:+.2f} "
                    f"| avg {info['avg_net']:+.2f} USDC"
                )
    else:
        lines.append("No rebalance data found.")

    lines.append("")

    # --- current config
    cw = snap["current_width"]
    cl, cu = snap["current_band"]
    lines.append(f"Current saved width (ticks): {cw}")
    lines.append(f"Current saved USD band: {cl} → {cu}")
    lines.append(f"Panic active: {snap['panic_active']}")

    return "\n".join(lines)


# ==============================
# HEURISTIC “AI” (no external API)
# ==============================
def heuristic_advice(question, snap):
    """Fallback advisor if you don't plug in a real LLM yet."""
    cs = snap["cycle_stats"]
    cr = snap["cycle_recent"]
    rs = snap["rebalance_stats"]

    answers = []

    # 1) Basic direction
    if cs and cs["avg_net"] < 0:
        answers.append(
            f"- Over full history, your cycles lose on average {cs['avg_net']:+.2f} USDC. "
            "Core range logic is not profitable yet."
        )
    elif cs and cs["avg_net"] > 0:
        answers.append(
            f"- Over full history, your cycles are profitable on average {cs['avg_net']:+.2f} USDC."
        )

    # 2) Recent vs full
    if cr and cr.get("last_n", {}).get("count", 0) > 0:
        ln = cr["last_n"]
        if ln["avg_net"] < cs["avg_net"]:
            answers.append(
                f"- Your last {ln['count']} cycles are *worse* than long-run: "
                f"{ln['avg_net']:+.2f} vs {cs['avg_net']:+.2f} USDC per cycle."
            )
        else:
            answers.append(
                f"- Your last {ln['count']} cycles are *better* than long-run: "
                f"{ln['avg_net']:+.2f} vs {cs['avg_net']:+.2f} USDC per cycle."
            )

    # 3) Duration buckets
    if cs and cs["dur_buckets"]:
        best_bucket = max(cs["dur_buckets"].items(), key=lambda kv: kv[1]["avg_net"])
        worst_bucket = min(cs["dur_buckets"].items(), key=lambda kv: kv[1]["avg_net"])
        answers.append(
            f"- Best duration bucket: {best_bucket[0]} min "
            f"(avg {best_bucket[1]['avg_net']:+.2f} USDC)."
        )
        answers.append(
            f"- Worst duration bucket: {worst_bucket[0]} min "
            f"(avg {worst_bucket[1]['avg_net']:+.2f} USDC). "
            "You could bias your range logic to spend more time in the better bucket."
        )

    # 4) Rebalance pain
    if rs and rs["by_type"]:
        for ttype, info in rs["by_type"].items():
            if info["total_net"] < 0:
                answers.append(
                    f"- {ttype} is costing you {info['total_net']:+.2f} USDC overall "
                    f"(avg {info['avg_net']:+.2f} each). Consider triggering it less often."
                )

    # 5) Quick logic depending on question keywords
    q = question.lower()
    if "range" in q or "width" in q:
        answers.append(
            "- To tune range: compare average net PnL in short vs long duration buckets. "
            "If very short cycles (<10 min) are negative on average, your range is too tight "
            "or exits are too frequent. If very long cycles (>=120 min) are negative, "
            "ranges might be too wide and you're not capturing enough fee density."
        )
    if "panic" in q:
        answers.append(
            "- Panic events should be rare and high-impact. If their avg net is strongly "
            "negative, tighten your panic trigger (deeper tick + USD move) and lengthen "
            "cooldowns so you don't re-enter straight into chop."
        )
    if "rebalance" in q or "50/50" in q:
        answers.append(
            "- Treat 50/50 rebalances as a pure *cost*: only do them when they enable a full "
            "LP reset you actually want (e.g. after exiting or changing width), rather than "
            "micro-correcting small drifts."
        )

    if not answers:
        answers.append("I don't see much in the data for that specific question yet. Try asking about ranges, panic, or rebalances.")

    return "\n".join(answers)


# ==============================
# PLUG-IN POINT: REAL LLM CALL
# ==============================
def call_llm(context_text, question):
    """
    If you want a real AI brain, plug your provider here.

    For example (pseudocode):

        from openai import OpenAI
        client = OpenAI()

        prompt = f\"\"\"You are a trading strategy coach...
        {context_text}

        User question: {question}
        \"\"\"

        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
        )
        return resp.output_text

    For now we just return None so the script falls back to heuristic mode.
    """
    return None


# ==============================
# CLI MAIN
# ==============================
def main():
    snap = build_snapshot()
    context = format_snapshot_text(snap)
    print(context)
    print("\nType your question about performance / ranges / panic / rebalances.")
    print("Type 'quit' or 'exit' to leave.\n")

    while True:
        try:
            q = input("❓ You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye.")
            break

        if not q:
            continue
        if q.lower() in {"q", "quit", "exit"}:
            print("👋 Bye.")
            break

        # 1) Try real LLM if you wire it up
        ai_answer = call_llm(context, q)
        if ai_answer:
            print("\n🤖 Strategy AI:\n" + ai_answer + "\n")
        else:
            # 2) Fallback: heuristic advisor
            print("\n🤖 Strategy AI (heuristic mode):")
            print(heuristic_advice(q, snap))
            print("")

if __name__ == "__main__":
    main()
