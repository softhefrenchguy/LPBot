#!/usr/bin/env python3
# analyze_with_ai.py — stats, fresh reset, auto-archive, last-5 analysis

import os
import json
import argparse
import shutil
from datetime import datetime, timedelta

# ----------------------------
# Paths / constants
# ----------------------------
BASE_DIR = os.path.dirname(__file__)
CYCLE_HISTORY_FILE = os.path.join(BASE_DIR, "cycle_history.jsonl")
REBAL_HISTORY_FILE = os.path.join(BASE_DIR, "rebalance_history.jsonl")
ARCHIVE_DIR = os.path.join(BASE_DIR, "archive")

# How many to keep in "main" files before auto-archive
MAX_CYCLES_KEEP = 300
MAX_REBAL_KEEP = 1000

# How many cycles count as "recent"
RECENT_N = 5

# For 24h PnL
HOURS_24 = 24


# ----------------------------
# Helpers
# ----------------------------
def ensure_archive_dir():
    if not os.path.exists(ARCHIVE_DIR):
        os.makedirs(ARCHIVE_DIR, exist_ok=True)


def load_jsonl(path):
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # skip bad lines
                continue
    return records


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def archive_file(path, label=""):
    """
    Move a full file into archive/ with timestamp. Used by --fresh.
    """
    if not os.path.exists(path):
        return None
    ensure_archive_dir()
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    base = os.path.basename(path)
    label_part = f"_{label}" if label else ""
    new_name = f"{base}.{ts}{label_part}.archive"
    dest = os.path.join(ARCHIVE_DIR, new_name)
    shutil.move(path, dest)
    return dest


def auto_archive_records(path, records, max_keep, label="auto"):
    """
    If len(records) > max_keep, keep only last max_keep in main file,
    archive the older ones as a new jsonl in /archive.
    """
    if len(records) <= max_keep:
        return records  # nothing to do

    ensure_archive_dir()
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    base = os.path.basename(path)
    archive_path = os.path.join(
        ARCHIVE_DIR, f"{base}.{ts}_{label}.jsonl"
    )

    # Split
    old = records[:-max_keep]
    keep = records[-max_keep:]

    # Write archive file
    write_jsonl(archive_path, old)
    # Rewrite main file with recent
    write_jsonl(path, keep)

    print(
        f"🗄️ Auto-archived {len(old)} old records from {base} → {archive_path} "
        f"(kept last {max_keep})"
    )
    return keep


def parse_utc(ts_str):
    # Expected formats used in your history:
    # "2025-11-28 15:01:57 UTC"
    if not ts_str:
        return None
    ts_str = ts_str.replace(" UTC", "")
    try:
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


# ----------------------------
# Stats helpers
# ----------------------------
def summarise_cycles(cycles):
    if not cycles:
        return None

    count = len(cycles)
    total_pnl = 0.0
    wins = 0
    losses = 0
    best = None
    worst = None
    total_duration = 0.0

    for c in cycles:
        pnl = float(c.get("net_pnl_usd", c.get("pnl_position_usd", 0.0)))
        total_pnl += pnl
        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1
        best = pnl if (best is None or pnl > best) else best
        worst = pnl if (worst is None or pnl < worst) else worst

        dur = float(c.get("duration_minutes", 0.0))
        total_duration += dur

    avg_pnl = total_pnl / count if count > 0 else 0.0
    avg_duration = total_duration / count if count > 0 else 0.0

    return {
        "count": count,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "wins": wins,
        "losses": losses,
        "best": best or 0.0,
        "worst": worst or 0.0,
        "avg_duration": avg_duration,
    }


def summarise_rebalances(rebals):
    # group by type
    by_type = {}
    for r in rebals:
        rtype = r.get("type", "unknown")
        pnl = float(r.get("net_usd", r.get("pnl_usd", 0.0)))
        stats = by_type.setdefault(
            rtype,
            {"count": 0, "total_pnl": 0.0}
        )
        stats["count"] += 1
        stats["total_pnl"] += pnl

    # compute averages
    for k, v in by_type.items():
        if v["count"] > 0:
            v["avg_pnl"] = v["total_pnl"] / v["count"]
        else:
            v["avg_pnl"] = 0.0
    return by_type


def filter_last_24h(records, time_key):
    if not records:
        return []
    cutoff = datetime.utcnow() - timedelta(hours=HOURS_24)
    out = []
    for r in records:
        ts_str = r.get(time_key)
        dt = parse_utc(ts_str)
        if dt and dt >= cutoff:
            out.append(r)
    return out


def compute_pnl_last_24h(cycles, rebalances):
    cycles_24 = filter_last_24h(cycles, "timestamp_withdraw")
    rebals_24 = filter_last_24h(rebalances, "timestamp_rebalance")

    cycles_pnl = sum(
        float(c.get("net_pnl_usd", c.get("pnl_position_usd", 0.0)))
        for c in cycles_24
    )
    rebals_pnl = sum(
        float(r.get("net_usd", r.get("pnl_usd", 0.0)))
        for r in rebals_24
    )
    return {
        "cycles_24": cycles_24,
        "rebals_24": rebals_24,
        "cycles_pnl_24": cycles_pnl,
        "rebals_pnl_24": rebals_pnl,
        "combined_24": cycles_pnl + rebals_pnl,
    }


# ----------------------------
# Main analysis
# ----------------------------
def do_analysis():
    # Load
    cycles = load_jsonl(CYCLE_HISTORY_FILE)
    rebals = load_jsonl(REBAL_HISTORY_FILE)

    # Auto-archive if big
    cycles = auto_archive_records(
        CYCLE_HISTORY_FILE, cycles, MAX_CYCLES_KEEP, label="cycles"
    )
    rebals = auto_archive_records(
        REBAL_HISTORY_FILE, rebals, MAX_REBAL_KEEP, label="rebalances"
    )

    print("📊 Local stats summary:\n")

    # --- Full-history cycles ---
    c_stats = summarise_cycles(cycles)
    if c_stats:
        print("=== Full-history LP cycles ===")
        print(
            f"count={c_stats['count']}, "
            f"total_pnl={c_stats['total_pnl']:.2f}, "
            f"avg_pnl={c_stats['avg_pnl']:.2f}"
        )
        print(
            f"wins={c_stats['wins']}, losses={c_stats['losses']}, "
            f"best={c_stats['best']:.4f}, worst={c_stats['worst']:.4f}"
        )
        print(f"avg_duration_min={c_stats['avg_duration']:.2f}\n")
    else:
        print("No LP cycles found yet.\n")

    # --- Recent last N cycles (N = 5) ---
    if cycles:
        recent = cycles[-RECENT_N:]
        r_stats = summarise_cycles(recent)
        print(f"=== Recent LP cycles (last ~{RECENT_N}) ===")
        print(
            f"count={r_stats['count']}, "
            f"total_pnl={r_stats['total_pnl']:.2f}, "
            f"avg_pnl={r_stats['avg_pnl']:.2f}"
        )
        print(
            f"wins={r_stats['wins']}, losses={r_stats['losses']}, "
            f"avg_duration_min={r_stats['avg_duration']:.2f}\n"
        )

    # --- Rebalances ---
    if rebals:
        print("=== Rebalances (all history) ===")
        reb_s = summarise_rebalances(rebals)
        total_rebal_pnl = 0.0
        for rtype, st in reb_s.items():
            print(
                f"type={rtype}, count={st['count']}, "
                f"total={st['total_pnl']:.2f}, avg={st['avg_pnl']:.2f}"
            )
            total_rebal_pnl += st["total_pnl"]
        print(f"\nRebalances total PnL (all): {total_rebal_pnl:.2f}\n")
    else:
        print("No rebalance records yet.\n")

    # --- Last 24h PnL ---
    pnl_24 = compute_pnl_last_24h(cycles, rebals)
    print("=== PnL last 24h ===")
    print(f"Cycles PnL:      {pnl_24['cycles_pnl_24']:.2f} USDC")
    print(f"Rebalances PnL:  {pnl_24['rebals_pnl_24']:.2f} USDC")
    print(f"Combined PnL:    {pnl_24['combined_24']:.2f} USDC\n")


def fresh_reset():
    """
    Fresh start: archive existing history files and leave empty ones.
    """
    print("🧹 Fresh start requested — archiving current history...\n")
    c_arch = archive_file(CYCLE_HISTORY_FILE, label="fresh")
    r_arch = archive_file(REBAL_HISTORY_FILE, label="fresh")

    if c_arch:
        print(f"📦 Archived cycle history → {c_arch}")
    else:
        print("ℹ️ No existing cycle_history.jsonl to archive.")

    if r_arch:
        print(f"📦 Archived rebalance history → {r_arch}")
    else:
        print("ℹ️ No existing rebalance_history.jsonl to archive.")

    # Create empty files (optional – your main bot will append anyway)
    open(CYCLE_HISTORY_FILE, "w", encoding="utf-8").close()
    open(REBAL_HISTORY_FILE, "w", encoding="utf-8").close()

    print("\n✅ Fresh start complete. New cycles/rebalances will be logged from now on.")


# ----------------------------
# CLI entry
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Analyze LP bot history (with fresh reset & auto-archive)."
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Archive existing history and start fresh.",
    )
    args = parser.parse_args()

    if args.fresh:
        fresh_reset()
    else:
        do_analysis()


if __name__ == "__main__":
    main()

