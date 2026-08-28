# ai_core/ai_engine.py
"""
Offline AI engine for analysing LP cycle history and tuning strategy.

Reads:
    cycle_history.jsonl  (one JSON per line, as your bot already writes)

Outputs:
    - ai_tuning.json          → thresholds used by main_auto_v3.py
    - cycle_analysis.csv      → flat per-cycle dataset (for Excel / further analysis)
    - cycle_analysis.html     → human-friendly summary report

The analyser:
    - Computes full PnL, fees, gas, vs-HODL metrics
    - Breaks down performance by exit_reason (band_top, trend_up_exit, etc.)
    - Derives (when enough data) strong-uptrend thresholds
"""

import os
import json
import csv
from typing import List, Dict, Any, Tuple
from statistics import mean, median
from collections import defaultdict

BASE_DIR = os.path.dirname(__file__)

# Likely locations of your history file
CANDIDATE_HISTORY_FILES = [
    os.path.join(BASE_DIR, "..", "cycle_history.jsonl"),
    os.path.join(BASE_DIR, "cycle_history.jsonl"),
]

OUT_TUNING_FILE = os.path.join(BASE_DIR, "ai_tuning.json")
OUT_CSV_FILE = os.path.join(BASE_DIR, "cycle_analysis.csv")
OUT_HTML_FILE = os.path.join(BASE_DIR, "cycle_analysis.html")


# ---------------------------
# Helpers
# ---------------------------

def find_history_file() -> str:
    for path in CANDIDATE_HISTORY_FILES:
        if os.path.exists(path):
            return os.path.abspath(path)
    raise FileNotFoundError(
        "Could not find cycle_history.jsonl. Tried:\n"
        + "\n".join(CANDIDATE_HISTORY_FILES)
    )


def load_cycles(path: str) -> List[Dict[str, Any]]:
    cycles: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cycles.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return cycles


def safe_ratio(num: float, den: float) -> float:
    if not den:
        return 0.0
    return num / den


def extract_trend_samples(cycles: List[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
    """
    From 'trend_up_exit' cycles, derive:
      - price move (%): (exit_eth_price - entry_eth_price) / entry_eth_price
      - slope proxy (% per minute): price_move_pct / duration_minutes
    """
    moves = []
    slopes = []
    for c in cycles:
        if c.get("exit_reason") != "trend_up_exit":
            continue
        entry_p = c.get("entry_eth_price")
        exit_p = c.get("exit_eth_price")
        dur_min = c.get("duration_minutes", 0.0)
        if not entry_p or not exit_p:
            continue

        move_pct = safe_ratio(exit_p - entry_p, entry_p)
        if move_pct <= 0:
            continue

        moves.append(move_pct)
        slope = safe_ratio(move_pct, dur_min)
        if slope > 0:
            slopes.append(slope)

    return moves, slopes


def robust_percentile(values: List[float], q: float, default: float) -> float:
    """
    q is in [0,1]. If not enough samples, return default.
    """
    if not values:
        return default
    sorted_vals = sorted(values)
    idx = int(q * (len(sorted_vals) - 1))
    idx = max(0, min(idx, len(sorted_vals) - 1))
    return sorted_vals[idx]


# ---------------------------
# AI tuning from history
# ---------------------------

def compute_tuning(cycles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute AI tuning parameters from real cycle history.
    """
    print(f"Loaded {len(cycles)} cycles from history.")

    band_top_cycles = [c for c in cycles if c.get("exit_reason") == "band_top"]
    trend_up_cycles = [c for c in cycles if c.get("exit_reason") == "trend_up_exit"]

    print(f"  band_top cycles      : {len(band_top_cycles)}")
    print(f"  trend_up_exit cycles : {len(trend_up_cycles)}")

    # --- defaults (your current values) ---
    default_gap_enter = 0.004   # 0.4%
    default_gap_exit = 0.002    # 0.2%
    default_slope_enter = 0.002 # 0.2%/min
    default_slope_exit = 0.0005 # 0.05%/min

    # === strong-uptrend thresholds from trend_up cycles ===
    moves, slopes = extract_trend_samples(trend_up_cycles)

    if len(moves) >= 5:
        gap_enter = robust_percentile(moves, 0.4, default_gap_enter)
        gap_exit = gap_enter * 0.5
        print(f"  [AI] Derived strong_up_gap_enter from history: {gap_enter*100:.3f}%")
        print(f"  [AI] Derived strong_up_gap_exit  from history: {gap_exit*100:.3f}%")
    else:
        gap_enter = default_gap_enter
        gap_exit = default_gap_exit
        print("  [AI] Not enough trend_up cycles; using default gap thresholds.")

    if len(slopes) >= 5:
        slope_enter = robust_percentile(slopes, 0.4, default_slope_enter)
        slope_exit = slope_enter * 0.25
        print(f"  [AI] Derived strong_up_slope_enter from history: {slope_enter*100:.3f}%/min")
        print(f"  [AI] Derived strong_up_slope_exit  from history: {slope_exit*100:.3f}%/min")
    else:
        slope_enter = default_slope_enter
        slope_exit = default_slope_exit
        print("  [AI] Not enough slope samples; using default slope thresholds.")

    # === band_top stats (separate, as requested) ===
    band_top_stats = {}
    if band_top_cycles:
        band_pnls = [c.get("pnl_position_usd", 0.0) for c in band_top_cycles]
        band_durs = [c.get("duration_minutes", 0.0) for c in band_top_cycles]
        band_pnls = [p for p in band_pnls if p is not None]
        band_durs = [d for d in band_durs if d is not None and d > 0]

        if band_pnls:
            band_top_stats["median_pnl_usd"] = median(band_pnls)
        if band_durs:
            band_top_stats["median_duration_min"] = median(band_durs)

        print("  [AI] band_top stats:")
        for k, v in band_top_stats.items():
            print(f"      {k}: {v:.4f}")
    else:
        print("  [AI] No band_top cycles found; skipping band_top stats.")

    # === smart recenter defaults (can be learned later) ===
    smart_recenter_cfg = {
        "distance_below_band_min": 0.01,
        "ema_stable_ticks": 10,
        "cooldown_seconds": 30.0
    }

    tuning = {
        "strong_up_gap_enter": float(gap_enter),
        "strong_up_gap_exit": float(gap_exit),
        "use_slope_filter": True,
        "strong_up_slope_enter": float(slope_enter),
        "strong_up_slope_exit": float(slope_exit),
        "band_top_stats": band_top_stats,
        "smart_recenter": smart_recenter_cfg,
    }

    return tuning


# ---------------------------
# Full analytics (PnL, gas, fees, etc.)
# ---------------------------

def compute_overall_stats(cycles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute global metrics over all cycles.
    """
    n = len(cycles)
    if n == 0:
        return {}

    def vals(key: str) -> List[float]:
        out = []
        for c in cycles:
            v = c.get(key)
            if isinstance(v, (int, float)):
                out.append(float(v))
        return out

    pnl_after_fees = vals("real_cycle_pnl_after_fees_usd")
    net_lp_pnl = vals("net_lp_pnl_usd")
    gas_costs = vals("gas_cost_usd")
    fees_total = vals("fees_total_usd")
    vs_hodl = vals("vs_hodl_eth_usd")
    durations = vals("duration_minutes")

    def safe_mean(x: List[float]) -> float:
        return mean(x) if x else 0.0

    wins = sum(1 for x in pnl_after_fees if x > 0)
    losses = sum(1 for x in pnl_after_fees if x < 0)
    winrate = wins / n if n > 0 else 0.0

    stats = {
        "num_cycles": n,
        "avg_pnl_after_fees": safe_mean(pnl_after_fees),
        "median_pnl_after_fees": median(pnl_after_fees) if pnl_after_fees else 0.0,
        "avg_net_lp_pnl": safe_mean(net_lp_pnl),
        "avg_gas_cost": safe_mean(gas_costs),
        "avg_fees_total": safe_mean(fees_total),
        "avg_vs_hodl": safe_mean(vs_hodl),
        "avg_duration_min": safe_mean(durations),
        "median_duration_min": median(durations) if durations else 0.0,
        "wins": wins,
        "losses": losses,
        "winrate": winrate,
    }
    return stats


def compute_stats_by_exit_reason(cycles: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Compute same metrics grouped by exit_reason.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in cycles:
        reason = c.get("exit_reason", "unknown")
        grouped[reason].append(c)

    per_reason: Dict[str, Dict[str, Any]] = {}
    for reason, group in grouped.items():
        per_reason[reason] = compute_overall_stats(group)
    return per_reason


def export_csv(cycles: List[Dict[str, Any]], path: str) -> None:
    """
    Write a flat CSV with useful columns, one row per cycle.
    """
    if not cycles:
        return

    # Select a subset of fields that are most relevant
    fieldnames = [
        "cycle_number",
        "timestamp_create",
        "timestamp_withdraw",
        "duration_minutes",
        "exit_reason",
        "value_in_usd",
        "value_out_usd",
        "pnl_position_usd",
        "pnl_vs_hodl_usd",
        "gas_cost_usd",
        "net_lp_pnl_usd",
        "real_cycle_pnl_before_fees_usd",
        "real_cycle_pnl_after_fees_usd",
        "fees_total_usd",
        "fees_sent_usd",
        "entry_eth_price",
        "exit_eth_price",
        "vs_hodl_eth_usd",
        "entry_usdc",
        "entry_weth",
        "principal_usdc",
        "principal_weth",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in cycles:
            row = {k: c.get(k, "") for k in fieldnames}
            writer.writerow(row)


def export_html_report(overall: Dict[str, Any],
                       by_reason: Dict[str, Dict[str, Any]],
                       out_path: str) -> None:
    """
    Write a simple HTML report with aggregated stats.
    """
    def fmt(x: float) -> str:
        return f"{x:.4f}"

    html = []
    html.append("<html><head><title>LP Cycle Analysis</title></head><body>")
    html.append("<h1>LP Cycle Analysis</h1>")

    # Overall stats
    html.append("<h2>Overall</h2>")
    if overall:
        html.append("<table border='1' cellpadding='4' cellspacing='0'>")
        for k, v in overall.items():
            html.append(f"<tr><th>{k}</th><td>{fmt(v) if isinstance(v, float) else v}</td></tr>")
        html.append("</table>")
    else:
        html.append("<p>No cycles.</p>")

    # By exit_reason
    html.append("<h2>By exit_reason</h2>")
    if by_reason:
        html.append("<table border='1' cellpadding='4' cellspacing='0'>")
        # header
        html.append("<tr><th>exit_reason</th><th>#cycles</th><th>avg_pnl_after_fees</th>"
                    "<th>avg_net_lp_pnl</th><th>avg_gas_cost</th>"
                    "<th>avg_fees_total</th><th>avg_vs_hodl</th>"
                    "<th>avg_duration_min</th><th>winrate</th></tr>")
        for reason, stats in by_reason.items():
            html.append(
                "<tr>"
                f"<td>{reason}</td>"
                f"<td>{stats.get('num_cycles', 0)}</td>"
                f"<td>{fmt(stats.get('avg_pnl_after_fees', 0.0))}</td>"
                f"<td>{fmt(stats.get('avg_net_lp_pnl', 0.0))}</td>"
                f"<td>{fmt(stats.get('avg_gas_cost', 0.0))}</td>"
                f"<td>{fmt(stats.get('avg_fees_total', 0.0))}</td>"
                f"<td>{fmt(stats.get('avg_vs_hodl', 0.0))}</td>"
                f"<td>{fmt(stats.get('avg_duration_min', 0.0))}</td>"
                f"<td>{fmt(stats.get('winrate', 0.0)*100.0)}%</td>"
                "</tr>"
            )
        html.append("</table>")
    else:
        html.append("<p>No grouped stats.</p>")

    html.append("</body></html>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))


# ---------------------------
# Main
# ---------------------------

def main():
    print("=== AI Engine: Analysing LP Cycle History ===")
    history_file = find_history_file()
    print(f"Using history file: {history_file}")

    cycles = load_cycles(history_file)
    if not cycles:
        print("❌ No cycles found in history file. Nothing to tune.")
        return

    # 1) AI tuning → ai_tuning.json (used by main_auto_v3)
    tuning = compute_tuning(cycles)
    with open(OUT_TUNING_FILE, "w", encoding="utf-8") as f:
        json.dump(tuning, f, indent=2)
    print(f"\n✅ Wrote AI tuning config to: {OUT_TUNING_FILE}")

    # 2) Full PnL / gas / fees stats
    print("\n=== Aggregate Performance ===")
    overall_stats = compute_overall_stats(cycles)
    for k, v in overall_stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print("\n=== Performance by exit_reason ===")
    by_reason = compute_stats_by_exit_reason(cycles)
    for reason, stats in by_reason.items():
        print(f"\n  exit_reason = {reason}")
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")

    # 3) CSV export
    export_csv(cycles, OUT_CSV_FILE)
    print(f"\n✅ Wrote detailed cycle CSV to: {OUT_CSV_FILE}")

    # 4) HTML report
    export_html_report(overall_stats, by_reason, OUT_HTML_FILE)
    print(f"✅ Wrote HTML summary report to: {OUT_HTML_FILE}")


if __name__ == "__main__":
    main()
