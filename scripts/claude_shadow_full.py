"""Full-history run of the isolated-call Claude-decision mechanism (see claude_shadow_test.py for
the 5-day pilot this scales up). One FRESH API call per real historical day -- no shared
conversation state between days, no mention of the real current date -- asking for target ETH/BTC
weights from only that day's point-in-time signal snapshot. Logs every day's decision next to what
the real rule-based system actually did, then prints an agreement-rate summary.
"""
import json
import os
import sys
import time

import anthropic
import pandas as pd

MODEL = "claude-sonnet-5"

FIELDS = [
    "date", "regime", "regime_days",
    "eth_price", "eth_24h_pct", "ema21", "ema55", "ema144",
    "stack_aligned", "stack_aligned_days", "entry_threshold_met", "off_days_since_break",
    "off_position", "off_weight_scaled", "off_hold_return", "off_days_held", "dd_20d",
    "vol_regime", "vol_percentile", "conviction_score", "conviction_bucket",
    "transition_strength", "transition_multiplier",
    "btc_price", "btc_24h_pct", "btc_ema15", "btc_ema40", "btc_ema120",
    "btc_stack_aligned", "btc_stack_aligned_days", "btc_entry_threshold_met",
    "btc_position", "btc_combined_weight", "btc_off_hold_return", "btc_off_days_held",
    "btc_regime", "btc_funding_z", "btc_conviction_score", "btc_conviction_bucket",
    "btc_transition_strength", "btc_transition_multiplier",
    "gold_price", "gold_stack_aligned", "gold_position_active", "gold_entry_threshold_met",
    "news_severity", "news_direction", "news_summary",
    "dvol_atm_iv_30d", "dvol_iv_percentile", "dvol_options_vol_regime", "dvol_rv_vol_regime", "dvol_agreement",
]

OUT_PATH = "artifacts/paper_trade/claude_shadow_full_history.csv"

# --- load and combine all real historical log segments (same source files as the pilot) ---
eth_files = [
    "artifacts/paper_trade/daily_checks_log_pre_migration_20260930T080142.csv",
    "artifacts/paper_trade/daily_checks_log.csv",
]
frames = [pd.read_csv(f, low_memory=False) for f in eth_files]
eth = pd.concat(frames, ignore_index=True)
eth["date"] = pd.to_datetime(eth["date"], errors="coerce")
eth = eth.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last").sort_values("date")
eth = eth[[c for c in FIELDS if c in eth.columns]].reset_index(drop=True)
print(f"Combined history: {len(eth)} days, {eth['date'].min().date()} to {eth['date'].max().date()}", flush=True)

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

SYSTEM = """You are a quantitative risk-and-sizing assistant for a small crypto/gold trading account. \
Each message gives you a point-in-time snapshot of trend/momentum signals, volatility conditions, \
and news context for ETH, BTC, and gold (PAXG). Decide a target portfolio weight (0.0 to 1.0) for \
each of ETH and BTC for TODAY, given only the information in this message. Do not assume you know \
what happens after this date -- treat this as the only information you have ever seen. \
Respond with ONLY a JSON object: {"eth_weight": <0-1>, "btc_weight": <0-1>, "reasoning": "<2-3 sentences>"}"""

# resume support: skip days already done if this got interrupted and re-run
done_dates = set()
if os.path.exists(OUT_PATH):
    prior = pd.read_csv(OUT_PATH)
    done_dates = set(prior["date"].astype(str))
    print(f"Resuming: {len(done_dates)} days already done, skipping those.", flush=True)

results = []
total_in_tok = 0
total_out_tok = 0
n_errors = 0

for i, row in eth.iterrows():
    date_str = str(row["date"].date())
    if date_str in done_dates:
        continue

    snapshot = {k: (None if pd.isna(row[k]) else row[k]) for k in FIELDS if k in row.index}
    snapshot["date"] = date_str
    prompt = f"Today's market snapshot:\n{json.dumps(snapshot, indent=2, default=str)}\n\nWhat should today's ETH and BTC target weights be?"

    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=400,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        total_in_tok += resp.usage.input_tokens
        total_out_tok += resp.usage.output_tokens
        text = resp.content[0].text.strip()
        start = text.index("{")
        end = text.rindex("}") + 1
        decision = json.loads(text[start:end])
        claude_eth_w = decision.get("eth_weight")
        claude_btc_w = decision.get("btc_weight")
        reasoning = decision.get("reasoning", "")
    except Exception as exc:
        n_errors += 1
        claude_eth_w = None
        claude_btc_w = None
        reasoning = f"ERROR: {exc}"

    row_out = {
        "date": date_str,
        "claude_eth_w": claude_eth_w,
        "claude_btc_w": claude_btc_w,
        "claude_reasoning": reasoning,
        "real_eth_w": row.get("off_weight_scaled"),
        "real_btc_w": row.get("btc_combined_weight"),
    }
    results.append(row_out)

    # append-as-we-go so a crash mid-run doesn't lose progress; also update done_dates
    pd.DataFrame([row_out]).to_csv(OUT_PATH, mode="a", header=not os.path.exists(OUT_PATH), index=False)

    if (i + 1) % 20 == 0 or (i + 1) == len(eth):
        print(f"  [{i+1}/{len(eth)}] {date_str}: claude eth={claude_eth_w} btc={claude_btc_w} | real eth={row_out['real_eth_w']} btc={row_out['real_btc_w']}", flush=True)

    time.sleep(0.3)  # gentle pacing, avoid rate limits

print(f"\nDone. {n_errors} errors. ~{total_in_tok} input / {total_out_tok} output tokens this run.", flush=True)

# --- summary stats ---
full = pd.read_csv(OUT_PATH)
for asset in ["eth", "btc"]:
    real_col, claude_col = f"real_{asset}_w", f"claude_{asset}_w"
    sub = full.dropna(subset=[claude_col])
    real_exposed = (sub[real_col].fillna(0) > 0.01)
    claude_exposed = (sub[claude_col].fillna(0) > 0.01)
    agree = (real_exposed == claude_exposed).mean()
    both_exposed = sub[real_exposed & claude_exposed]
    mean_abs_diff = (both_exposed[real_col] - both_exposed[claude_col]).abs().mean() if len(both_exposed) else float("nan")
    print(f"\n{asset.upper()}: flat-vs-exposed agreement = {agree:.1%} over {len(sub)} days "
          f"(real exposed {real_exposed.sum()}d, claude exposed {claude_exposed.sum()}d, "
          f"both exposed {len(both_exposed)}d, mean |weight diff| when both exposed = {mean_abs_diff:.3f})")
