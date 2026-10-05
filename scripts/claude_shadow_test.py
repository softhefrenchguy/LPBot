"""Small (5-day) test of the isolated-call Claude-decision mechanism: build a point-in-time
snapshot from real historical production logs, make a FRESH API call per day (no shared
conversation state between days, no mention of the real current date), ask for target weights,
and log what Claude would have decided -- compared against what the real system actually did.
"""
import json
import os
import random

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

# --- load and combine all real historical log segments ---
eth_files = [
    "artifacts/paper_trade/daily_checks_log_pre_migration_20260930T080142.csv",
    "artifacts/paper_trade/daily_checks_log.csv",
]
frames = []
for f in eth_files:
    d = pd.read_csv(f, low_memory=False)
    frames.append(d)
eth = pd.concat(frames, ignore_index=True)
eth["date"] = pd.to_datetime(eth["date"], errors="coerce")
eth = eth.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last").sort_values("date")
eth = eth[[c for c in FIELDS if c in eth.columns]]
print(f"Combined history: {len(eth)} days, {eth['date'].min().date()} to {eth['date'].max().date()}")

# --- pick 5 real historical days spread across the history for this small test ---
random.seed(7)
idx = sorted(random.sample(range(len(eth)), 5))
sample_days = eth.iloc[idx]
print("Testing on these real historical days:", [d.date() for d in sample_days["date"]])

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

SYSTEM = """You are a quantitative risk-and-sizing assistant for a small crypto/gold trading account. \
Each message gives you a point-in-time snapshot of trend/momentum signals, volatility conditions, \
and news context for ETH, BTC, and gold (PAXG). Decide a target portfolio weight (0.0 to 1.0) for \
each of ETH and BTC for TODAY, given only the information in this message. Do not assume you know \
what happens after this date -- treat this as the only information you have ever seen. \
Respond with ONLY a JSON object: {"eth_weight": <0-1>, "btc_weight": <0-1>, "reasoning": "<2-3 sentences>"}"""

results = []
for _, row in sample_days.iterrows():
    snapshot = {k: (None if pd.isna(row[k]) else row[k]) for k in FIELDS if k in row.index}
    snapshot["date"] = str(row["date"].date())
    prompt = f"Today's market snapshot:\n{json.dumps(snapshot, indent=2, default=str)}\n\nWhat should today's ETH and BTC target weights be?"

    # Fresh call -- no prior messages, no shared state with any other day's call.
    resp = client.messages.create(
        model=MODEL,
        max_tokens=400,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        decision = json.loads(text[start:end])
    except Exception as exc:
        decision = {"parse_error": str(exc), "raw": text}

    real_eth_w = row.get("off_weight_scaled", float("nan"))
    real_btc_w = row.get("btc_combined_weight", float("nan"))
    results.append({
        "date": str(row["date"].date()),
        "claude_eth_w": decision.get("eth_weight"),
        "claude_btc_w": decision.get("btc_weight"),
        "claude_reasoning": decision.get("reasoning", decision.get("raw", "")),
        "real_eth_w": real_eth_w,
        "real_btc_w": real_btc_w,
    })
    print(f"\n=== {row['date'].date()} ===")
    print(f"  REAL:   eth_w={real_eth_w}  btc_w={real_btc_w}")
    print(f"  CLAUDE: eth_w={decision.get('eth_weight')}  btc_w={decision.get('btc_weight')}")
    print(f"  reasoning: {decision.get('reasoning', decision.get('raw',''))[:300]}")

out = pd.DataFrame(results)
out.to_csv("artifacts/paper_trade/_claude_shadow_test_sample.csv", index=False)
print("\nSaved to artifacts/paper_trade/_claude_shadow_test_sample.csv")
print(f"\nTotal input+output tokens this run (5 calls): check Anthropic usage dashboard for cost")
