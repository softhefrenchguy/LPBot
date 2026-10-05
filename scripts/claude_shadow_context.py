"""Same isolated-call Claude-decision mechanism as claude_shadow_full.py, but each call now also
gets a trailing N-day history of the core signal fields, to test whether lack of multi-day context
(not the prompt's philosophy -- that's held identical to the baseline run) explains the baseline
run's miscalibration. Same system prompt, same gate logic, same scoring method -- context is the
only variable changed, so the two runs are directly comparable.
"""
import json
import os
import time

import anthropic
import pandas as pd

MODEL = "claude-sonnet-5"
HISTORY_DAYS = 10

FULL_FIELDS = [
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

# condensed subset carried for each of the trailing history days (keep token cost sane)
HISTORY_FIELDS = [
    "date", "eth_price", "regime", "stack_aligned", "entry_threshold_met",
    "vol_regime", "conviction_score", "conviction_bucket",
    "btc_price", "btc_regime", "btc_stack_aligned", "btc_entry_threshold_met",
    "btc_conviction_score", "btc_conviction_bucket",
]

OUT_PATH = "artifacts/paper_trade/claude_shadow_context_history.csv"

eth_files = [
    "artifacts/paper_trade/daily_checks_log_pre_migration_20260930T080142.csv",
    "artifacts/paper_trade/daily_checks_log.csv",
]
frames = [pd.read_csv(f, low_memory=False) for f in eth_files]
eth = pd.concat(frames, ignore_index=True)
eth["date"] = pd.to_datetime(eth["date"], errors="coerce")
eth = eth.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last").sort_values("date")
eth = eth[[c for c in FULL_FIELDS if c in eth.columns]].reset_index(drop=True)
print(f"Combined history: {len(eth)} days, {eth['date'].min().date()} to {eth['date'].max().date()}", flush=True)

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# same system prompt as the baseline run -- philosophy unchanged, only the user message gains history
SYSTEM = """You are a quantitative risk-and-sizing assistant for a small crypto/gold trading account. \
Each message gives you a point-in-time snapshot of trend/momentum signals, volatility conditions, \
and news context for ETH, BTC, and gold (PAXG). Decide a target portfolio weight (0.0 to 1.0) for \
each of ETH and BTC for TODAY, given only the information in this message. Do not assume you know \
what happens after this date -- treat this as the only information you have ever seen. \
Respond with ONLY a JSON object: {"eth_weight": <0-1>, "btc_weight": <0-1>, "reasoning": "<2-3 sentences>"}"""

done_dates = set()
if os.path.exists(OUT_PATH):
    prior = pd.read_csv(OUT_PATH)
    done_dates = set(prior["date"].astype(str))
    print(f"Resuming: {len(done_dates)} days already done.", flush=True)

n_errors = 0
total_in_tok = 0
total_out_tok = 0

for i, row in eth.iterrows():
    if i < HISTORY_DAYS:
        continue  # not enough trailing history yet
    date_str = str(row["date"].date())
    if date_str in done_dates:
        continue

    hist_rows = eth.iloc[i - HISTORY_DAYS:i]
    history = []
    for _, hrow in hist_rows.iterrows():
        d = {k: (None if pd.isna(hrow[k]) else hrow[k]) for k in HISTORY_FIELDS if k in hrow.index}
        d["date"] = str(hrow["date"].date())
        history.append(d)

    today_snapshot = {k: (None if pd.isna(row[k]) else row[k]) for k in FULL_FIELDS if k in row.index}
    today_snapshot["date"] = date_str

    prompt = (
        f"Trailing {HISTORY_DAYS}-day history (oldest first) of key signals:\n"
        f"{json.dumps(history, indent=2, default=str)}\n\n"
        f"Today's full market snapshot:\n{json.dumps(today_snapshot, indent=2, default=str)}\n\n"
        f"What should today's ETH and BTC target weights be?"
    )

    try:
        resp = client.messages.create(
            model=MODEL, max_tokens=400, system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        total_in_tok += resp.usage.input_tokens
        total_out_tok += resp.usage.output_tokens
        text_blocks = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        text = text_blocks[0].strip() if text_blocks else ""
        start = text.index("{")
        end = text.rindex("}") + 1
        raw_json = text[start:end]
        try:
            decision = json.loads(raw_json)
        except json.JSONDecodeError:
            import re
            decision = json.loads(re.sub(r",\s*([}\]])", r"\1", raw_json))
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
    pd.DataFrame([row_out]).to_csv(OUT_PATH, mode="a", header=not os.path.exists(OUT_PATH), index=False)

    if (i + 1) % 20 == 0 or (i + 1) == len(eth):
        print(f"  [{i+1}/{len(eth)}] {date_str}: claude eth={claude_eth_w} btc={claude_btc_w} | "
              f"real eth={row_out['real_eth_w']} btc={row_out['real_btc_w']}", flush=True)

    time.sleep(0.3)

print(f"\nDone. {n_errors} errors. ~{total_in_tok} input / {total_out_tok} output tokens this run.", flush=True)
