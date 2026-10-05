"""Split the common 175-day window in half and score all three series (real, no-context Claude,
context Claude) separately in each half, to check whether the context-run's edge is consistent or
driven by one lucky stretch.
"""
import numpy as np
import pandas as pd

eth_files = [
    "artifacts/paper_trade/daily_checks_log_pre_migration_20260930T080142.csv",
    "artifacts/paper_trade/daily_checks_log.csv",
]
prices = pd.concat([pd.read_csv(f, low_memory=False) for f in eth_files], ignore_index=True)
prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
prices = prices.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last").sort_values("date")
prices = prices[["date", "eth_price", "btc_price"]]

no_ctx = pd.read_csv("artifacts/paper_trade/claude_shadow_full_history.csv")
no_ctx["date"] = pd.to_datetime(no_ctx["date"])
no_ctx = no_ctx[~no_ctx["claude_reasoning"].astype(str).str.startswith("ERROR")]
no_ctx = no_ctx.rename(columns={"claude_eth_w": "noctx_eth_w", "claude_btc_w": "noctx_btc_w"})

ctx = pd.read_csv("artifacts/paper_trade/claude_shadow_context_history.csv")
ctx["date"] = pd.to_datetime(ctx["date"])
ctx = ctx[~ctx["claude_reasoning"].astype(str).str.startswith("ERROR")]
ctx = ctx.rename(columns={"claude_eth_w": "ctx_eth_w", "claude_btc_w": "ctx_btc_w"})

df = no_ctx[["date", "real_eth_w", "real_btc_w", "noctx_eth_w", "noctx_btc_w"]].merge(
    ctx[["date", "ctx_eth_w", "ctx_btc_w"]], on="date", how="inner"
).merge(prices, on="date", how="left").sort_values("date").reset_index(drop=True)

for c in ["real_eth_w", "real_btc_w", "noctx_eth_w", "noctx_btc_w", "ctx_eth_w", "ctx_btc_w"]:
    df[c] = df[c].fillna(0.0)

df["eth_next_ret"] = df["eth_price"].pct_change().shift(-1)
df["btc_next_ret"] = df["btc_price"].pct_change().shift(-1)
df = df.iloc[:-1].reset_index(drop=True)

COST_BPS = 10.0


def score(sub, eth_w, btc_w, label):
    eth_w = eth_w.reset_index(drop=True)
    btc_w = btc_w.reset_index(drop=True)
    sub = sub.reset_index(drop=True)
    eth_turn = eth_w.diff().abs().fillna(eth_w.iloc[0])
    btc_turn = btc_w.diff().abs().fillna(btc_w.iloc[0])
    day_ret = eth_w * sub["eth_next_ret"] + btc_w * sub["btc_next_ret"]
    day_ret = day_ret - (eth_turn + btc_turn) * (COST_BPS / 10000.0)
    day_ret = day_ret.fillna(0.0)
    eq = (1 + day_ret).cumprod()
    years = len(day_ret) / 365.0
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 else float("nan")
    sharpe = day_ret.mean() / day_ret.std() * np.sqrt(365) if day_ret.std() > 0 else float("nan")
    maxdd = (eq / eq.cummax() - 1).min()
    total_ret = eq.iloc[-1] - 1
    print(f"  {label:34s} total_ret={total_ret:+7.1%}  CAGR={cagr:+7.1%}  Sharpe={sharpe:6.3f}  MaxDD={maxdd:7.1%}")


mid = len(df) // 2
halves = [("First half", df.iloc[:mid]), ("Second half", df.iloc[mid:])]

for label, sub in halves:
    print(f"\n{label}: {sub['date'].min().date()} to {sub['date'].max().date()}  ({len(sub)} days)")
    score(sub, sub["real_eth_w"], sub["real_btc_w"], "Real rule-based system:")
    score(sub, sub["noctx_eth_w"], sub["noctx_btc_w"], "Claude, single-day snapshot:")
    score(sub, sub["ctx_eth_w"], sub["ctx_btc_w"], "Claude, +10-day context:")
