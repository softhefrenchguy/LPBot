"""Score real system, no-context Claude, and context-enriched Claude all on the EXACT same set of
dates (intersection of both runs' valid days), so the three numbers are directly comparable -- the
context run's window starts 10 days later and dropped a few more error days, so a naive side-by-side
of the two prior scoring runs was comparing slightly different windows.
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
df = df.iloc[:-1]

COST_BPS = 10.0


def score(eth_w, btc_w, label):
    eth_turn = eth_w.diff().abs().fillna(eth_w.iloc[0])
    btc_turn = btc_w.diff().abs().fillna(btc_w.iloc[0])
    day_ret = eth_w * df["eth_next_ret"] + btc_w * df["btc_next_ret"]
    day_ret = day_ret - (eth_turn + btc_turn) * (COST_BPS / 10000.0)
    day_ret = day_ret.fillna(0.0)
    eq = (1 + day_ret).cumprod()
    years = len(day_ret) / 365.0
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 else float("nan")
    sharpe = day_ret.mean() / day_ret.std() * np.sqrt(365) if day_ret.std() > 0 else float("nan")
    maxdd = (eq / eq.cummax() - 1).min()
    total_ret = eq.iloc[-1] - 1
    print(f"{label:34s} total_ret={total_ret:+7.1%}  CAGR={cagr:+7.1%}  Sharpe={sharpe:6.3f}  MaxDD={maxdd:7.1%}")


print(f"Common window (all 3 series have valid data): {df['date'].min().date()} to {df['date'].max().date()}  ({len(df)} days)\n")
score(df["real_eth_w"], df["real_btc_w"], "Real rule-based system:")
score(df["noctx_eth_w"], df["noctx_btc_w"], "Claude, single-day snapshot:")
score(df["ctx_eth_w"], df["ctx_btc_w"], "Claude, +10-day context:")
