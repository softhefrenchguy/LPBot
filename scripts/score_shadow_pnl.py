"""Score the full-history Claude-shadow run against what actually happened. For each day, apply
that day's weight (real vs claude) to the NEXT day's realized price return -- same lag convention
for both series, so the comparison is apples-to-apples and has no lookahead bias either way.
"""
import numpy as np
import pandas as pd

FIELDS = ["date", "eth_price", "btc_price"]

eth_files = [
    "artifacts/paper_trade/daily_checks_log_pre_migration_20260930T080142.csv",
    "artifacts/paper_trade/daily_checks_log.csv",
]
prices = pd.concat([pd.read_csv(f, low_memory=False) for f in eth_files], ignore_index=True)
prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
prices = prices.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last").sort_values("date")
prices = prices[FIELDS]

shadow = pd.read_csv("artifacts/paper_trade/claude_shadow_full_history.csv")
shadow["date"] = pd.to_datetime(shadow["date"])

df = shadow.merge(prices, on="date", how="left").sort_values("date").reset_index(drop=True)
df["real_eth_w"] = df["real_eth_w"].fillna(0.0)
df["real_btc_w"] = df["real_btc_w"].fillna(0.0)
df["claude_eth_w"] = df["claude_eth_w"].fillna(0.0)
df["claude_btc_w"] = df["claude_btc_w"].fillna(0.0)

df["eth_next_ret"] = df["eth_price"].pct_change().shift(-1)
df["btc_next_ret"] = df["btc_price"].pct_change().shift(-1)
df = df.iloc[:-1]  # drop last row, no next-day return available

COST_BPS = 10.0  # one-way, applied on weight changes


def score(eth_w: pd.Series, btc_w: pd.Series, label: str) -> None:
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
    print(f"{label:28s} total_ret={total_ret:+7.1%}  CAGR={cagr:+7.1%}  Sharpe={sharpe:6.3f}  MaxDD={maxdd:7.1%}")


print(f"Window: {df['date'].min().date()} to {df['date'].max().date()}  ({len(df)} days)\n")
score(df["real_eth_w"], df["real_btc_w"], "Real rule-based system:")
score(df["claude_eth_w"], df["claude_btc_w"], "Claude isolated-call shadow:")

# also: pure buy&hold of each asset over the same window, for scale reference
bh_eth = (1 + df["eth_next_ret"].fillna(0)).cumprod().iloc[-1] - 1
bh_btc = (1 + df["btc_next_ret"].fillna(0)).cumprod().iloc[-1] - 1
print(f"\n(reference) ETH buy&hold total return over window: {bh_eth:+.1%}")
print(f"(reference) BTC buy&hold total return over window: {bh_btc:+.1%}")
