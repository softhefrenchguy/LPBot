"""Score the context-enriched shadow run the same way as the baseline run, for a direct comparison."""
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

shadow = pd.read_csv("artifacts/paper_trade/claude_shadow_context_history.csv")
shadow["date"] = pd.to_datetime(shadow["date"])
shadow = shadow[~shadow["claude_reasoning"].astype(str).str.startswith("ERROR")]

df = shadow.merge(prices, on="date", how="left").sort_values("date").reset_index(drop=True)
for c in ["real_eth_w", "real_btc_w", "claude_eth_w", "claude_btc_w"]:
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


print(f"Window: {df['date'].min().date()} to {df['date'].max().date()}  ({len(df)} usable days, "
      f"dropped {shadow['claude_reasoning'].isna().sum()} rows with no valid decision)\n")
score(df["real_eth_w"], df["real_btc_w"], "Real rule-based system:")
score(df["claude_eth_w"], df["claude_btc_w"], "Claude WITH 10-day context:")

# agreement stats, same method as baseline
for asset in ["eth", "btc"]:
    real_col, claude_col = f"real_{asset}_w", f"claude_{asset}_w"
    real_exposed = (df[real_col] > 0.01)
    claude_exposed = (df[claude_col] > 0.01)
    agree = (real_exposed == claude_exposed).mean()
    both_exposed = df[real_exposed & claude_exposed]
    mean_abs_diff = (both_exposed[real_col] - both_exposed[claude_col]).abs().mean() if len(both_exposed) else float("nan")
    print(f"{asset.upper()}: flat-vs-exposed agreement = {agree:.1%}  "
          f"(real exposed {real_exposed.sum()}d, claude exposed {claude_exposed.sum()}d, "
          f"mean |diff| when both exposed = {mean_abs_diff:.3f})")
