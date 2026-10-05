import pandas as pd

df = pd.read_csv("artifacts/paper_trade/claude_shadow_full_history.csv")
df["real_eth_w"] = df["real_eth_w"].fillna(0)
df["real_btc_w"] = df["real_btc_w"].fillna(0)

extra = df[(df["real_eth_w"] < 0.01) & (df["claude_eth_w"] > 0.01)]
print(f"Sample days Claude took ETH exposure the real system did not (n={len(extra)}):")
for _, r in extra.sample(min(6, len(extra)), random_state=1).iterrows():
    print(f"  {r['date']}: claude_eth_w={r['claude_eth_w']:.2f} | {str(r['claude_reasoning'])[:220]}")

print()
under = df[(df["real_eth_w"] > 0.01) & ((df["real_eth_w"] - df["claude_eth_w"]) > 0.15)]
print(f"Sample days Claude under-sized vs a real ETH position (n={len(under)}):")
for _, r in under.sample(min(5, len(under)), random_state=1).iterrows():
    print(f"  {r['date']}: real={r['real_eth_w']:.2f} claude={r['claude_eth_w']:.2f} | {str(r['claude_reasoning'])[:220]}")
