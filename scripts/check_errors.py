import pandas as pd
df = pd.read_csv("artifacts/paper_trade/claude_shadow_context_history.csv")
err = df[df["claude_reasoning"].astype(str).str.startswith("ERROR")]
print(f"{len(err)} error rows")
for _, r in err.head(5).iterrows():
    print(f"  {r['date']}: {str(r['claude_reasoning'])[:200]}")
