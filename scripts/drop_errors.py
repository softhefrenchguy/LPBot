import pandas as pd
path = "artifacts/paper_trade/claude_shadow_context_history.csv"
df = pd.read_csv(path)
before = len(df)
df = df[~df["claude_reasoning"].astype(str).str.startswith("ERROR")]
df.to_csv(path, index=False)
print(f"Dropped {before - len(df)} error rows, {len(df)} remain.")
