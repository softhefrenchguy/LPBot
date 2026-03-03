import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

df = pd.read_csv("artifacts/backtest/breakout_backtest.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                    subplot_titles=("Equity", "Compression/Breakout", "Weight", "Price"),
                    vertical_spacing=0.06)

fig.add_trace(go.Scatter(x=df["timestamp"], y=df["eq"], name="Eq"), row=1, col=1)
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["compress"].astype(int), name="Compress"), row=2, col=1)
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["breakout"].astype(int), name="Breakout"), row=2, col=1)
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["weight"], name="Weight"), row=3, col=1)
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["close"], name="Price"), row=4, col=1)

fig.update_layout(height=1000, title="Breakout After Compression")
out = "artifacts/backtest/breakout_backtest.html"
fig.write_html(out)
print("wrote", out)
