import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

df = pd.read_csv("artifacts/backtest/ema_only_backtest.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

fig = make_subplots(
    rows=4,
    cols=1,
    shared_xaxes=True,
    subplot_titles=("Equity", "EMA + State", "Weight + Target", "Price"),
    vertical_spacing=0.06,
)

fig.add_trace(go.Scatter(x=df["timestamp"], y=df["eq"], name="Eq"), row=1, col=1)
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["ema"], name="EMA"), row=2, col=1)
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["state"], name="State"), row=2, col=1)
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["weight"], name="Weight"), row=3, col=1)
fig.add_trace(
    go.Scatter(x=df["timestamp"], y=df["target_weight"], name="Target"),
    row=3,
    col=1,
)
fig.add_trace(go.Scatter(x=df["timestamp"], y=df["close"], name="Price"), row=4, col=1)

fig.update_layout(height=1000, title="EMA‑Only Backtest (Costs + Rebalance Limits)")
out = "artifacts/backtest/ema_only_backtest.html"
fig.write_html(out)
print("wrote", out)
