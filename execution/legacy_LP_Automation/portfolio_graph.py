import json
import matplotlib.pyplot as plt
from datetime import datetime

data = []
with open("portfolio_history.jsonl") as f:
    for line in f:
        try:
            data.append(json.loads(line))
        except:
            pass

if not data:
    print("No history data found!")
    exit()

times = [datetime.strptime(d["timestamp"], "%Y-%m-%d %H:%M:%S UTC") for d in data]
values = [d["actual_value_usd"] for d in data]
market = [d["market_effect_usd"] for d in data]
bot = [d["bot_effect_usd"] for d in data]

plt.figure(figsize=(10,6))
plt.plot(times, values, label="Total Portfolio (USD)", linewidth=2)
plt.plot(times, [v + b for v,b in zip(values, bot)], linestyle="--", label="Including Bot Effect")
plt.xlabel("Time (UTC)")
plt.ylabel("Value (USD)")
plt.title("Portfolio Performance Over Time")
plt.grid(True)
plt.legend()
plt.show()
