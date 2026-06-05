import pandas as pd, numpy as np, json

src="artifacts/ob/ob_forward.csv"
out="artifacts/paper/ob_metrics.html"
roll=50

try:
    df = pd.read_csv(src)
except Exception as e:
    open(out,"w",encoding="utf-8").write(f"<pre>read error: {e}</pre>")
    raise SystemExit()

df = df.dropna(subset=["imbalance","ret_fwd"])
df["imbalance"] = pd.to_numeric(df["imbalance"], errors="coerce")
df["ret_fwd"] = pd.to_numeric(df["ret_fwd"], errors="coerce")
df = df.dropna(subset=["imbalance","ret_fwd"])

if len(df) < roll:
    open(out,"w",encoding="utf-8").write(
        f"<html><head><meta http-equiv='refresh' content='60'></head>"
        f"<body>Not enough rows yet: {len(df)} (need {roll})</body></html>"
    )
    raise SystemExit()

df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
df = df.sort_values("ts")

corr = df["imbalance"].rolling(roll).corr(df["ret_fwd"])
hit  = (np.sign(df["imbalance"]) == np.sign(df["ret_fwd"])).astype(int).rolling(roll).mean()
out_df = pd.DataFrame({"ts":df["ts"],"corr":corr,"hit":hit}).dropna()

labels = out_df["ts"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
corr_vals = [round(x,6) for x in out_df["corr"]]
hit_vals  = [round(x,6) for x in out_df["hit"]]

html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="60">
  <title>OB Metrics</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
  <h3>OB Metrics (rolling window={roll})</h3>
  <p>points: {len(corr_vals)}</p>
  <canvas id="corr" height="140"></canvas>
  <canvas id="hit" height="140"></canvas>
<script>
const labels={json.dumps(labels)};
const corrData={json.dumps(corr_vals)};
const hitData={json.dumps(hit_vals)};
new Chart(document.getElementById('corr'), {{
  type:'line',
  data:{{labels,datasets:[{{label:'Rolling Corr',data:corrData,borderColor:'#2b8cbe',borderWidth:2,pointRadius:0}}]}},
  options:{{scales:{{x:{{display:true}},y:{{display:true}}}}}}
}});
new Chart(document.getElementById('hit'), {{
  type:'line',
  data:{{labels,datasets:[{{label:'Rolling Hit Rate',data:hitData,borderColor:'#31a354',borderWidth:2,pointRadius:0}}]}},
  options:{{scales:{{x:{{display:true}},y:{{display:true,suggestedMin:0.4,suggestedMax:0.6}}}}}}
}});
</script>
</body>
</html>
"""
open(out,"w",encoding="utf-8").write(html)
print("wrote", out)
