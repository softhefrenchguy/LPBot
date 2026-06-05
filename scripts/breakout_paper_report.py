import argparse
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


def _perf_from_log_returns(log_r: pd.Series) -> dict:
    x = pd.to_numeric(log_r, errors="coerce").fillna(0.0)
    if len(x) == 0:
        return {"ret": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_dd": np.nan}
    bars_per_year = 365 * 24 * 12  # 5m bars
    eq = np.exp(np.cumsum(x.to_numpy()))
    peak = np.maximum.accumulate(eq)
    ann_vol = float(np.std(x.to_numpy()) * np.sqrt(bars_per_year))
    sharpe = float((np.mean(x.to_numpy()) * bars_per_year) / (ann_vol + 1e-12))
    return {
        "ret": float(eq[-1] - 1.0),
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": float((eq / peak - 1.0).min()),
    }


def _fmt_pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x:.2%}"


def _fmt_num(x: float, n: int = 4) -> str:
    return "" if pd.isna(x) else f"{x:.{n}f}"


def _write_html(out_path: Path, title: str, body: str) -> None:
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="60">
  <title>{title}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
    h2 {{ margin: 0 0 10px 0; }}
    .meta {{ margin-bottom: 16px; color: #444; }}
    .card {{ border: 1px solid #ddd; border-radius: 12px; padding: 16px; margin-bottom: 16px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
    th, td {{ border: 1px solid #eee; padding: 8px 10px; text-align: right; }}
    th {{ background: #f7f7f7; text-align: center; }}
  </style>
</head>
<body>
{body}
</body>
</html>"""
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Breakout paper report generator")
    p.add_argument("--log-csv", default="artifacts/paper/breakout_paper.csv")
    p.add_argument("--out", default="artifacts/paper/breakout_report.html")
    p.add_argument("--daily-out-csv", default="artifacts/paper/breakout_daily_summary.csv")
    p.add_argument("--daily-days", type=int, default=14)
    p.add_argument("--trade-cost-bps", type=float, default=5.0)
    p.add_argument("--v1-log-csv", default="artifacts/paper/paper_log_v2.csv")
    p.add_argument("--ob-forward-csv", default="artifacts/ob/ob_forward.csv")
    p.add_argument("--ob-roll", type=int, default=200)
    p.add_argument("--blend-v1-weight", type=float, default=0.7)
    p.add_argument("--score-windows-days", default="7,30")
    p.add_argument("--pass-ret-vs-spot-min", type=float, default=-0.01)
    p.add_argument("--pass-maxdd-vs-v1-max", type=float, default=0.0)
    p.add_argument("--pass-turnover-30d-max", type=float, default=25.0)
    p.add_argument("--max-rows", type=int, default=2000)
    p.add_argument("--interval-seconds", type=int, default=300)
    p.add_argument("--once", action="store_true")
    args = p.parse_args()

    log_path = Path(args.log_csv)
    out_path = Path(args.out)
    daily_out_path = Path(args.daily_out_csv)
    v1_log_path = Path(args.v1_log_csv)
    ob_forward_path = Path(args.ob_forward_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    daily_out_path.parent.mkdir(parents=True, exist_ok=True)
    score_windows = []
    for tok in str(args.score_windows_days).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            d = int(tok)
            if d > 0:
                score_windows.append(d)
        except ValueError:
            continue
    if not score_windows:
        score_windows = [7, 30]

    while True:
        if not log_path.exists():
            _write_html(out_path, "Breakout Report", f"<h2>Breakout Paper Report</h2><div class='meta'>log not found: {log_path}</div>")
            if args.once:
                break
            time.sleep(max(1, args.interval_seconds))
            continue

        df = pd.read_csv(log_path)
        if df.empty:
            _write_html(out_path, "Breakout Report", "<h2>Breakout Paper Report</h2><div class='meta'>no rows yet</div>")
            if args.once:
                break
            time.sleep(max(1, args.interval_seconds))
            continue

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        df_all = df.copy()
        df = df.tail(args.max_rows).copy()

        for col in ["close", "weight", "core_r", "eq"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "eq" not in df.columns or df["eq"].isna().all():
            if "core_r" in df.columns:
                r = df["core_r"].fillna(0.0)
                df["eq"] = np.exp(np.cumsum(r))
            else:
                df["eq"] = 1.0

        # Rebase equity to 1.0 for the displayed window
        first_eq = float(df["eq"].iloc[0]) if len(df) else 1.0
        if first_eq != 0:
            df["eq"] = df["eq"] / first_eq

        last_ts = df["timestamp"].iloc[-1] if "timestamp" in df.columns else None
        avg_weight = float(df["weight"].fillna(0.0).mean()) if "weight" in df.columns else 0.0
        time_in_market = float((df.get("weight", 0.0) > 0).mean()) * 100 if "weight" in df.columns else 0.0

        labels = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist() if "timestamp" in df.columns else list(range(len(df)))
        eq_vals = [round(x, 6) for x in df["eq"].fillna(0.0).tolist()]
        weight_vals = [round(x, 6) for x in df.get("weight", pd.Series([0.0]*len(df))).fillna(0.0).tolist()]
        close_vals = [round(x, 6) for x in df.get("close", pd.Series([0.0]*len(df))).fillna(0.0).tolist()]

        # V1 comparison series (rebased to 1.0 on first available point in this display window)
        v1_combined_vals = [None] * len(df)
        v1_core_vals = [None] * len(df)
        v1_spot_vals = [None] * len(df)
        blend_eq_vals = [None] * len(df)
        v1_meta = "v1 comparison unavailable"
        v1_full = None
        if v1_log_path.exists() and "timestamp" in df.columns and len(df) > 0:
            try:
                v1 = pd.read_csv(v1_log_path)
                if not v1.empty and {"timestamp", "combined_r", "core_r", "r"}.issubset(v1.columns):
                    v1["timestamp"] = pd.to_datetime(v1["timestamp"], utc=True, errors="coerce")
                    v1 = v1.dropna(subset=["timestamp"]).sort_values("timestamp")
                    for c in ["combined_r", "core_r", "r", "weight"]:
                        if c not in v1.columns:
                            v1[c] = 0.0
                        v1[c] = pd.to_numeric(v1[c], errors="coerce").fillna(0.0)
                    v1_full = v1.copy()

                    v1["combined_eq"] = np.exp(np.cumsum(v1["combined_r"].to_numpy()))
                    v1["core_eq"] = np.exp(np.cumsum(v1["core_r"].to_numpy()))
                    v1["spot_eq"] = np.exp(np.cumsum(v1["r"].to_numpy()))

                    ts0 = df["timestamp"].iloc[0]
                    ts1 = df["timestamp"].iloc[-1]
                    v1w = v1[(v1["timestamp"] >= ts0) & (v1["timestamp"] <= ts1)].copy()
                    if not v1w.empty:
                        for col in ["combined_eq", "core_eq", "spot_eq"]:
                            base = float(v1w[col].iloc[0])
                            if base != 0:
                                v1w[col] = v1w[col] / base
                        idx = pd.Index(df["timestamp"])
                        map_df = v1w.set_index("timestamp")
                        v1_combined_vals = [None if pd.isna(x) else round(float(x), 6) for x in map_df["combined_eq"].reindex(idx).tolist()]
                        v1_core_vals = [None if pd.isna(x) else round(float(x), 6) for x in map_df["core_eq"].reindex(idx).tolist()]
                        v1_spot_vals = [None if pd.isna(x) else round(float(x), 6) for x in map_df["spot_eq"].reindex(idx).tolist()]
                        v1_meta = f"v1 window: {ts0} -> {ts1}"
            except Exception as e:
                v1_meta = f"v1 comparison error: {e}"

        scorecard_html = "<div class='meta'>rolling scorecard unavailable</div>"
        if v1_full is not None and {"timestamp", "core_r", "weight"}.issubset(df_all.columns):
            try:
                br_full = df_all[["timestamp", "core_r", "weight"]].copy()
                br_full["core_r"] = pd.to_numeric(br_full["core_r"], errors="coerce").fillna(0.0)
                br_full["weight"] = pd.to_numeric(br_full["weight"], errors="coerce").fillna(0.0)
                br_full = br_full.dropna(subset=["timestamp"]).sort_values("timestamp")

                m = pd.merge(
                    v1_full[["timestamp", "combined_r", "r", "weight"]],
                    br_full,
                    on="timestamp",
                    how="inner",
                    suffixes=("_v1", "_br"),
                ).sort_values("timestamp")

                if len(m) > 0:
                    cost_k = args.trade_cost_bps / 10000.0
                    w_v1 = m["weight_v1"].fillna(0.0)
                    w_br = m["weight_br"].fillna(0.0)
                    r_v1 = m["combined_r"].fillna(0.0)
                    r_spot = m["r"].fillna(0.0)
                    r_br = m["core_r"].fillna(0.0)

                    cost_v1 = -w_v1.diff().abs().fillna(0.0) * cost_k
                    cost_br = -w_br.diff().abs().fillna(0.0) * cost_k
                    blend_a = float(args.blend_v1_weight)
                    blend_b = 1.0 - blend_a
                    w_bl = blend_a * w_v1 + blend_b * w_br
                    r_bl = blend_a * r_v1 + blend_b * r_br
                    cost_bl = -w_bl.diff().abs().fillna(0.0) * cost_k

                    m["ret_v1_net"] = r_v1 + cost_v1
                    m["ret_spot_net"] = r_spot
                    m["ret_br_net"] = r_br + cost_br
                    m["ret_blend_net"] = r_bl + cost_bl
                    m["w_v1"] = w_v1
                    m["w_br"] = w_br
                    m["w_bl"] = w_bl

                    # Blend equity in visible chart window
                    ts0 = df["timestamp"].iloc[0]
                    ts1 = df["timestamp"].iloc[-1]
                    mw = m[(m["timestamp"] >= ts0) & (m["timestamp"] <= ts1)].copy()
                    if len(mw) > 0:
                        mw["blend_eq"] = np.exp(np.cumsum(mw["ret_blend_net"].to_numpy()))
                        base = float(mw["blend_eq"].iloc[0])
                        if base != 0:
                            mw["blend_eq"] = mw["blend_eq"] / base
                        blend_map = mw.set_index("timestamp")["blend_eq"]
                        idx = pd.Index(df["timestamp"])
                        blend_eq_vals = [None if pd.isna(x) else round(float(x), 6) for x in blend_map.reindex(idx).tolist()]

                    # Rolling scorecard
                    score_rows = []
                    for d in score_windows:
                        end_ts = m["timestamp"].iloc[-1]
                        start_ts = end_ts - pd.Timedelta(days=d)
                        w = m[m["timestamp"] >= start_ts].copy()
                        if len(w) < 10:
                            continue
                        series_map = {
                            "Breakout Net": ("ret_br_net", "w_br"),
                            "V1 Combined Net": ("ret_v1_net", "w_v1"),
                            "Spot": ("ret_spot_net", None),
                            f"Blend {int(blend_a*100)}/{int(blend_b*100)} Net": ("ret_blend_net", "w_bl"),
                        }
                        for name, (ret_col, w_col) in series_map.items():
                            pstat = _perf_from_log_returns(w[ret_col])
                            avg_w = float(w[w_col].mean()) if w_col else np.nan
                            tim = float((w[w_col] > 0).mean() * 100.0) if w_col else 100.0
                            turnover = float(w[w_col].diff().abs().fillna(0.0).sum()) if w_col else 0.0
                            score_rows.append(
                                {
                                    "window_days": d,
                                    "strategy": name,
                                    "ret": pstat["ret"],
                                    "ann_vol": pstat["ann_vol"],
                                    "sharpe": pstat["sharpe"],
                                    "max_dd": pstat["max_dd"],
                                    "avg_weight": avg_w,
                                    "time_in_market_pct": tim,
                                    "turnover": turnover,
                                }
                            )

                    if score_rows:
                        score_df = pd.DataFrame(score_rows)
                        rows_html = []
                        for _, row in score_df.sort_values(["window_days", "strategy"]).iterrows():
                            avg_w_txt = "" if pd.isna(row["avg_weight"]) else f"{row['avg_weight']:.4f}"
                            tim_txt = "" if pd.isna(row["time_in_market_pct"]) else f"{row['time_in_market_pct']:.2f}%"
                            rows_html.append(
                                "<tr>"
                                f"<td style='text-align:center'>{int(row['window_days'])}d</td>"
                                f"<td style='text-align:left'>{row['strategy']}</td>"
                                f"<td>{_fmt_pct(row['ret'])}</td>"
                                f"<td>{_fmt_num(row['ann_vol'],4)}</td>"
                                f"<td>{_fmt_num(row['sharpe'],4)}</td>"
                                f"<td>{_fmt_pct(row['max_dd'])}</td>"
                                f"<td>{avg_w_txt}</td>"
                                f"<td>{tim_txt}</td>"
                                f"<td>{_fmt_num(row['turnover'],4)}</td>"
                                "</tr>"
                            )

                        pass_row = ""
                        s30 = score_df[score_df["window_days"] == 30]
                        if len(s30) >= 3:
                            try:
                                br30 = float(s30.loc[s30["strategy"] == "Breakout Net", "ret"].iloc[0])
                                sp30 = float(s30.loc[s30["strategy"] == "Spot", "ret"].iloc[0])
                                brdd30 = float(s30.loc[s30["strategy"] == "Breakout Net", "max_dd"].iloc[0])
                                v1dd30 = float(s30.loc[s30["strategy"] == "V1 Combined Net", "max_dd"].iloc[0])
                                brto30 = float(s30.loc[s30["strategy"] == "Breakout Net", "turnover"].iloc[0])

                                c1 = (br30 - sp30) >= float(args.pass_ret_vs_spot_min)
                                c2 = (brdd30 - v1dd30) <= float(args.pass_maxdd_vs_v1_max)
                                c3 = brto30 <= float(args.pass_turnover_30d_max)
                                status = "PASS" if (c1 and c2 and c3) else "FAIL"
                                pass_row = (
                                    "<div class='meta'>"
                                    f"30d guardrails: {status} | "
                                    f"(Breakout-Spot ret: {(br30-sp30):.2%} >= {args.pass_ret_vs_spot_min:.2%})={c1} | "
                                    f"(Breakout-V1 maxDD diff: {(brdd30-v1dd30):.2%} <= {args.pass_maxdd_vs_v1_max:.2%})={c2} | "
                                    f"(Breakout turnover: {brto30:.2f} <= {args.pass_turnover_30d_max:.2f})={c3}"
                                    "</div>"
                                )
                            except Exception:
                                pass_row = ""

                        scorecard_html = (
                            "<div class='card'>"
                            f"<div class='meta'>Rolling scorecard (net of est. trading cost {args.trade_cost_bps:.2f} bps/unit turnover)</div>"
                            + pass_row
                            + "<table><tr>"
                            "<th>Window</th><th>Strategy</th><th>Net Return</th><th>Ann Vol</th><th>Sharpe</th>"
                            "<th>Max DD</th><th>Avg Weight</th><th>Time In Market</th><th>Turnover</th>"
                            "</tr>"
                            + "".join(rows_html)
                            + "</table></div>"
                        )
            except Exception as e:
                scorecard_html = f"<div class='meta'>rolling scorecard error: {e}</div>"

        daily_table_html = "<div class='meta'>daily summary unavailable</div>"
        if {"timestamp", "weight", "core_r"}.issubset(df_all.columns):
            full = df_all.copy()
            full["weight"] = pd.to_numeric(full["weight"], errors="coerce").fillna(0.0)
            full["core_r"] = pd.to_numeric(full["core_r"], errors="coerce").fillna(0.0)
            full["weight_prev"] = full["weight"].shift(1).fillna(0.0)
            full["turnover"] = (full["weight"] - full["weight_prev"]).abs()
            full["trade_cost_r"] = -full["turnover"] * (args.trade_cost_bps / 10000.0)
            full["net_r"] = full["core_r"] + full["trade_cost_r"]
            full["active"] = full["weight_prev"] > 0
            full["date"] = full["timestamp"].dt.date

            records = []
            for d, g in full.groupby("date", sort=True):
                gross_eq = np.exp(np.cumsum(g["core_r"]))
                net_eq = np.exp(np.cumsum(g["net_r"]))
                net_peak = np.maximum.accumulate(net_eq)
                net_mdd = float((net_eq / net_peak - 1.0).min()) if len(net_eq) else 0.0
                active = g["active"]
                hit = float((g.loc[active, "core_r"] > 0).mean() * 100.0) if active.any() else np.nan

                records.append(
                    {
                        "date": str(d),
                        "bars": int(len(g)),
                        "daily_return_gross": float(gross_eq.iloc[-1] - 1.0),
                        "daily_return_net": float(net_eq.iloc[-1] - 1.0),
                        "daily_max_dd_net": net_mdd,
                        "turnover": float(g["turnover"].sum()),
                        "avg_weight": float(g["weight"].mean()),
                        "time_in_market_pct": float((g["weight"] > 0).mean() * 100.0),
                        "hit_rate_active_pct": hit,
                    }
                )

            daily = pd.DataFrame(records)
            daily.to_csv(daily_out_path, index=False)
            daily_show = daily.tail(args.daily_days).copy()
            daily_show = daily_show.iloc[::-1]

            rows_html = []
            for _, row in daily_show.iterrows():
                hit_txt = "" if pd.isna(row["hit_rate_active_pct"]) else f"{row['hit_rate_active_pct']:.2f}%"
                rows_html.append(
                    "<tr>"
                    f"<td style='text-align:center'>{row['date']}</td>"
                    f"<td>{int(row['bars'])}</td>"
                    f"<td>{row['daily_return_gross']:.4%}</td>"
                    f"<td>{row['daily_return_net']:.4%}</td>"
                    f"<td>{row['daily_max_dd_net']:.4%}</td>"
                    f"<td>{row['turnover']:.4f}</td>"
                    f"<td>{row['avg_weight']:.4f}</td>"
                    f"<td>{row['time_in_market_pct']:.2f}%</td>"
                    f"<td>{hit_txt}</td>"
                    "</tr>"
                )

            daily_table_html = (
                "<div class='card'>"
                f"<div class='meta'>Daily summary (last {args.daily_days} days) | estimated cost: {args.trade_cost_bps:.2f} bps per unit turnover | csv: {daily_out_path}</div>"
                "<table>"
                "<tr>"
                "<th>Date</th><th>Bars</th><th>Gross Return</th><th>Net Return</th><th>Net Max DD</th>"
                "<th>Turnover</th><th>Avg Weight</th><th>Time In Market</th><th>Hit Rate (active)</th>"
                "</tr>"
                + "".join(rows_html)
                + "</table></div>"
            )

        ob_table_html = "<div class='meta'>order book forward summary unavailable</div>"
        if ob_forward_path.exists():
            try:
                ob = pd.read_csv(ob_forward_path)
                if not ob.empty and {"ts", "imbalance", "ret_fwd"}.issubset(ob.columns):
                    if "horizon_sec" not in ob.columns:
                        ob["horizon_sec"] = 60
                    ob["imbalance"] = pd.to_numeric(ob["imbalance"], errors="coerce")
                    ob["ret_fwd"] = pd.to_numeric(ob["ret_fwd"], errors="coerce")
                    ob = ob.dropna(subset=["imbalance", "ret_fwd"])

                    if not ob.empty:
                        ts_num = pd.to_numeric(ob["ts"], errors="coerce")
                        if ts_num.notna().mean() > 0.9:
                            ob["ts"] = pd.to_datetime(ts_num, unit="s", utc=True, errors="coerce")
                        else:
                            ob["ts"] = pd.to_datetime(ob["ts"], utc=True, errors="coerce")
                        ob = ob.dropna(subset=["ts"]).sort_values("ts")

                        roll = max(10, int(args.ob_roll))
                        ob_rows_html = []
                        for horizon, g in ob.groupby("horizon_sec", sort=True):
                            g = g.sort_values("ts")
                            corr = g["imbalance"].corr(g["ret_fwd"])
                            hit = (np.sign(g["imbalance"]) == np.sign(g["ret_fwd"])).mean() * 100.0
                            r_corr = g["imbalance"].rolling(roll).corr(g["ret_fwd"]).dropna()
                            r_hit = (
                                (np.sign(g["imbalance"]) == np.sign(g["ret_fwd"]))
                                .astype(float)
                                .rolling(roll)
                                .mean()
                                .dropna()
                                * 100.0
                            )
                            last_rc = float(r_corr.iloc[-1]) if len(r_corr) else np.nan
                            last_rh = float(r_hit.iloc[-1]) if len(r_hit) else np.nan
                            last_ts_h = g["ts"].iloc[-1]

                            corr_txt = "" if pd.isna(corr) else f"{float(corr):.4f}"
                            hit_txt = "" if pd.isna(hit) else f"{float(hit):.2f}%"
                            last_rc_txt = "" if pd.isna(last_rc) else f"{last_rc:.4f}"
                            last_rh_txt = "" if pd.isna(last_rh) else f"{last_rh:.2f}%"

                            ob_rows_html.append(
                                "<tr>"
                                f"<td style='text-align:center'>{int(horizon)}</td>"
                                f"<td>{int(len(g))}</td>"
                                f"<td style='text-align:center'>{last_ts_h}</td>"
                                f"<td>{corr_txt}</td>"
                                f"<td>{hit_txt}</td>"
                                f"<td>{last_rc_txt}</td>"
                                f"<td>{last_rh_txt}</td>"
                                "</tr>"
                            )

                        if ob_rows_html:
                            ob_table_html = (
                                "<div class='card'>"
                                f"<div class='meta'>Order book forward summary | csv: {ob_forward_path} | rolling window: {roll}</div>"
                                "<table>"
                                "<tr>"
                                "<th>Horizon (sec)</th><th>Rows</th><th>Last TS</th><th>Overall Corr</th>"
                                "<th>Overall Hit</th><th>Last Rolling Corr</th><th>Last Rolling Hit</th>"
                                "</tr>"
                                + "".join(ob_rows_html)
                                + "</table></div>"
                            )
            except Exception as e:
                ob_table_html = f"<div class='meta'>order book summary error: {e}</div>"

        blend_v1_pct = int(round(float(args.blend_v1_weight) * 100))
        blend_br_pct = int(round((1.0 - float(args.blend_v1_weight)) * 100))

        body = f"""
<h2>Breakout Paper Report</h2>
<div class="meta">rows: {len(df)} | last_ts: {last_ts} | avg_weight: {avg_weight:.4f} | time_in_market: {time_in_market:.2f}%</div>
<div class="meta">{v1_meta}</div>
{scorecard_html}
{daily_table_html}
{ob_table_html}
<div class="card">
  <canvas id="eq" height="140"></canvas>
</div>
<div class="card">
  <canvas id="eqcmp" height="140"></canvas>
</div>
<div class="card">
  <canvas id="weight" height="120"></canvas>
</div>
<div class="card">
  <canvas id="price" height="140"></canvas>
</div>
<script>
const labels = {json.dumps(labels)};
const eqData = {json.dumps(eq_vals)};
const v1CombinedData = {json.dumps(v1_combined_vals)};
const v1CoreData = {json.dumps(v1_core_vals)};
const v1SpotData = {json.dumps(v1_spot_vals)};
const blendEqData = {json.dumps(blend_eq_vals)};
const weightData = {json.dumps(weight_vals)};
const priceData = {json.dumps(close_vals)};

new Chart(document.getElementById('eq'), {{
  type: 'line',
  data: {{ labels, datasets: [{{ label: 'Equity', data: eqData, borderColor: '#2b8cbe', borderWidth: 2, pointRadius: 0 }}] }},
  options: {{ animation: false, plugins: {{ legend: {{ display: true }} }}, scales: {{ x: {{ display: false }} }} }}
}});
new Chart(document.getElementById('eqcmp'), {{
  type: 'line',
  data: {{
    labels,
    datasets: [
      {{ label: 'Breakout Eq', data: eqData, borderColor: '#2b8cbe', borderWidth: 2, pointRadius: 0 }},
      {{ label: 'V1 Combined Eq', data: v1CombinedData, borderColor: '#31a354', borderWidth: 2, pointRadius: 0 }},
      {{ label: 'V1 Core Eq', data: v1CoreData, borderColor: '#fdae61', borderWidth: 2, pointRadius: 0 }},
      {{ label: 'V1 Spot Eq', data: v1SpotData, borderColor: '#7f7f7f', borderWidth: 2, pointRadius: 0 }},
      {{ label: 'Blend {blend_v1_pct}/{blend_br_pct} Net Eq', data: blendEqData, borderColor: '#756bb1', borderWidth: 2, pointRadius: 0 }}
    ]
  }},
  options: {{ animation: false, plugins: {{ legend: {{ display: true }} }}, scales: {{ x: {{ display: false }} }} }}
}});
new Chart(document.getElementById('weight'), {{
  type: 'line',
  data: {{ labels, datasets: [{{ label: 'Weight', data: weightData, borderColor: '#fdae61', borderWidth: 2, pointRadius: 0 }}] }},
  options: {{ animation: false, plugins: {{ legend: {{ display: true }} }}, scales: {{ x: {{ display: false }}, y: {{ min: 0, max: 1 }} }} }}
}});
new Chart(document.getElementById('price'), {{
  type: 'line',
  data: {{ labels, datasets: [{{ label: 'Price', data: priceData, borderColor: '#31a354', borderWidth: 2, pointRadius: 0 }}] }},
  options: {{ animation: false, plugins: {{ legend: {{ display: true }} }}, scales: {{ x: {{ display: false }} }} }}
}});
</script>
"""
        _write_html(out_path, "Breakout Report", body)

        if args.once:
            break
        time.sleep(max(1, args.interval_seconds))


if __name__ == "__main__":
    main()
