from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_values(raw: str) -> tuple[set[str], set[float]]:
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    str_vals: set[str] = set()
    num_vals: set[float] = set()
    for p in parts:
        str_vals.add(p.lower())
        try:
            num_vals.add(float(p))
        except Exception:
            pass
    return str_vals, num_vals


def _parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def _match_values(series: pd.Series, str_vals: set[str], num_vals: set[float]) -> pd.Series:
    mask = pd.Series(False, index=series.index)
    if str_vals:
        mask = mask | series.astype(str).str.lower().isin(str_vals)
    if num_vals:
        num = pd.to_numeric(series, errors="coerce")
        mask = mask | num.isin(num_vals)
    return mask


def _load_price(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["timestamp", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)


def _load_exposure(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["timestamp", "weight"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)


def _load_regime(path: Path, regime_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns or regime_col not in df.columns:
        raise ValueError(f"regime csv missing required columns timestamp/{regime_col}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp")[["timestamp", regime_col]].rename(columns={regime_col: "regime"}).reset_index(drop=True)


def _bull_override_signal(
    ts: pd.Series,
    close: pd.Series,
    bar_minutes: int,
    timeframe: str,
    ema: int,
    slope_ema: int,
) -> pd.Series:
    tf_map = {"same": f"{bar_minutes}min", "1h": "1h", "4h": "4h", "1d": "1d"}
    tf = tf_map[timeframe]
    tmp = pd.DataFrame({"close": close.to_numpy()}, index=ts)

    if tf == f"{bar_minutes}min":
        tclose = tmp["close"]
        level = tclose.ewm(span=ema, adjust=False).mean()
        slope_src = tclose.ewm(span=slope_ema, adjust=False).mean()
        slope = slope_src.diff()
        return ((tclose > level) & (slope > 0)).fillna(False)

    htf_close = tmp["close"].resample(tf).last().shift(1).ffill()
    level = htf_close.ewm(span=ema, adjust=False).mean()
    slope_src = htf_close.ewm(span=slope_ema, adjust=False).mean()
    slope = slope_src.diff()
    bull_htf = ((htf_close > level) & (slope > 0)).fillna(False)
    return bull_htf.reindex(tmp.index, method="ffill").fillna(False)


def _perf(log_r: pd.Series, bar_minutes: int) -> dict[str, float]:
    x = pd.to_numeric(log_r, errors="coerce").fillna(0.0)
    if len(x) == 0:
        return {"ret": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "mdd": np.nan}
    bars_per_year = 365 * 24 * (60 / bar_minutes)
    eq = np.exp(np.cumsum(x.to_numpy()))
    peak = np.maximum.accumulate(eq)
    ann_vol = float(np.std(x.to_numpy()) * np.sqrt(bars_per_year))
    sharpe = float((np.mean(x.to_numpy()) * bars_per_year) / (ann_vol + 1e-12))
    return {
        "ret": float(eq[-1] - 1.0),
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "mdd": float((eq / peak - 1.0).min()),
    }


def _write_top_html(path: Path, top: pd.DataFrame, meta: list[str]) -> None:
    meta_html = "".join([f"<div class='meta'>{m}</div>" for m in meta])
    rows = []
    for _, r in top.iterrows():
        rows.append(
            "<tr>"
            f"<td>{int(r['rank'])}</td>"
            f"<td>{r['neutral_scale']:.2f}</td>"
            f"<td>{r['riskoff_scale']:.2f}</td>"
            f"<td>{int(r['bull_ema'])}</td>"
            f"<td>{int(r['bull_slope_ema'])}</td>"
            f"<td>{r['bull_min_weight']:.2f}</td>"
            f"<td>{r['ret_90d']:.2%}</td>"
            f"<td>{r['sharpe_90d']:.3f}</td>"
            f"<td>{r['mdd_90d']:.2%}</td>"
            f"<td>{r['ret_365d']:.2%}</td>"
            f"<td>{r['sharpe_365d']:.3f}</td>"
            f"<td>{r['mdd_365d']:.2%}</td>"
            f"<td>{r['excess_90d']:.2%}</td>"
            f"<td>{r['excess_365d']:.2%}</td>"
            f"<td>{r['score']:.4f}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>HMM+ENet Scale Sweep Top</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color:#111; }}
.meta {{ margin-bottom: 8px; color:#444; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border: 1px solid #eee; padding: 6px 8px; text-align: right; }}
th {{ background:#f7f7f7; text-align: center; }}
</style></head><body>
<h2>HMM + ENet Scaler Sweep Top 5</h2>
{meta_html}
<table><tr>
<th>Rank</th><th>Neutral</th><th>RiskOff</th><th>Bull EMA</th><th>Bull Slope EMA</th><th>Bull Floor</th>
<th>Ret90</th><th>Sh90</th><th>MDD90</th><th>Ret365</th><th>Sh365</th><th>MDD365</th><th>Excess90</th><th>Excess365</th><th>Score</th>
</tr>{''.join(rows)}</table></body></html>"""
    path.write_text(html, encoding="utf-8")


def _write_curve_html(path: Path, df: pd.DataFrame, title: str, subtitle: str) -> None:
    d = df.tail(3000).copy()
    d["eq"] = np.exp(np.cumsum(pd.to_numeric(d["strat_r"], errors="coerce").fillna(0.0).to_numpy()))
    d["eq_spot"] = np.exp(np.cumsum(pd.to_numeric(d["spot_r"], errors="coerce").fillna(0.0).to_numpy()))
    d["eq"] = d["eq"] / float(d["eq"].iloc[0] + 1e-12)
    d["eq_spot"] = d["eq_spot"] / float(d["eq_spot"].iloc[0] + 1e-12)

    labels = d["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
    eq = [round(float(x), 6) for x in d["eq"].tolist()]
    sp = [round(float(x), 6) for x in d["eq_spot"].tolist()]
    w = [round(float(x), 6) for x in pd.to_numeric(d["weight"], errors="coerce").fillna(0.0).tolist()]
    rs = [round(float(x), 6) for x in pd.to_numeric(d["regime_scale"], errors="coerce").fillna(0.0).tolist()]
    rf = [int(x) for x in pd.to_numeric(d["riskoff"], errors="coerce").fillna(0.0).tolist()]
    bu = [int(x) for x in pd.to_numeric(d["bull_on"], errors="coerce").fillna(0.0).tolist()]

    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>body{{font-family:Arial,sans-serif;margin:24px;color:#111}}.card{{border:1px solid #ddd;border-radius:12px;padding:16px;margin-bottom:16px}}.meta{{margin-bottom:8px;color:#444}}</style>
</head><body><h2>{title}</h2><div class='meta'>{subtitle}</div>
<div class="card"><canvas id="eq" height="120"></canvas></div>
<div class="card"><canvas id="w" height="100"></canvas></div>
<div class="card"><canvas id="s" height="90"></canvas></div>
<script>
const labels={json.dumps(labels)};
const eq={json.dumps(eq)}, sp={json.dumps(sp)}, w={json.dumps(w)}, rs={json.dumps(rs)}, rf={json.dumps(rf)}, bu={json.dumps(bu)};
const base={{animation:false,plugins:{{legend:{{display:true}}}},scales:{{x:{{display:false}}}}}};
new Chart(document.getElementById('eq'),{{type:'line',data:{{labels,datasets:[{{label:'Strategy Eq',data:eq,borderColor:'#2ca02c',borderWidth:2,pointRadius:0}},{{label:'Spot Eq',data:sp,borderColor:'#7f7f7f',borderWidth:2,pointRadius:0}}]}},options:base}});
new Chart(document.getElementById('w'),{{type:'line',data:{{labels,datasets:[{{label:'Weight',data:w,borderColor:'#1f77b4',borderWidth:2,pointRadius:0}},{{label:'Regime Scale',data:rs,borderColor:'#ff7f0e',borderWidth:2,pointRadius:0}}]}},options:{{...base,scales:{{...base.scales,y:{{min:0,max:1}}}}}}}});
new Chart(document.getElementById('s'),{{type:'line',data:{{labels,datasets:[{{label:'RiskOff',data:rf,borderColor:'#d62728',borderWidth:2,pointRadius:0}},{{label:'BullOn',data:bu,borderColor:'#9467bd',borderWidth:2,pointRadius:0}}]}},options:{{...base,scales:{{...base.scales,y:{{min:0,max:1}}}}}}}});
</script></body></html>"""
    path.write_text(html, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep HMM scaler + bull override over ENet base exposure")
    ap.add_argument("--price-csv", default="data/ETHUSDC_5m.csv")
    ap.add_argument("--exposure-csv", default="artifacts/paper/enet_base_exposure.csv")
    ap.add_argument("--regime-csv", default="data/regimes/8h/macro/ETHUSDC/regimes_8h.csv")
    ap.add_argument("--regime-col", default="state")
    ap.add_argument("--riskon-values", default="1")
    ap.add_argument("--riskoff-values", default="0")
    ap.add_argument("--neutral-scale-list", default="0.5,0.6,0.7")
    ap.add_argument("--riskoff-scale-list", default="0.2,0.3,0.4")
    ap.add_argument("--bull-ema-list", default="150,200,250")
    ap.add_argument("--bull-slope-ema-list", default="30,50")
    ap.add_argument("--bull-min-weight-list", default="0.25,0.35,0.45")
    ap.add_argument("--bull-timeframe", choices=["same", "1h", "4h", "1d"], default="1h")
    ap.add_argument("--trade-cost-bps", type=float, default=5.0)
    ap.add_argument("--bar-minutes", type=int, default=5)
    ap.add_argument("--out-csv", default="artifacts/paper/hmm_enet_scaled_sweep.csv")
    ap.add_argument("--out-top-html", default="artifacts/paper/hmm_enet_scaled_sweep_top5.html")
    ap.add_argument("--out-top-dir", default="artifacts/paper/hmm_enet_scaled_top5")
    args = ap.parse_args()

    riskon_str, riskon_num = _parse_values(args.riskon_values)
    riskoff_str, riskoff_num = _parse_values(args.riskoff_values)
    neutral_scales = _parse_float_list(args.neutral_scale_list)
    riskoff_scales = _parse_float_list(args.riskoff_scale_list)
    bull_emas = _parse_int_list(args.bull_ema_list)
    bull_slope_emas = _parse_int_list(args.bull_slope_ema_list)
    bull_floors = _parse_float_list(args.bull_min_weight_list)

    price = _load_price(Path(args.price_csv))
    exp = _load_exposure(Path(args.exposure_csv))
    reg = _load_regime(Path(args.regime_csv), args.regime_col)

    base = price.merge(exp, on="timestamp", how="inner").sort_values("timestamp")
    base = pd.merge_asof(base, reg, on="timestamp", direction="backward").dropna(subset=["regime"]).reset_index(drop=True)
    base["r"] = np.log(base["close"] / base["close"].shift(1)).fillna(0.0)
    base["spot_r"] = base["r"]

    riskon_mask = _match_values(base["regime"], riskon_str, riskon_num)
    riskoff_mask = _match_values(base["regime"], riskoff_str, riskoff_num)

    rows: list[dict] = []
    curves: dict[tuple, pd.DataFrame] = {}
    k = float(args.trade_cost_bps) / 10000.0

    for ns in neutral_scales:
        for rs in riskoff_scales:
            for be in bull_emas:
                for bse in bull_slope_emas:
                    for bf in bull_floors:
                        regime_scale = pd.Series(float(ns), index=base.index)
                        regime_scale = regime_scale.mask(riskon_mask, 1.0)
                        regime_scale = regime_scale.mask(riskoff_mask, float(rs))
                        w = pd.to_numeric(base["weight"], errors="coerce").fillna(0.0) * regime_scale

                        bull = _bull_override_signal(
                            ts=base["timestamp"],
                            close=base["close"],
                            bar_minutes=args.bar_minutes,
                            timeframe=args.bull_timeframe,
                            ema=int(be),
                            slope_ema=int(bse),
                        )
                        w = np.where(bull.to_numpy(), np.maximum(w.to_numpy(), float(bf)), w.to_numpy())
                        w = pd.Series(w, index=base.index).clip(lower=0.0, upper=1.0)

                        to = w.diff().abs().fillna(0.0)
                        strat_r = w.shift(1).fillna(0.0) * base["r"] + (-to * k)

                        sim = pd.DataFrame(
                            {
                                "timestamp": base["timestamp"].to_numpy(),
                                "close": pd.to_numeric(base["close"], errors="coerce").to_numpy(),
                                "regime": base["regime"].to_numpy(),
                                "riskoff": riskoff_mask.astype(int).to_numpy(),
                                "regime_scale": pd.to_numeric(regime_scale, errors="coerce").to_numpy(),
                                "bull_on": pd.Series(bull).astype(int).to_numpy(),
                                "weight": pd.to_numeric(w, errors="coerce").to_numpy(),
                                "strat_r": pd.to_numeric(strat_r, errors="coerce").to_numpy(),
                                "spot_r": pd.to_numeric(base["spot_r"], errors="coerce").to_numpy(),
                            }
                        )
                        sim = sim.sort_values("timestamp").reset_index(drop=True)
                        end = sim["timestamp"].iloc[-1]
                        sub90 = sim[sim["timestamp"] >= end - pd.Timedelta(days=90)].copy()
                        sub365 = sim[sim["timestamp"] >= end - pd.Timedelta(days=365)].copy()
                        m90 = _perf(sub90["strat_r"], args.bar_minutes)
                        s90 = _perf(sub90["spot_r"], args.bar_minutes)
                        m365 = _perf(sub365["strat_r"], args.bar_minutes)
                        s365 = _perf(sub365["spot_r"], args.bar_minutes)
                        score = (
                            1.0 * m90["ret"]
                            + 0.6 * m365["ret"]
                            + 0.2 * m90["sharpe"]
                            + 0.2 * m365["sharpe"]
                            + 0.2 * m90["mdd"]
                            + 0.2 * m365["mdd"]
                        )
                        rec = {
                            "neutral_scale": ns,
                            "riskoff_scale": rs,
                            "bull_ema": int(be),
                            "bull_slope_ema": int(bse),
                            "bull_min_weight": bf,
                            "ret_90d": m90["ret"],
                            "sharpe_90d": m90["sharpe"],
                            "mdd_90d": m90["mdd"],
                            "spot_ret_90d": s90["ret"],
                            "excess_90d": m90["ret"] - s90["ret"],
                            "ret_365d": m365["ret"],
                            "sharpe_365d": m365["sharpe"],
                            "mdd_365d": m365["mdd"],
                            "spot_ret_365d": s365["ret"],
                            "excess_365d": m365["ret"] - s365["ret"],
                            "tim_90d": float((pd.to_numeric(sub90["weight"], errors="coerce") > 0).mean() * 100.0),
                            "tim_365d": float((pd.to_numeric(sub365["weight"], errors="coerce") > 0).mean() * 100.0),
                            "score": score,
                        }
                        key = (ns, rs, int(be), int(bse), bf)
                        rows.append(rec)
                        curves[key] = sim

    res = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    res["rank"] = np.arange(1, len(res) + 1)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_csv, index=False)

    top = res.head(5).copy()
    _write_top_html(
        Path(args.out_top_html),
        top,
        meta=[
            f"configs={len(res)} | riskon=1.0 | cost={args.trade_cost_bps}bps",
            f"regime_file={args.regime_csv} ({args.regime_col}) | bull_tf={args.bull_timeframe}",
        ],
    )

    top_dir = Path(args.out_top_dir)
    top_dir.mkdir(parents=True, exist_ok=True)
    for _, r in top.iterrows():
        key = (
            float(r["neutral_scale"]),
            float(r["riskoff_scale"]),
            int(r["bull_ema"]),
            int(r["bull_slope_ema"]),
            float(r["bull_min_weight"]),
        )
        sim = curves[key]
        out_html = top_dir / f"rank_{int(r['rank'])}.html"
        title = f"HMM+ENet Scaled Rank {int(r['rank'])}"
        subtitle = (
            f"neutral={r['neutral_scale']:.2f}, riskoff={r['riskoff_scale']:.2f}, "
            f"bull_ema={int(r['bull_ema'])}, bull_slope={int(r['bull_slope_ema'])}, floor={r['bull_min_weight']:.2f} | "
            f"ret90={r['ret_90d']:.2%}, ret365={r['ret_365d']:.2%}"
        )
        _write_curve_html(out_html, sim, title, subtitle)

    print(f"wrote {out_csv}")
    print(f"wrote {args.out_top_html}")
    print(f"wrote top5 curve htmls in {top_dir}")
    print(top[["rank", "neutral_scale", "riskoff_scale", "bull_ema", "bull_slope_ema", "bull_min_weight", "ret_90d", "ret_365d", "excess_90d", "excess_365d", "score"]].to_string(index=False))


if __name__ == "__main__":
    main()
