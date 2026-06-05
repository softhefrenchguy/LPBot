import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _pick_guess(df: pd.DataFrame, guess_col: str | None, weight_threshold: float) -> tuple[pd.Series, str]:
    if guess_col:
        if guess_col not in df.columns:
            raise ValueError(f"guess column not found: {guess_col}")
        s = pd.to_numeric(df[guess_col], errors="coerce")
        if s.notna().any():
            return (s > 0.5).fillna(False), f"{guess_col}>0.5"
        txt = df[guess_col].astype(str).str.lower()
        return txt.isin({"1", "true", "bull", "riskon", "on"}).fillna(False), guess_col

    if "bull_on" in df.columns:
        s = pd.to_numeric(df["bull_on"], errors="coerce").fillna(0.0)
        return (s > 0.5), "bull_on"
    if "riskoff" in df.columns:
        s = pd.to_numeric(df["riskoff"], errors="coerce").fillna(0.0)
        return ~(s > 0.5), "not riskoff"
    if "weight" in df.columns:
        s = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
        return s > float(weight_threshold), f"weight>{weight_threshold}"

    raise ValueError("could not infer guess signal (need --guess-col or one of bull_on/riskoff/weight)")


def _safe_ratio(a: float, b: float) -> float:
    return float(a / b) if abs(b) > 1e-12 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate bull/bear guesses against future ETH direction.")
    ap.add_argument("--input-csv", default="artifacts/paper/hmm_enet_scaled_v3.csv")
    ap.add_argument("--timestamp-col", default="timestamp")
    ap.add_argument("--price-col", default="close")
    ap.add_argument("--guess-col", default=None)
    ap.add_argument("--weight-threshold", type=float, default=0.0)
    ap.add_argument("--horizon-bars", type=int, default=1, help="Future bars for outcome (1=next bar).")
    ap.add_argument("--rolling-window", type=int, default=200)
    ap.add_argument("--out-events-csv", default="artifacts/paper/bull_bear_events.csv")
    ap.add_argument("--out-summary-csv", default="artifacts/paper/bull_bear_summary.csv")
    ap.add_argument("--out-html", default="artifacts/paper/bull_bear_report.html")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.timestamp_col not in df.columns or args.price_col not in df.columns:
        raise ValueError(f"input must contain {args.timestamp_col} and {args.price_col}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df[args.timestamp_col], utc=True, errors="coerce")
    df["close"] = pd.to_numeric(df[args.price_col], errors="coerce")
    df = df.dropna(subset=["timestamp", "close"]).sort_values("timestamp")

    guess_bull, guess_src = _pick_guess(df, args.guess_col, args.weight_threshold)
    df["guess_bull"] = guess_bull.to_numpy(dtype=bool)
    df["guess"] = np.where(df["guess_bull"], "bull", "bear")

    h = max(1, int(args.horizon_bars))
    df["fwd_close"] = df["close"].shift(-h)
    df["fwd_ret"] = (df["fwd_close"] / df["close"]) - 1.0
    df = df.dropna(subset=["fwd_ret"]).copy()
    df["actual_up"] = df["fwd_ret"] > 0
    df["actual"] = np.where(df["actual_up"], "up", "down_or_flat")
    df["correct"] = (df["guess_bull"] == df["actual_up"])

    tp = int(((df["guess_bull"]) & (df["actual_up"])).sum())
    fp = int(((df["guess_bull"]) & (~df["actual_up"])).sum())
    tn = int(((~df["guess_bull"]) & (~df["actual_up"])).sum())
    fn = int(((~df["guess_bull"]) & (df["actual_up"])).sum())
    n = len(df)

    acc = _safe_ratio(tp + tn, n)
    prec = _safe_ratio(tp, tp + fp)
    rec = _safe_ratio(tp, tp + fn)
    spec = _safe_ratio(tn, tn + fp)
    bull_rate = float(df["guess_bull"].mean())
    up_rate = float(df["actual_up"].mean())

    df["rolling_acc"] = df["correct"].rolling(max(5, int(args.rolling_window))).mean()
    df["cum_acc"] = np.cumsum(df["correct"].to_numpy(dtype=float)) / np.arange(1, n + 1)

    summary = pd.DataFrame(
        [
            {
                "rows": n,
                "guess_source": guess_src,
                "horizon_bars": h,
                "accuracy": acc,
                "precision_bull": prec,
                "recall_bull": rec,
                "specificity_bear": spec,
                "bull_guess_rate": bull_rate,
                "actual_up_rate": up_rate,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        ]
    )

    out_events = Path(args.out_events_csv)
    out_events.parent.mkdir(parents=True, exist_ok=True)
    df[
        [
            "timestamp",
            "close",
            "guess",
            "guess_bull",
            "fwd_close",
            "fwd_ret",
            "actual",
            "actual_up",
            "correct",
            "rolling_acc",
            "cum_acc",
        ]
    ].to_csv(out_events, index=False)

    out_summary = Path(args.out_summary_csv)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, index=False)

    # Charts
    ts = df["timestamp"]
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=["Price + Guess", "Forward Return (Outcome)", "Rolling / Cumulative Accuracy"],
    )
    fig.add_trace(go.Scatter(x=ts, y=df["close"], name="Close", line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=ts,
            y=df["guess_bull"].astype(int),
            name="Guess Bull(1)/Bear(0)",
            line=dict(color="#ff7f0e"),
            yaxis="y2",
            opacity=0.5,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=ts, y=df["fwd_ret"], name="Fwd Ret", line=dict(color="#2ca02c")), row=2, col=1)
    fig.add_trace(go.Scatter(x=ts, y=df["rolling_acc"], name="Rolling Acc", line=dict(color="#d62728")), row=3, col=1)
    fig.add_trace(go.Scatter(x=ts, y=df["cum_acc"], name="Cum Acc", line=dict(color="#9467bd")), row=3, col=1)

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Fwd Ret", row=2, col=1)
    fig.update_yaxes(title_text="Accuracy", row=3, col=1, range=[0, 1])
    fig.update_layout(height=1100, title="Bull/Bear Guess Evaluation")

    table_html = summary.round(6).to_html(index=False, border=0)
    plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    html = (
        "<html><head><meta charset='utf-8'><title>Bull Bear Guess Report</title></head><body>"
        "<h3>Bull/Bear Guess Evaluation</h3>"
        f"<p>input={args.input_csv} | horizon_bars={h} | guess={guess_src}</p>"
        "<h4>Summary</h4>"
        f"{table_html}"
        f"{plot_html}"
        "</body></html>"
    )
    out_html = Path(args.out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")

    print("wrote", out_events)
    print("wrote", out_summary)
    print("wrote", out_html)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
