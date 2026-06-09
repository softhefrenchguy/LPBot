from __future__ import annotations

import argparse
import os
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _load_surface(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing vol surface CSV: {path}")
    df = pd.read_csv(path)
    required = {"moneyness", "days_to_expiry", "mark_iv", "strike", "expiry_date", "underlying_price"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{path} missing required columns: {sorted(missing)}")
    for col in ["moneyness", "days_to_expiry", "mark_iv", "strike", "underlying_price", "open_interest"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["expiry_date"] = pd.to_datetime(df["expiry_date"], utc=True, errors="coerce")
    df = df.dropna(subset=["moneyness", "days_to_expiry", "mark_iv"])
    df = df[(df["mark_iv"] > 0) & (df["days_to_expiry"] > 0)].copy()
    if df.empty:
        raise SystemExit("No usable option rows after cleaning.")
    return df.sort_values(["days_to_expiry", "moneyness"]).reset_index(drop=True)


def _atm_by_expiry(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["atm_dist"] = (x["moneyness"] - 1.0).abs()
    sort_cols = ["days_to_expiry", "atm_dist"]
    ascending = [True, True]
    if "open_interest" in x.columns:
        sort_cols.append("open_interest")
        ascending.append(False)
    atm = x.sort_values(sort_cols, ascending=ascending).drop_duplicates("days_to_expiry", keep="first")
    return atm.sort_values("days_to_expiry")


def _nearest_expiry_slice(df: pd.DataFrame, target_days: float) -> pd.DataFrame:
    days = sorted(df["days_to_expiry"].dropna().unique())
    if not days:
        return pd.DataFrame()
    nearest = min(days, key=lambda d: abs(float(d) - float(target_days)))
    return df[df["days_to_expiry"] == nearest].sort_values("moneyness").copy()


def _surface_plot(df: pd.DataFrame) -> go.Figure:
    # Grid by expiry/moneyness. Plotly can handle sparse surfaces with NaNs.
    pivot = df.pivot_table(index="days_to_expiry", columns="moneyness", values="mark_iv", aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1)
    fig = go.Figure(
        data=[
            go.Surface(
                x=pivot.columns.to_numpy(dtype=float),
                y=pivot.index.to_numpy(dtype=float),
                z=pivot.to_numpy(dtype=float),
                colorscale="RdBu_r",
                colorbar={"title": "IV %"},
                hovertemplate="Moneyness: %{x:.3f}<br>DTE: %{y:.0f}<br>IV: %{z:.1f}%<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="ETH Options Implied Volatility Surface",
        scene={
            "xaxis_title": "Moneyness (strike / spot)",
            "yaxis_title": "Days to expiry",
            "zaxis_title": "Implied vol (%)",
        },
        template="plotly_dark",
        height=760,
    )
    return fig


def _term_plot(df: pd.DataFrame) -> go.Figure:
    atm = _atm_by_expiry(df)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=atm["days_to_expiry"],
            y=atm["mark_iv"],
            mode="lines+markers",
            name="ATM IV",
            hovertemplate="DTE: %{x:.0f}<br>ATM IV: %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title="ETH ATM Implied Volatility Term Structure",
        xaxis_title="Days to expiry",
        yaxis_title="ATM implied vol (%)",
        template="plotly_dark",
        height=520,
    )
    return fig


def _skew_plot(df: pd.DataFrame, target_days: float) -> tuple[go.Figure, float]:
    sl = _nearest_expiry_slice(df, target_days)
    if sl.empty:
        raise SystemExit("No expiry slice available for skew plot.")
    expiry_days = float(sl["days_to_expiry"].iloc[0])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sl["moneyness"],
            y=sl["mark_iv"],
            mode="lines+markers",
            name=f"{expiry_days:.0f}d skew",
            text=sl["instrument_name"] if "instrument_name" in sl.columns else None,
            hovertemplate="%{text}<br>Moneyness: %{x:.3f}<br>IV: %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_vline(x=1.0, line_width=1, line_dash="dash", line_color="white")
    fig.update_layout(
        title=f"ETH Vol Skew / Smile Near {target_days:.0f}d Expiry (actual {expiry_days:.0f}d)",
        xaxis_title="Moneyness (strike / spot)",
        yaxis_title="Implied vol (%)",
        template="plotly_dark",
        height=520,
    )
    return fig, expiry_days


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate Plotly ETH options vol surface visualizations.")
    ap.add_argument("--input", default="artifacts/options/vol_surface_latest.csv")
    ap.add_argument("--out-dir", default="artifacts/options")
    ap.add_argument("--skew-days", type=float, default=30.0)
    ap.add_argument("--no-open", action="store_true", help="Do not open generated HTML files in browser.")
    args = ap.parse_args()

    df = _load_surface(Path(args.input))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    surface_path = out_dir / "vol_surface_3d.html"
    term_path = out_dir / "vol_surface_term.html"
    skew_path = out_dir / "vol_surface_skew.html"

    _surface_plot(df).write_html(surface_path, include_plotlyjs="cdn")
    _term_plot(df).write_html(term_path, include_plotlyjs="cdn")
    _, expiry_days = _skew_plot(df, args.skew_days)
    _skew_plot(df, args.skew_days)[0].write_html(skew_path, include_plotlyjs="cdn")

    atm = _atm_by_expiry(df)
    short_iv = float(atm.iloc[0]["mark_iv"]) if not atm.empty else np.nan
    long_iv = float(atm.iloc[-1]["mark_iv"]) if not atm.empty else np.nan
    slope = long_iv - short_iv if np.isfinite(short_iv) and np.isfinite(long_iv) else np.nan

    print("=" * 48)
    print("ETH OPTIONS VOL SURFACE VISUALS")
    print("=" * 48)
    print(f"Rows plotted: {len(df):,}")
    print(f"Expiries: {df['expiry_date'].nunique()}")
    print(f"Moneyness range: {df['moneyness'].min():.3f} - {df['moneyness'].max():.3f}")
    print(f"IV range: {df['mark_iv'].min():.1f}% - {df['mark_iv'].max():.1f}%")
    print(f"ATM term slope (last - first): {slope:.1f} vol pts" if np.isfinite(slope) else "ATM term slope: NA")
    print(f"Skew slice target: {args.skew_days:.0f}d | actual: {expiry_days:.0f}d")
    print("Saved:")
    print(f"  {surface_path}")
    print(f"  {term_path}")
    print(f"  {skew_path}")
    print("=" * 48)

    if not args.no_open:
        for path in [surface_path, term_path, skew_path]:
            webbrowser.open(path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
