from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class SeriesPack:
    name: str
    equity: pd.Series
    color: str


REGIME_COLORS = {
    "BULL": "rgba(34, 197, 94, 0.12)",
    "CHOP": "rgba(234, 179, 8, 0.12)",
    "BEAR": "rgba(239, 68, 68, 0.12)",
}

LINE_COLORS = {
    "Base": "#1f77b4",
    "ETH Spot": "#6b7280",
    "PAXG strict cond": "#f59e0b",
    "IAUM strict cond": "#10b981",
}


def _load_base_daily(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False)
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce")
    d = d.dropna(subset=["timestamp"]).sort_values("timestamp").copy()
    d["date"] = d["timestamp"].dt.floor("D")
    g = d.groupby("date", as_index=False).last()
    out = g[["date", "eq", "spot_eq"]].copy()
    out["eq"] = pd.to_numeric(out["eq"], errors="coerce")
    out["spot_eq"] = pd.to_numeric(out["spot_eq"], errors="coerce")
    out = out.dropna(subset=["eq", "spot_eq"]).sort_values("date").copy()
    return out


def _load_strict_daily(path: Path, model: str) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False)
    d["date"] = pd.to_datetime(d["date"], utc=True, errors="coerce").dt.floor("D")
    d = d.dropna(subset=["date"]).sort_values(["model", "date"]).copy()
    m = d[d["model"] == model].copy()
    if m.empty:
        raise ValueError(f"Model not found in strict file: {model}")
    out = m[["date", "base_day_ret", "strategy_day_ret"]].copy()
    out["base_day_ret"] = pd.to_numeric(out["base_day_ret"], errors="coerce").fillna(0.0)
    out["strategy_day_ret"] = pd.to_numeric(out["strategy_day_ret"], errors="coerce").fillna(0.0)
    out = out.sort_values("date").copy()
    out["base_eq"] = (1.0 + out["base_day_ret"]).cumprod()
    out["strategy_eq"] = (1.0 + out["strategy_day_ret"]).cumprod()
    return out


def _load_strict_base(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False)
    d["date"] = pd.to_datetime(d["date"], utc=True, errors="coerce").dt.floor("D")
    d = d.dropna(subset=["date"]).sort_values(["date", "model"]).copy()
    out = d[["date", "base_day_ret"]].drop_duplicates(subset=["date"], keep="last").sort_values("date").copy()
    out["base_day_ret"] = pd.to_numeric(out["base_day_ret"], errors="coerce").fillna(0.0)
    out["Base"] = (1.0 + out["base_day_ret"]).cumprod()
    return out


def _normalize_at_start(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        s = pd.to_numeric(out[c], errors="coerce")
        first_idx = s.first_valid_index()
        if first_idx is None:
            continue
        base = float(s.loc[first_idx])
        if np.isfinite(base) and base != 0:
            out[c] = s / base
        else:
            out[c] = s
    return out


def _series_stats(date_s: pd.Series, eq_s: pd.Series) -> dict[str, float]:
    e = pd.to_numeric(eq_s, errors="coerce").dropna()
    if len(e) < 2:
        return {"cagr": np.nan, "sharpe": np.nan, "maxdd": np.nan}
    r = e.pct_change().fillna(0.0)
    years = max((date_s.iloc[-1] - date_s.iloc[0]).days / 365.25, 1e-9)
    cagr = float(e.iloc[-1] ** (1.0 / years) - 1.0)
    vol = float(r.std(ddof=0) * np.sqrt(365.0))
    sharpe = float((r.mean() * 365.0) / vol) if vol > 0 else np.nan
    dd = e / e.cummax() - 1.0
    maxdd = float(dd.min())
    return {"cagr": cagr, "sharpe": sharpe, "maxdd": maxdd}


def _maxdd_peak_trough(date_s: pd.Series, eq_s: pd.Series) -> tuple[pd.Timestamp, float, pd.Timestamp, float]:
    e = pd.to_numeric(eq_s, errors="coerce")
    dd = e / e.cummax() - 1.0
    trough_idx = int(dd.idxmin())
    peak_idx = int(e.loc[:trough_idx].idxmax())
    return (
        pd.Timestamp(date_s.loc[peak_idx]),
        float(e.loc[peak_idx]),
        pd.Timestamp(date_s.loc[trough_idx]),
        float(e.loc[trough_idx]),
    )


def _regime_spans(regime_csv: Path, x_min: pd.Timestamp, x_max: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    r = pd.read_csv(regime_csv, low_memory=False)
    r["date"] = pd.to_datetime(r["day"], utc=True, errors="coerce").dt.floor("D")
    r = r.dropna(subset=["date"]).sort_values("date").copy()
    r = r[(r["date"] >= x_min.floor("D")) & (r["date"] <= x_max.floor("D"))].copy()
    if r.empty:
        return []
    spans: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    cur_reg = str(r["regime_v2"].iloc[0])
    cur_start = pd.Timestamp(r["date"].iloc[0])
    prev_date = pd.Timestamp(r["date"].iloc[0])
    for i in range(1, len(r)):
        d = pd.Timestamp(r["date"].iloc[i])
        reg = str(r["regime_v2"].iloc[i])
        if reg != cur_reg:
            spans.append((cur_start, prev_date + pd.Timedelta(days=1), cur_reg))
            cur_reg = reg
            cur_start = d
        prev_date = d
    spans.append((cur_start, prev_date + pd.Timedelta(days=1), cur_reg))
    return spans


def _build_chart(
    out_html: Path,
    title: str,
    data: pd.DataFrame,
    top_series: list[SeriesPack],
    dd_series_names: list[str],
    regime_csv: Path,
) -> None:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.72, 0.28],
        subplot_titles=("Equity Curves", "Drawdown"),
    )

    x_min = pd.Timestamp(data["date"].min())
    x_max = pd.Timestamp(data["date"].max())
    for x0, x1, regime in _regime_spans(regime_csv, x_min, x_max):
        fig.add_shape(
            type="rect",
            xref="x",
            yref="paper",
            x0=x0,
            x1=x1,
            y0=0.0,
            y1=1.0,
            fillcolor=REGIME_COLORS.get(regime, "rgba(156, 163, 175, 0.08)"),
            line=dict(width=0),
            layer="below",
        )

    stats_lines: list[str] = []
    for s in top_series:
        y = pd.to_numeric(data[s.name], errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=y,
                mode="lines",
                name=s.name,
                line=dict(color=s.color, width=2.2),
            ),
            row=1,
            col=1,
        )

        p_dt, p_eq, t_dt, t_eq = _maxdd_peak_trough(data["date"], y)
        fig.add_trace(
            go.Scatter(
                x=[p_dt],
                y=[p_eq],
                mode="markers",
                name=f"{s.name} MaxDD Peak",
                marker=dict(color=s.color, size=9, symbol="triangle-up"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[t_dt],
                y=[t_eq],
                mode="markers",
                name=f"{s.name} MaxDD Trough",
                marker=dict(color=s.color, size=9, symbol="triangle-down"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        st = _series_stats(data["date"], y)
        stats_lines.append(
            f"<span style='color:{s.color}'><b>{s.name}</b></span>: "
            f"CAGR {st['cagr']*100:.2f}% | Sharpe {st['sharpe']:.2f} | MaxDD {st['maxdd']*100:.2f}%"
        )

    for name in dd_series_names:
        y = pd.to_numeric(data[name], errors="coerce")
        dd = y / y.cummax() - 1.0
        color = LINE_COLORS.get(name, "#111827")
        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=dd,
                mode="lines",
                name=f"{name} DD",
                line=dict(color=color, width=1.8, dash="dot"),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    fig.add_hline(y=0.0, line_width=1, line_dash="dash", line_color="#9ca3af", row=2, col=1)
    fig.add_annotation(
        x=0.995,
        y=0.995,
        xref="paper",
        yref="paper",
        xanchor="right",
        yanchor="top",
        align="right",
        text="<br>".join(stats_lines),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#d1d5db",
        borderwidth=1,
        font=dict(size=11),
        showarrow=False,
    )

    fig.update_layout(
        template="plotly_white",
        title=title,
        legend=dict(orientation="h", y=1.08, x=0.0),
        margin=dict(l=70, r=40, t=85, b=55),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Equity (Start=1.0)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")


def main() -> None:
    out_dir = Path("artifacts/backtest/charts")
    out_dir.mkdir(parents=True, exist_ok=True)

    gapfilled_base_path = Path("artifacts/backtest/combined_offtf_ema21_55_144_defv1_flat_routerA_6y_gapfilled_patch.csv")
    base_path = gapfilled_base_path if gapfilled_base_path.exists() else Path(
        "artifacts/backtest/combined_offtf_ema21_55_144_defv1_flat_routerA_6y.csv"
    )
    strict_path = Path("artifacts/backtest/strict_idle_gold_backtest_daily_2026-03-28_gapfilled.csv")
    regime_path = Path("artifacts/backtest/regime_classifier_v2_daily.csv")

    # ETH spot comes from the (patched) combined series; base equity comes from strict gapfilled base_day_ret.
    spot_daily = _load_base_daily(base_path).rename(columns={"eq": "CombinedEq", "spot_eq": "ETH Spot"})
    spot_daily = _normalize_at_start(spot_daily, ["ETH Spot"])
    strict_base = _load_strict_base(strict_path)
    strict_base["ETH Spot"] = strict_base["date"].map(spot_daily.set_index("date")["ETH Spot"])
    strict_base = strict_base.dropna(subset=["ETH Spot"]).sort_values("date").copy()
    strict_base = _normalize_at_start(strict_base, ["Base", "ETH Spot"])

    _build_chart(
        out_html=out_dir / "equity_curve_base.html",
        title="Combined Router A — Base Only",
        data=strict_base[["date", "Base", "ETH Spot"]].copy(),
        top_series=[
            SeriesPack("Base", strict_base["Base"], LINE_COLORS["Base"]),
            SeriesPack("ETH Spot", strict_base["ETH Spot"], LINE_COLORS["ETH Spot"]),
        ],
        dd_series_names=["Base", "ETH Spot"],
        regime_csv=regime_path,
    )

    paxg = _load_strict_daily(strict_path, "PAXG_strict_cond")
    iaum = _load_strict_daily(strict_path, "IAUM_strict_cond")

    # Build ETH spot series over strict date window from base daily spot_eq.
    spot_map = spot_daily.set_index("date")["ETH Spot"].sort_index()

    paxg_df = paxg[["date", "base_eq", "strategy_eq"]].copy()
    paxg_df = paxg_df.rename(columns={"base_eq": "Base", "strategy_eq": "PAXG strict cond"})
    paxg_df["ETH Spot"] = paxg_df["date"].map(spot_map)
    paxg_df = paxg_df.dropna(subset=["Base", "PAXG strict cond", "ETH Spot"]).sort_values("date")
    paxg_df = _normalize_at_start(paxg_df, ["Base", "PAXG strict cond", "ETH Spot"])

    _build_chart(
        out_html=out_dir / "equity_curve_paxg.html",
        title="Combined Router A + PAXG Rotation",
        data=paxg_df[["date", "PAXG strict cond", "Base", "ETH Spot"]].copy(),
        top_series=[
            SeriesPack("PAXG strict cond", paxg_df["PAXG strict cond"], LINE_COLORS["PAXG strict cond"]),
            SeriesPack("Base", paxg_df["Base"], LINE_COLORS["Base"]),
            SeriesPack("ETH Spot", paxg_df["ETH Spot"], LINE_COLORS["ETH Spot"]),
        ],
        dd_series_names=["PAXG strict cond", "Base", "ETH Spot"],
        regime_csv=regime_path,
    )

    iaum_df = iaum[["date", "base_eq", "strategy_eq"]].copy()
    iaum_df = iaum_df.rename(columns={"base_eq": "Base", "strategy_eq": "IAUM strict cond"})
    iaum_df["ETH Spot"] = iaum_df["date"].map(spot_map)
    iaum_df = iaum_df.dropna(subset=["Base", "IAUM strict cond", "ETH Spot"]).sort_values("date")
    iaum_df = _normalize_at_start(iaum_df, ["Base", "IAUM strict cond", "ETH Spot"])

    _build_chart(
        out_html=out_dir / "equity_curve_iaum.html",
        title="Combined Router A + IAUM Rotation",
        data=iaum_df[["date", "IAUM strict cond", "Base", "ETH Spot"]].copy(),
        top_series=[
            SeriesPack("IAUM strict cond", iaum_df["IAUM strict cond"], LINE_COLORS["IAUM strict cond"]),
            SeriesPack("Base", iaum_df["Base"], LINE_COLORS["Base"]),
            SeriesPack("ETH Spot", iaum_df["ETH Spot"], LINE_COLORS["ETH Spot"]),
        ],
        dd_series_names=["IAUM strict cond", "Base", "ETH Spot"],
        regime_csv=regime_path,
    )

    cmp = paxg[["date", "base_eq", "strategy_eq"]].rename(columns={"base_eq": "Base", "strategy_eq": "PAXG strict cond"})
    cmp = cmp.merge(
        iaum[["date", "strategy_eq"]].rename(columns={"strategy_eq": "IAUM strict cond"}),
        on="date",
        how="inner",
    )
    cmp["ETH Spot"] = cmp["date"].map(spot_map)
    cmp = cmp.dropna(subset=["Base", "PAXG strict cond", "IAUM strict cond", "ETH Spot"]).sort_values("date")
    cmp = _normalize_at_start(cmp, ["Base", "PAXG strict cond", "IAUM strict cond", "ETH Spot"])

    _build_chart(
        out_html=out_dir / "equity_curve_comparison.html",
        title="Strategy Comparison: Base vs PAXG vs IAUM",
        data=cmp[["date", "Base", "PAXG strict cond", "IAUM strict cond", "ETH Spot"]].copy(),
        top_series=[
            SeriesPack("Base", cmp["Base"], LINE_COLORS["Base"]),
            SeriesPack("PAXG strict cond", cmp["PAXG strict cond"], LINE_COLORS["PAXG strict cond"]),
            SeriesPack("IAUM strict cond", cmp["IAUM strict cond"], LINE_COLORS["IAUM strict cond"]),
            SeriesPack("ETH Spot", cmp["ETH Spot"], LINE_COLORS["ETH Spot"]),
        ],
        dd_series_names=["Base", "PAXG strict cond", "IAUM strict cond"],
        regime_csv=regime_path,
    )

    print(f"Wrote charts to {out_dir}")
    for n in [
        "equity_curve_base.html",
        "equity_curve_paxg.html",
        "equity_curve_iaum.html",
        "equity_curve_comparison.html",
    ]:
        p = out_dir / n
        print(f"{p} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
