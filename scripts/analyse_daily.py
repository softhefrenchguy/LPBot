from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv


DEFAULT_MODEL = "claude-sonnet-4-6"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _log(path: Path, message: str) -> None:
    _ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{_utc_now()}] {message}\n")


def _safe_num(value: Any, default: float = np.nan) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    return str(value)


def _first(row: pd.Series, names: list[str], default: Any = np.nan) -> Any:
    for name in names:
        if name in row.index:
            value = row.get(name)
            try:
                if pd.isna(value):
                    continue
            except Exception:
                pass
            return value
    return default


def _fmt_num(value: Any, decimals: int = 1, default: str = "n/a") -> str:
    x = _safe_num(value)
    if not np.isfinite(x):
        return default
    return f"{x:.{decimals}f}"


def _fmt_price(value: Any, default: str = "n/a") -> str:
    x = _safe_num(value)
    if not np.isfinite(x):
        return default
    return f"${x:,.0f}"


def _to_percent(value: Any) -> float:
    x = _safe_num(value)
    if not np.isfinite(x):
        return np.nan
    return x * 100.0 if abs(x) <= 2.0 else x


def _fmt_pct(value: Any, decimals: int = 1, signed: bool = False, default: str = "n/a") -> str:
    x = _to_percent(value)
    if not np.isfinite(x):
        return default
    sign = "+" if signed else ""
    return f"{x:{sign}.{decimals}f}%"


def _fmt_bool(value: Any) -> str:
    raw = _safe_str(value).strip().lower()
    if raw in {"true", "1", "yes", "y"}:
        return "YES"
    if raw in {"false", "0", "no", "n"}:
        return "NO"
    return _safe_str(value, "n/a")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _latest_row(path: Path) -> dict[str, Any]:
    df = _read_csv(path)
    if df.empty:
        return {}
    if "date" in df.columns:
        df = df.sort_values("date")
    elif "timestamp_utc" in df.columns:
        df = df.sort_values("timestamp_utc")
    return df.tail(1).iloc[0].to_dict()


def _load_history(path: Path, lookback_days: int) -> pd.DataFrame:
    df = _read_csv(path)
    if df.empty:
        raise FileNotFoundError(f"Missing or empty daily checks log: {path}")
    if "date" not in df.columns:
        raise ValueError(f"daily checks log has no date column: {path}")
    df = df.dropna(subset=["date"]).sort_values("date")
    return df.tail(lookback_days).reset_index(drop=True)


def _history_line(row: pd.Series) -> str:
    date = _safe_str(_first(row, ["date"]), "unknown-date")
    eth_price = _first(row, ["eth_price"])
    eth_24h = _first(row, ["eth_24h_pct"])
    regime = _safe_str(_first(row, ["eth_regime", "regime"]), "n/a")
    dd_20d = _first(row, ["dd_20d"])
    ema50 = _first(row, ["ema50", "eth_ema50", "ema21"])
    btc_ema15 = _first(row, ["btc_ema15", "btc_ema21"])
    vol_regime = _safe_str(_first(row, ["vol_regime"]), "n/a")
    excess = _first(row, ["portfolio_excess_vs_basket", "excess"])
    news_direction = _safe_str(_first(row, ["news_direction"]), "n/a")
    return (
        f"{date} | ETH {_fmt_price(eth_price)} ({_fmt_pct(eth_24h, signed=True)}) | "
        f"Regime: {regime} | DD/20d: {_fmt_pct(dd_20d)} | "
        f"EMA50: {_fmt_num(ema50, 0)} | BTC EMA15: {_fmt_num(btc_ema15, 0)} | "
        f"Vol: {vol_regime} | Excess: {_fmt_pct(excess, 2, signed=True)} | "
        f"News: {news_direction}"
    )


def _build_history_block(df_history: pd.DataFrame) -> str:
    return "\n".join(_history_line(row) for _, row in df_history.iterrows())


def _build_today_block(today: pd.Series, news_latest: dict[str, Any], dvol_latest: dict[str, Any], trades: pd.DataFrame) -> str:
    news_summary = _safe_str(_first(today, ["news_summary"]), "")
    if not news_summary and news_latest:
        news_summary = _safe_str(news_latest.get("summary"), "")

    dvol_atm = _first(today, ["dvol_atm_iv_30d"], dvol_latest.get("atm_iv_30d", np.nan))
    dvol_pctile = _first(today, ["dvol_iv_percentile", "dvol_iv_percentile_pct"], dvol_latest.get("iv_percentile", np.nan))
    dvol_regime = _safe_str(_first(today, ["dvol_options_vol_regime"], dvol_latest.get("options_vol_regime", "n/a")), "n/a")
    dvol_rv_regime = _safe_str(_first(today, ["dvol_rv_vol_regime"], dvol_latest.get("rv_vol_regime", "n/a")), "n/a")
    dvol_agreement = _safe_str(_first(today, ["dvol_agreement"], dvol_latest.get("agreement", "n/a")), "n/a")
    dvol_days = int(_safe_num(_first(today, ["dvol_history_days"], dvol_latest.get("history_days", 0)), 0.0))
    dvol_disagree = (
        dvol_regime.strip().upper() not in {"", "N/A", "NA", "NAN"}
        and dvol_rv_regime.strip().upper() not in {"", "N/A", "NA", "NAN"}
        and dvol_regime.strip().upper() != dvol_rv_regime.strip().upper()
    )
    eth_price = _safe_num(_first(today, ["eth_price"]))
    eth_ema50 = _safe_num(_first(today, ["ema50", "eth_ema50", "ema21"]))
    eth_ema120 = _safe_num(_first(today, ["ema120", "eth_ema120", "ema55"]))
    btc_ema15 = _safe_num(_first(today, ["btc_ema15", "btc_ema21"]))
    btc_ema40 = _safe_num(_first(today, ["btc_ema40", "btc_ema55"]))
    eth_ema50_gap = eth_ema120 - eth_ema50 if np.isfinite(eth_ema120) and np.isfinite(eth_ema50) else np.nan
    eth_ema50_gap_pct = eth_ema50_gap / eth_price * 100.0 if np.isfinite(eth_ema50_gap) and np.isfinite(eth_price) and eth_price else np.nan
    btc_ema15_gap = btc_ema40 - btc_ema15 if np.isfinite(btc_ema40) and np.isfinite(btc_ema15) else np.nan

    trade_count = len(trades) if trades is not None else 0
    last_trade = "None"
    if trade_count:
        tr = trades.tail(1).iloc[0]
        last_trade = (
            f"{_safe_str(tr.get('entry_date'))} -> {_safe_str(tr.get('exit_date'))}, "
            f"{_fmt_pct(tr.get('return_pct'), 2, signed=True)}, "
            f"reason={_safe_str(tr.get('exit_reason'), 'n/a')}"
        )

    lines = [
        f"ETH: {_fmt_price(_first(today, ['eth_price']))} ({_fmt_pct(_first(today, ['eth_24h_pct']), signed=True)})",
        f"Regime: {_safe_str(_first(today, ['eth_regime', 'regime']), 'n/a')} (streak {_fmt_num(_first(today, ['eth_bear_streak', 'regime_days']), 0)}d)",
        f"DD/20d: {_fmt_pct(_first(today, ['dd_20d']))}",
        "EMA50/120/300: "
        f"{_fmt_num(_first(today, ['ema50', 'eth_ema50', 'ema21']), 0)} / "
        f"{_fmt_num(_first(today, ['ema120', 'eth_ema120', 'ema55']), 0)} / "
        f"{_fmt_num(_first(today, ['ema300', 'eth_ema300', 'ema144']), 0)}",
        f"Stack aligned: {_fmt_bool(_first(today, ['eth_stack_aligned', 'stack_aligned']))}",
        f"Days since break: {_fmt_num(_first(today, ['days_since_break', 'off_days_since_break']), 0)}",
        f"ETH EMA50->120 gap: ${_fmt_num(eth_ema50_gap, 0)} ({_fmt_num(eth_ema50_gap_pct, 1)}% from alignment)",
        "",
        f"BTC: {_fmt_price(_first(today, ['btc_price']))}",
        f"BTC Regime: {_safe_str(_first(today, ['btc_regime']), 'n/a')}",
        "BTC EMA15/40/120: "
        f"{_fmt_num(_first(today, ['btc_ema15', 'btc_ema21']), 0)} / "
        f"{_fmt_num(_first(today, ['btc_ema40', 'btc_ema55']), 0)} / "
        f"{_fmt_num(_first(today, ['btc_ema120', 'btc_ema144']), 0)}",
        f"BTC EMA15->40 gap: ${_fmt_num(btc_ema15_gap, 0)}",
        f"Funding z: {_fmt_num(_first(today, ['btc_funding_z']), 3)}",
        "",
        f"Vol regime: {_safe_str(_first(today, ['vol_regime']), 'n/a')} ({_fmt_num(_first(today, ['vol_percentile']), 0)}th pct)",
        f"Conviction: {_safe_str(_first(today, ['eth_conviction', 'conviction_bucket']), 'n/a')}",
        f"P(up): {_fmt_num(_first(today, ['p_up', 'def_p_up']), 3)}",
        f"Mean-rev z: {_fmt_num(_first(today, ['mr_zscore', 'mean_reversion_z_score']), 2)}",
        "",
        f"Options IV (30d): {_fmt_num(dvol_atm, 1)}%",
        f"IV percentile: {_fmt_num(dvol_pctile, 0)}th",
        f"IV regime: {dvol_regime}",
        f"RV regime: {dvol_rv_regime}",
        f"IV/RV agreement: {dvol_agreement}",
        (
            f"IV DISAGREE: Options={dvol_regime}, RV={dvol_rv_regime}; "
            "historical avg next 5d: +2.64%"
            if dvol_disagree
            else "IV DISAGREE: NO"
        ),
        f"Signal active: {'YES' if dvol_days >= 30 else 'NO'}",
        "",
        f"Strategy: {_fmt_pct(_first(today, ['portfolio_strategy_ret', 'cum_strategy_ret']), 2, signed=True)}",
        f"ETH spot: {_fmt_pct(_first(today, ['eth_sleeve_spot_ret', 'cum_spot_ret']), 2, signed=True)}",
        f"Excess vs basket: {_fmt_pct(_first(today, ['portfolio_excess_vs_basket', 'excess']), 2, signed=True)}",
        f"Peak DD: {_fmt_pct(_first(today, ['portfolio_peak_dd', 'peak_dd']), 2, signed=True)}",
        f"Days live: {_fmt_num(_first(today, ['days_live']), 0)}",
        f"Completed trades: {trade_count}",
        f"Last trade: {last_trade}",
        "",
        f"News: {news_summary[:200] if news_summary else 'None'}",
    ]
    return "\n".join(lines)


def _system_prompt() -> str:
    return """You are LPBot's daily analyst for a systematic ETH/BTC trend-following strategy.
You receive 14 days of rolling history plus today's detail.

Strategy context:
- ETH EMA 50/120/300, confirm 3 days
- BTC EMA 15/40/120, confirm 5 days
- Signal-weighted, gross cap 0.8
- Mean-reversion overlay in CHOP regime
- Vol filter: HIGH=0.5x, LOW=1.2x
- PAXG gold reserve in BEAR regime
- Validated Sharpe 1.762, CAGR 39.78%
- Paper trading since 2026-03-21
- One completed trade: -10.21% (Apr-May 2026)
- Currently waiting for next entry signal

Your analysis must:
- Identify trends visible across the rolling history, not just today
- Note what is changing vs prior days
- Flag any signals approaching thresholds
- State clearly what to watch next
- Be concise: max 250 words
- No fluff, no repeating raw numbers
- Write as if texting a busy trader who checks Discord once a day

Always end your analysis with exactly this format:

**Next entry trigger:**
ETH: EMA50 (${ema50:.0f}) needs to cross EMA120 (${ema120:.0f}) —
gap ${gap:.0f} ({gap_pct:.1f}%).
Estimated {weeks} weeks at current trajectory.

BTC: EMA15 (${btc_ema15:.0f}) needs to cross EMA40 (${btc_ema40:.0f}) —
gap ${btc_gap:.0f}.

Never omit this section.
Make it specific with actual numbers."""


def _call_claude(api_key: str, model: str, history_block: str, today_block: str, limited_history: bool) -> tuple[str, dict[str, Any]]:
    import anthropic

    limited = "\n\nLimited history available: fewer than 3 rows loaded." if limited_history else ""
    user_prompt = f"""14-DAY ROLLING HISTORY (oldest first):
{history_block}

TODAY IN DETAIL:
{today_block}
{limited}

Analyse the trends and what matters."""

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=500,
        system=_system_prompt(),
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = response.content[0].text if response.content else ""
    usage = {
        "input_tokens": getattr(response.usage, "input_tokens", None),
        "output_tokens": getattr(response.usage, "output_tokens", None),
    }
    return text.strip(), usage


def _post_discord(webhook_url: str, date: str, days_live: int, analysis: str, timeout: float) -> tuple[bool, str]:
    content = f"**LPBot Analysis - {date} (Day {days_live})**\n\n{analysis}"
    if len(content) > 1900:
        content = content[:1875].rstrip() + "\n\n[truncated]"
    response = requests.post(
        webhook_url,
        data=json.dumps({"content": content}),
        headers={"Content-Type": "application/json", "User-Agent": "LPBot/analysis"},
        timeout=timeout,
    )
    if response.status_code in {200, 204}:
        return True, f"posted status={response.status_code}"
    return False, f"failed status={response.status_code} body={response.text[:200]}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Claude daily LPBot analysis and post it to Discord.")
    parser.add_argument("--daily-log", default="artifacts/paper_trade/daily_checks_log.csv")
    parser.add_argument("--news-log", default="artifacts/news/news_log.csv")
    parser.add_argument("--dvol-log", default="artifacts/options/dvol_log.csv")
    parser.add_argument("--completed-trades", default="artifacts/paper_trade/completed_trades.csv")
    parser.add_argument("--analysis-dir", default="artifacts/analysis")
    parser.add_argument("--log-file", default="logs/analysis.log")
    parser.add_argument("--lookback-days", type=int, default=14)
    parser.add_argument("--model", default=os.environ.get("CLAUDE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--discord-timeout-sec", type=float, default=10.0)
    parser.add_argument("--no-discord", action="store_true", help="Generate and archive analysis without posting to Discord.")
    args = parser.parse_args()

    load_dotenv()
    log_path = Path(args.log_file)
    analysis_dir = Path(args.analysis_dir)
    _ensure_dir(analysis_dir)

    try:
        df_history = _load_history(Path(args.daily_log), int(args.lookback_days))
    except Exception as exc:
        _log(log_path, f"ERROR loading daily history: {exc}")
        return 0

    if len(df_history) < 3:
        _log(log_path, f"WARNING limited history rows={len(df_history)}")

    today = df_history.iloc[-1]
    today_date = _safe_str(_first(today, ["date"]), datetime.now().strftime("%Y-%m-%d"))
    days_live = int(_safe_num(_first(today, ["days_live"]), 0.0))
    news_latest = _latest_row(Path(args.news_log))
    dvol_latest = _latest_row(Path(args.dvol_log))
    trades = _read_csv(Path(args.completed_trades))

    history_block = _build_history_block(df_history)
    today_block = _build_today_block(today, news_latest, dvol_latest, trades)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        _log(log_path, "ERROR ANTHROPIC_API_KEY not set; skipping Claude analysis")
        return 0

    try:
        analysis, usage = _call_claude(api_key, str(args.model), history_block, today_block, len(df_history) < 3)
    except Exception:
        _log(log_path, "ERROR Claude API failed:\n" + traceback.format_exc())
        return 0

    archive_path = analysis_dir / f"analysis_{today_date}.txt"
    archive_path.write_text(analysis + "\n", encoding="utf-8")

    webhook_url = os.environ.get("DISCORD_ANALYSIS_WEBHOOK_URL", "").strip()
    discord_status = "skipped"
    if args.no_discord:
        discord_status = "skipped --no-discord"
    elif not webhook_url:
        discord_status = "skipped DISCORD_ANALYSIS_WEBHOOK_URL not set"
    else:
        try:
            ok, msg = _post_discord(webhook_url, today_date, days_live, analysis, float(args.discord_timeout_sec))
            discord_status = msg
            if not ok:
                _log(log_path, f"ERROR Discord post failed: {msg}")
        except Exception:
            discord_status = "failed exception"
            _log(log_path, "ERROR Discord post exception:\n" + traceback.format_exc())

    _log(
        log_path,
        "analysis_complete "
        f"date={today_date} history_rows={len(df_history)} model={args.model} "
        f"tokens={usage} discord={discord_status}\n{analysis}",
    )
    print(f"Loaded {len(df_history)} history rows")
    print(f"Saved {archive_path}")
    print(f"Discord: {discord_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
