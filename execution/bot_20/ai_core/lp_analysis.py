import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List


# ============================================================
# CONFIG OBJECT
# ============================================================

@dataclass
class LPAnalysisConfig:
    trend_up_threshold: float = 0.05
    trend_down_threshold: float = -0.05
    vola_high_threshold: float = 0.02
    vola_low_threshold: float = 0.005


# ============================================================
# SAFE TIMESTAMP EXTRACTION
# ============================================================

def extract_timestamp(row, keys):
    """
    Try multiple field names until one exists.
    """
    for k in keys:
        if k in row and pd.notna(row[k]) and row[k] != "":
            return pd.to_datetime(row[k], utc=True)
    return None


# ============================================================
# TREND REGIME
# ============================================================

def infer_trend_regime(features: pd.DataFrame, cfg: LPAnalysisConfig):
    """
    Uses EMA gap + EMA slope to classify price trend regime.
    """
    gap = features["ema_gap_pct"]
    slope = features["ema5_slope"]

    trend = []

    for g, s in zip(gap, slope):

        if g > cfg.trend_up_threshold and s > 0:
            trend.append("strong_uptrend")

        elif g > 0 and s > 0:
            trend.append("mild_uptrend")

        elif g < cfg.trend_down_threshold and s < 0:
            trend.append("strong_downtrend")

        elif g < 0 and s < 0:
            trend.append("mild_downtrend")

        else:
            trend.append("sideways")

    return trend


# ============================================================
# VOLATILITY REGIME
# ============================================================

def infer_vola_regime(features: pd.DataFrame, cfg: LPAnalysisConfig):
    vola = features["vol_ewma_60"]

    regime = []
    for v in vola:
        if v > cfg.vola_high_threshold:
            regime.append("high_volatility")
        elif v < cfg.vola_low_threshold:
            regime.append("low_volatility")
        else:
            regime.append("normal_volatility")

    return regime


# ============================================================
# MAP CYCLES TO REGIMES
# ============================================================

def map_cycle_to_regimes(cycles: pd.DataFrame, features: pd.DataFrame, cfg: LPAnalysisConfig):
    """
    Takes your cycles and attaches the market regime active at exit time.
    """

    out = []

    for idx, row in cycles.iterrows():

        # ----------------------------------------------------
        # Extract timestamps safely
        # ----------------------------------------------------
        t_exit = extract_timestamp(
            row,
            [
                "timestamp_withdraw",
                "timestamp_exit",
                "timestamp_close",
                "exit_timestamp",
                "timestamp",
            ],
        )

        t_entry = extract_timestamp(
            row,
            [
                "timestamp_create",
                "timestamp_entry",
                "timestamp_open",
                "entry_timestamp",
                "timestamp",
            ],
        )

        if t_exit is None:
            continue

        # ----------------------------------------------------
        # Find the feature row closest BEFORE exit time
        # ----------------------------------------------------
        f = features[features["timestamp"] <= t_exit]

        if len(f) == 0:
            continue

        f = f.iloc[-1]  # latest feature before exit

        out.append(
            {
                "cycle_number": row.get("cycle_number", None),
                "timestamp_exit": t_exit,
                "exit_reason": row.get("exit_reason", None),
                "duration": row.get("duration_minutes", None),
                "wallet_pnl": row.get("wallet_pnl_after_fees", row.get("real_pnl_after_fees")),
                "trend_regime": f["trend_regime"],
                "vola_regime": f["vola_regime"],
                "ema_gap_pct": f["ema_gap_pct"],
                "ema_slope": f["ema5_slope"],
                "volatility": f["vol_ewma_60"],
            }
        )

    return pd.DataFrame(out)


# ============================================================
# LOAD CYCLE HISTORY
# ============================================================

def load_cycle_history(path):
    return pd.read_json(path, lines=True)


# ============================================================
# MAIN ANALYSIS ENTRYPOINT
# ============================================================

def run_cycle_regime_analysis(
    cycle_history_path: str,
    features_csv_path: str,
    output_path: str,
    cfg: LPAnalysisConfig,
):
    print("=== Running LP Cycle Regime Analysis ===")

    # Load data
    cycles = load_cycle_history(cycle_history_path)
    features = pd.read_csv(features_csv_path)

    # Ensure timestamp format
    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True)

    # Compute regimes
    features["trend_regime"] = infer_trend_regime(features, cfg)
    features["vola_regime"] = infer_vola_regime(features, cfg)

    mapped = map_cycle_to_regimes(cycles, features, cfg)

    mapped.to_csv(output_path, index=False)
    print(f"[OK] Saved cycle regime analysis → {output_path}")
    print("Preview:")
    print(mapped.tail(10))

