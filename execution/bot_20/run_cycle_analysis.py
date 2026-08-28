from ai_core.lp_analysis import (
    run_cycle_regime_analysis,
    LPAnalysisConfig
)

# Paths used by the rewritten analysis engine
CYCLE_HISTORY = "cycle_history.jsonl"
FEATURES_CSV = "data/eth_usdc_features_1m.csv"
OUTPUT = "analysis/cycle_regimes.csv"

def main():
    print("=== Running LP Cycle Regime Analysis ===")

    cfg = LPAnalysisConfig(
        trend_up_threshold=0.05,
        trend_down_threshold=-0.05,
        vola_high_threshold=0.02,
        vola_low_threshold=0.005,
    )

    run_cycle_regime_analysis(
        cycle_history_path=CYCLE_HISTORY,
        features_csv_path=FEATURES_CSV,
        output_path=OUTPUT,
        cfg=cfg,
    )

if __name__ == "__main__":
    main()
