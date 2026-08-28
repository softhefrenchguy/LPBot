import numpy as np

from ai_core.montecarlo.price_models import generate_price_path
from ai_core.montecarlo.volume_models import load_hourly_volumes, sample_volume_path
from ai_core.montecarlo.simulator import run_strategy_on_path

def main():
    start_price = 2000.0
    sigma = 0.8
    mu = 0.0

    widths = [60, 80, 100, 120]
    offsets = [30, 40, 50]

    n_paths = 100
    n_steps = 24 * 30  # 30 days of hourly steps

    rng = np.random.default_rng(42)

    vols_hist = load_hourly_volumes("eth_usdc_005_hourly_volume.csv")

    for w in widths:
        for o in offsets:
            pnl_list = []
            dd_list = []
            fee_list = []

            for _ in range(n_paths):
                prices = generate_price_path(start_price, mu, sigma, n_steps=n_steps)
                volume_path = sample_volume_path(n_steps, vols_hist, rng)

                final_eth, fees_eth, max_dd = run_strategy_on_path(
                    prices,
                    volume_path=volume_path,
                    width_ticks=w,
                    offset_ticks=o,
                    fee_tier=0.0005,
                    pool_share=0.00003,
                )
                pnl_list.append(final_eth)
                dd_list.append(max_dd)
                fee_list.append(fees_eth)

            pnl_arr = np.array(pnl_list)
            dd_arr = np.array(dd_list)
            fee_arr = np.array(fee_list)

            print(
                f"Width={w:3} Offset={o:3} → "
                f"Avg={pnl_arr.mean():.4f} ETH  | Worst={pnl_arr.min():.4f} | Std={pnl_arr.std():.4f} "
                f"| AvgDD={dd_arr.mean():.3f} | AvgFees={fee_arr.mean():.4f} ETH"
            )

if __name__ == "__main__":
    main()

