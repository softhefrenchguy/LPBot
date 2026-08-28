import numpy as np

def generate_price_path(
    start_price,
    mu_annual,
    sigma_annual,
    n_steps,
    dt_hours=1,
    rng=None
):
    """
    Geometric Brownian Motion (continuous compounding)
    
    mu_annual: expected return (0 for neutral)
    sigma_annual: annualized vol (ETH ~ 80% = 0.80)
    """

    if rng is None:
        rng = np.random.default_rng()

    prices = np.zeros(n_steps)
    prices[0] = start_price

    dt_year = dt_hours / (24 * 365)
    mu_dt = mu_annual * dt_year
    sigma_dt = sigma_annual * np.sqrt(dt_year)

    for t in range(1, n_steps):
        z = rng.standard_normal()
        # log-normal GBM update
        prices[t] = prices[t-1] * np.exp((mu_dt - 0.5*sigma_dt**2) + sigma_dt * z)

    return prices
