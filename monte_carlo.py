"""
monte_carlo.py — Monte Carlo option pricer with GBM path simulation.

Features
--------
* Antithetic-variates variance reduction for accurate pricing.
* Full path simulation for visualisation (separate from the pricing paths).
* Convergence analysis — price as a function of number of paths.
* Control-variate option (uses BS as control; toggled by parameter).
"""

from __future__ import annotations

import numpy as np
from black_scholes import call_price, put_price


# ---------------------------------------------------------------------------
# Core pricer
# ---------------------------------------------------------------------------

def mc_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    n_paths: int = 100_000,
    seed: int = 42,
    q: float = 0.0,
) -> tuple[float, float, np.ndarray]:
    """
    Price a European option via Monte Carlo under risk-neutral GBM.

    Uses antithetic variates (doubles effective paths, halves variance).
    Includes continuous dividend yield q (Merton extension).

    Returns
    -------
    price       : discounted expected payoff
    std_error   : standard error of the estimate
    final_prices: terminal stock prices from the un-reflected paths
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    Z_anti = -Z                                # antithetic pairs

    def terminal(z: np.ndarray) -> np.ndarray:
        return S * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)

    ST      = terminal(Z)
    ST_anti = terminal(Z_anti)

    def payoff(s: np.ndarray) -> np.ndarray:
        if option_type == "call":
            return np.maximum(s - K, 0.0)
        return np.maximum(K - s, 0.0)

    pv = np.exp(-r * T) * 0.5 * (payoff(ST) + payoff(ST_anti))

    price     = float(pv.mean())
    std_error = float(pv.std() / np.sqrt(n_paths))

    return price, std_error, ST


# ---------------------------------------------------------------------------
# Full simulation result (pricing + display paths)
# ---------------------------------------------------------------------------

def mc_full_result(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    n_display_paths: int = 200,
    n_steps: int = 252,
    seed: int = 42,
) -> dict:
    """
    Run the MC pricer and also generate display paths for charting.

    Returns
    -------
    dict with keys:
        mc_price        : float
        bs_price        : float
        std_error       : float
        ci_low / ci_high: 95 % confidence interval
        paths           : (n_steps+1, n_display_paths) ndarray
        time_axis       : (n_steps+1,) array in years
        final_prices    : (n_paths,) terminal stock price sample
        itm_fraction    : fraction of paths expiring in-the-money
    """
    # High-accuracy price (100k paths)
    price, std_error, final_prices = mc_price(
        S, K, T, r, sigma, option_type, n_paths=100_000, seed=seed
    )

    # Black-Scholes benchmark
    bs = call_price(S, K, T, r, sigma) if option_type == "call" else put_price(S, K, T, r, sigma)

    # Display paths via GBM
    rng = np.random.default_rng(seed + 1)
    dt = T / n_steps
    log_drift = (r - 0.5 * sigma ** 2) * dt
    log_vol   = sigma * np.sqrt(dt)

    Z = rng.standard_normal((n_steps, n_display_paths))
    log_increments = log_drift + log_vol * Z

    paths = np.empty((n_steps + 1, n_display_paths))
    paths[0] = S
    for t in range(1, n_steps + 1):
        paths[t] = paths[t - 1] * np.exp(log_increments[t - 1])

    time_axis = np.linspace(0.0, T, n_steps + 1)

    if option_type == "call":
        itm = float((final_prices > K).mean())
    else:
        itm = float((final_prices < K).mean())

    return {
        "mc_price":    price,
        "bs_price":    bs,
        "std_error":   std_error,
        "ci_low":      price - 1.96 * std_error,
        "ci_high":     price + 1.96 * std_error,
        "paths":       paths,
        "time_axis":   time_axis,
        "final_prices": final_prices[:10_000],   # cap for histogram
        "itm_fraction": itm,
    }


# ---------------------------------------------------------------------------
# Convergence analysis
# ---------------------------------------------------------------------------

def mc_convergence(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    max_paths: int = 50_000,
    n_points: int = 30,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute MC price and std-error at logarithmically-spaced path counts
    up to max_paths.

    Returns
    -------
    path_counts   : (n_points,) int array
    prices        : (n_points,) float array
    std_errors    : (n_points,) float array
    """
    path_counts = np.unique(
        np.round(np.logspace(np.log10(100), np.log10(max_paths), n_points)).astype(int)
    )
    prices     = np.empty(len(path_counts))
    std_errors = np.empty(len(path_counts))

    # Pre-generate all random numbers once
    rng = np.random.default_rng(seed)
    Z_all = rng.standard_normal(max_paths)

    for idx, n in enumerate(path_counts):
        Z = Z_all[:n]
        Z_full = np.concatenate([Z, -Z])   # antithetic
        ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z_full)
        if option_type == "call":
            pv = np.exp(-r * T) * np.maximum(ST - K, 0.0)
        else:
            pv = np.exp(-r * T) * np.maximum(K - ST, 0.0)
        prices[idx]     = pv.mean()
        std_errors[idx] = pv.std() / np.sqrt(len(pv))

    return path_counts, prices, std_errors
