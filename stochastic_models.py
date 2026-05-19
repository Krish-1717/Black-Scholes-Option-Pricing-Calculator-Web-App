"""
stochastic_models.py — Advanced stochastic process models for option pricing.

Models
------
* Heston (1993)         — Stochastic volatility (mean-reverting CIR vol process)
* Merton Jump-Diffusion — GBM + Poisson jump component
* GBM                  — Standard Black-Scholes process (baseline)

Exotic payoffs (all priceable by any model)
-------------------------------------------
* European call / put
* Asian call / put     (arithmetic average price)
* Barrier knock-in/out (call / put)
* Digital (cash-or-nothing) call / put
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Literal


# ─────────────────────────────────────────────────────────────────────────────
# Parameter containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GBMParams:
    """Standard Black-Scholes / GBM parameters."""
    S: float          # Current stock price
    K: float          # Strike price
    T: float          # Time to expiry (years)
    r: float          # Risk-free rate
    sigma: float      # Volatility
    q: float = 0.0    # Continuous dividend yield


@dataclass
class HestonParams:
    """
    Heston (1993) stochastic volatility model parameters.

    dS = (r-q)S dt + √v S dW₁
    dv = κ(θ-v) dt + ξ√v dW₂
    corr(dW₁, dW₂) = ρ

    Feller condition for v to stay positive: 2κθ > ξ²
    """
    S: float          # Current stock price
    K: float          # Strike price
    T: float          # Time to expiry (years)
    r: float          # Risk-free rate
    q: float          # Dividend yield
    v0: float         # Initial variance (σ₀²)
    kappa: float      # Mean-reversion speed κ
    theta: float      # Long-run variance θ
    xi: float         # Vol of vol ξ
    rho: float        # Correlation ρ ∈ (-1, 1)


@dataclass
class JumpDiffusionParams:
    """
    Merton (1976) jump-diffusion parameters.

    dS/S = (r - q - λμ_J) dt + σ dW + (J-1) dN
    J ~ log-normal: ln J ~ N(μ_J, σ_J²)
    N ~ Poisson(λ)
    """
    S: float          # Current stock price
    K: float          # Strike price
    T: float          # Time to expiry (years)
    r: float          # Risk-free rate
    sigma: float      # Diffusion volatility
    q: float          # Dividend yield
    lam: float        # Jump intensity λ (expected jumps/year)
    mu_j: float       # Mean log jump size
    sigma_j: float    # Std dev of log jump size


# ─────────────────────────────────────────────────────────────────────────────
# Path generators
# ─────────────────────────────────────────────────────────────────────────────

def _gbm_paths(
    params: GBMParams,
    n_paths: int,
    n_steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate GBM paths with antithetic variates.

    Returns
    -------
    paths    : (n_steps+1, n_paths) array of stock prices
    time_axis: (n_steps+1,) array
    """
    rng = np.random.default_rng(seed)
    dt = params.T / n_steps
    drift = (params.r - params.q - 0.5 * params.sigma ** 2) * dt
    vol   = params.sigma * np.sqrt(dt)

    half = n_paths // 2
    Z = rng.standard_normal((n_steps, half))
    Z_full = np.concatenate([Z, -Z], axis=1)   # antithetic

    log_inc = drift + vol * Z_full
    paths = np.empty((n_steps + 1, n_paths))
    paths[0] = params.S
    for t in range(1, n_steps + 1):
        paths[t] = paths[t - 1] * np.exp(log_inc[t - 1])

    return paths, np.linspace(0.0, params.T, n_steps + 1)


def _heston_paths(
    params: HestonParams,
    n_paths: int,
    n_steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Euler-Maruyama discretisation of the Heston model with antithetic variates.
    Uses the full-truncation scheme to keep variance positive.
    """
    rng = np.random.default_rng(seed)
    dt = params.T / n_steps
    half = n_paths // 2

    # Correlated Brownians
    Z1 = rng.standard_normal((n_steps, half))
    Z2 = params.rho * Z1 + np.sqrt(1 - params.rho ** 2) * rng.standard_normal((n_steps, half))

    # Antithetic
    Z1_full = np.concatenate([Z1, -Z1], axis=1)
    Z2_full = np.concatenate([Z2, -Z2], axis=1)

    paths = np.empty((n_steps + 1, n_paths))
    vols  = np.empty((n_steps + 1, n_paths))
    paths[0] = params.S
    vols[0]  = params.v0

    for t in range(n_steps):
        v_pos = np.maximum(vols[t], 0.0)  # full-truncation
        sqrt_v = np.sqrt(v_pos)

        # Stock process
        paths[t + 1] = paths[t] * np.exp(
            (params.r - params.q - 0.5 * v_pos) * dt
            + sqrt_v * np.sqrt(dt) * Z1_full[t]
        )

        # Variance process (CIR)
        vols[t + 1] = (
            vols[t]
            + params.kappa * (params.theta - v_pos) * dt
            + params.xi * sqrt_v * np.sqrt(dt) * Z2_full[t]
        )

    return paths, np.linspace(0.0, params.T, n_steps + 1)


def _jump_diffusion_paths(
    params: JumpDiffusionParams,
    n_paths: int,
    n_steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Merton jump-diffusion paths via Euler scheme with antithetic diffusion variates.
    """
    rng = np.random.default_rng(seed)
    dt = params.T / n_steps

    # Expected jump contribution to drift correction
    k_bar = np.exp(params.mu_j + 0.5 * params.sigma_j ** 2) - 1.0
    drift = (params.r - params.q - 0.5 * params.sigma ** 2 - params.lam * k_bar) * dt
    vol   = params.sigma * np.sqrt(dt)

    half = n_paths // 2
    Z = rng.standard_normal((n_steps, half))
    Z_full = np.concatenate([Z, -Z], axis=1)

    paths = np.empty((n_steps + 1, n_paths))
    paths[0] = params.S

    for t in range(n_steps):
        # Diffusion component
        log_diff = drift + vol * Z_full[t]

        # Jump component — Poisson number of jumps per step
        n_jumps = rng.poisson(params.lam * dt, size=n_paths)
        log_jump = np.zeros(n_paths)
        for path_i in range(n_paths):
            if n_jumps[path_i] > 0:
                jumps = rng.normal(params.mu_j, params.sigma_j, n_jumps[path_i])
                log_jump[path_i] = jumps.sum()

        paths[t + 1] = paths[t] * np.exp(log_diff + log_jump)

    return paths, np.linspace(0.0, params.T, n_steps + 1)


# ─────────────────────────────────────────────────────────────────────────────
# Payoff functions
# ─────────────────────────────────────────────────────────────────────────────

def _european_payoff(paths: np.ndarray, K: float, otype: str) -> np.ndarray:
    ST = paths[-1]
    if otype == "call":
        return np.maximum(ST - K, 0.0)
    return np.maximum(K - ST, 0.0)


def _asian_payoff(paths: np.ndarray, K: float, otype: str) -> np.ndarray:
    """Arithmetic average price Asian option."""
    avg = paths.mean(axis=0)
    if otype == "call":
        return np.maximum(avg - K, 0.0)
    return np.maximum(K - avg, 0.0)


def _barrier_payoff(
    paths: np.ndarray,
    K: float,
    barrier: float,
    otype: str,
    barrier_type: Literal["knock-in", "knock-out"],
    barrier_direction: Literal["up", "down"],
) -> np.ndarray:
    """
    Barrier option payoff.

    barrier_type      : knock-in (activated when barrier hit) or knock-out (extinguished)
    barrier_direction : up (barrier above current price) or down (barrier below)
    """
    ST = paths[-1]
    base_payoff = np.maximum(ST - K, 0.0) if otype == "call" else np.maximum(K - ST, 0.0)

    if barrier_direction == "up":
        hit = paths.max(axis=0) >= barrier
    else:
        hit = paths.min(axis=0) <= barrier

    if barrier_type == "knock-in":
        return base_payoff * hit
    else:  # knock-out
        return base_payoff * (~hit)


def _digital_payoff(paths: np.ndarray, K: float, otype: str, payout: float = 1.0) -> np.ndarray:
    """Cash-or-nothing digital option."""
    ST = paths[-1]
    if otype == "call":
        return payout * (ST > K).astype(float)
    return payout * (ST < K).astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# Unified simulation engine
# ─────────────────────────────────────────────────────────────────────────────

def simulate(
    model: Literal["gbm", "heston", "jump"],
    model_params,
    option_style: Literal["european", "asian", "barrier", "digital"],
    option_type: Literal["call", "put"],
    n_paths: int = 100_000,
    n_steps: int = 252,
    seed: int = 42,
    barrier: float | None = None,
    barrier_type: Literal["knock-in", "knock-out"] = "knock-out",
    barrier_direction: Literal["up", "down"] = "up",
    digital_payout: float = 1.0,
    n_display_paths: int = 200,
) -> dict:
    """
    Unified Monte Carlo engine.

    Returns
    -------
    dict with keys:
        price          : float   discounted expected payoff
        std_error      : float
        ci_low/ci_high : float   95% confidence interval
        itm_fraction   : float
        paths          : ndarray (n_steps+1, n_display_paths)  display sample
        time_axis      : ndarray
        final_prices   : ndarray terminal prices
        pct5/25/75/95  : ndarray percentile bands across time
        mean_path      : ndarray
        model          : str
        option_style   : str
    """
    # Ensure n_paths is even — antithetic variates require paired paths
    n_paths = n_paths + (n_paths % 2)

    # ── Generate paths ────────────────────────────────────────────────────────
    if model == "gbm":
        all_paths, t_ax = _gbm_paths(model_params, n_paths, n_steps, seed)
    elif model == "heston":
        all_paths, t_ax = _heston_paths(model_params, n_paths, n_steps, seed)
    elif model == "jump":
        all_paths, t_ax = _jump_diffusion_paths(model_params, n_paths, n_steps, seed)
    else:
        raise ValueError(f"Unknown model: {model}")

    K = model_params.K
    r = model_params.r
    T = model_params.T

    # ── Compute payoffs ───────────────────────────────────────────────────────
    if option_style == "european":
        payoffs = _european_payoff(all_paths, K, option_type)
    elif option_style == "asian":
        payoffs = _asian_payoff(all_paths, K, option_type)
    elif option_style == "barrier":
        if barrier is None:
            raise ValueError("barrier must be provided for barrier options")
        payoffs = _barrier_payoff(all_paths, K, barrier, option_type,
                                  barrier_type, barrier_direction)
    elif option_style == "digital":
        payoffs = _digital_payoff(all_paths, K, option_type, digital_payout)
    else:
        raise ValueError(f"Unknown option_style: {option_style}")

    pv = np.exp(-r * T) * payoffs
    price     = float(pv.mean())
    std_error = float(pv.std() / np.sqrt(n_paths))

    # ── Percentile bands ──────────────────────────────────────────────────────
    pct5  = np.percentile(all_paths, 5,  axis=1)
    pct25 = np.percentile(all_paths, 25, axis=1)
    pct75 = np.percentile(all_paths, 75, axis=1)
    pct95 = np.percentile(all_paths, 95, axis=1)
    mean_path = all_paths.mean(axis=1)

    # Display sample (cap for browser performance)
    idx_sample = np.random.default_rng(seed + 99).choice(
        n_paths, size=min(n_display_paths, n_paths), replace=False
    )
    display_paths = all_paths[:, idx_sample]

    # ITM fraction
    ST = all_paths[-1]
    if option_type == "call":
        itm = float((ST > K).mean())
    else:
        itm = float((ST < K).mean())

    return {
        "price":       price,
        "std_error":   std_error,
        "ci_low":      price - 1.96 * std_error,
        "ci_high":     price + 1.96 * std_error,
        "itm_fraction": itm,
        "paths":       display_paths,
        "time_axis":   t_ax,
        "final_prices": ST[:10_000],
        "pct5":        pct5,
        "pct25":       pct25,
        "pct75":       pct75,
        "pct95":       pct95,
        "mean_path":   mean_path,
        "model":       model,
        "option_style": option_style,
        "n_paths":     n_paths,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Heston default parameter estimator from historical vol
# ─────────────────────────────────────────────────────────────────────────────

def heston_defaults_from_hv(S, K, T, r, sigma, q=0.0) -> HestonParams:
    """
    Sensible Heston defaults calibrated from historical vol.
    Uses empirically common equity parameters.
    """
    v0 = sigma ** 2
    return HestonParams(
        S=S, K=K, T=T, r=r, q=q,
        v0=v0,
        kappa=2.0,               # moderate mean-reversion
        theta=v0,                # long-run vol same as current
        xi=0.3,                  # moderate vol-of-vol
        rho=-0.70,               # typical equity leverage effect
    )


def jump_defaults_from_hv(S, K, T, r, sigma, q=0.0) -> JumpDiffusionParams:
    """
    Sensible Merton jump-diffusion defaults from historical vol.
    """
    return JumpDiffusionParams(
        S=S, K=K, T=T, r=r, q=q,
        sigma=sigma * 0.80,     # diffusion vol (reduced; jumps carry rest)
        lam=5.0,                # ~5 jumps per year
        mu_j=-0.02,             # average log-jump size (small downward)
        sigma_j=0.06,           # dispersion of jump sizes
    )
