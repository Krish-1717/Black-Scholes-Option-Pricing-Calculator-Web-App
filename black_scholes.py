"""
black_scholes.py — Black-Scholes / Merton option pricing formulas.

All functions accept an optional continuous dividend yield q (default 0.0),
implementing the Merton (1973) extension of the Black-Scholes model.
"""

import numpy as np
from scipy.stats import norm


# ── Core components ───────────────────────────────────────────────────────────

def d1(S, K, T, r, sigma, q=0.0):
    """d1 component including continuous dividend yield q."""
    if sigma <= 0 or T <= 0:
        return float("inf") if S >= K else float("-inf")
    return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def d2(S, K, T, r, sigma, q=0.0):
    """d2 = d1 - σ√T."""
    if sigma <= 0 or T <= 0:
        return float("inf") if S >= K else float("-inf")
    return d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)


# ── Option prices ─────────────────────────────────────────────────────────────

def call_price(S, K, T, r, sigma, q=0.0):
    """
    Black-Scholes call price with optional continuous dividend yield.

    Parameters
    ----------
    S     : float  Current stock price
    K     : float  Strike price
    T     : float  Time to expiry in years
    r     : float  Risk-free rate (decimal)
    sigma : float  Annualised volatility (decimal)
    q     : float  Continuous dividend yield (decimal), default 0.0

    Returns
    -------
    float  Call option price
    """
    if T <= 0:
        return max(S - K, 0.0)
    D1 = d1(S, K, T, r, sigma, q)
    D2 = d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.cdf(D1) - K * np.exp(-r * T) * norm.cdf(D2)


def put_price(S, K, T, r, sigma, q=0.0):
    """
    Black-Scholes put price with optional continuous dividend yield.

    Parameters
    ----------
    S     : float  Current stock price
    K     : float  Strike price
    T     : float  Time to expiry in years
    r     : float  Risk-free rate (decimal)
    sigma : float  Annualised volatility (decimal)
    q     : float  Continuous dividend yield (decimal), default 0.0

    Returns
    -------
    float  Put option price
    """
    if T <= 0:
        return max(K - S, 0.0)
    D1 = d1(S, K, T, r, sigma, q)
    D2 = d2(S, K, T, r, sigma, q)
    return K * np.exp(-r * T) * norm.cdf(-D2) - S * np.exp(-q * T) * norm.cdf(-D1)


# ── Convenience helpers ───────────────────────────────────────────────────────

def prob_itm_call(S, K, T, r, sigma, q=0.0):
    """Risk-neutral probability that a call expires in the money: N(d2)."""
    if T <= 0:
        return 1.0 if S > K else 0.0
    return norm.cdf(d2(S, K, T, r, sigma, q))


def prob_itm_put(S, K, T, r, sigma, q=0.0):
    """Risk-neutral probability that a put expires in the money: N(-d2)."""
    if T <= 0:
        return 1.0 if K > S else 0.0
    return norm.cdf(-d2(S, K, T, r, sigma, q))


def breakeven_call(K, premium):
    """Breakeven stock price at expiry for a long call."""
    return K + premium


def breakeven_put(K, premium):
    """Breakeven stock price at expiry for a long put."""
    return K - premium
