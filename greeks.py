"""
greeks.py — First-order option Greeks with continuous dividend yield support.

All functions accept an optional q parameter (continuous dividend yield, default 0.0).
"""

import numpy as np
from scipy.stats import norm
from black_scholes import d1, d2


def delta(S, K, T, r, sigma, option_type='call', q=0.0):
    """
    Delta — rate of change of option price with respect to stock price.

    Call delta: e^(-qT) * N(d1)
    Put  delta: e^(-qT) * (N(d1) - 1)
    """
    if T <= 0:
        if option_type == 'call':
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    D1 = d1(S, K, T, r, sigma, q)
    disc = np.exp(-q * T)
    if option_type == 'call':
        return disc * norm.cdf(D1)
    return disc * (norm.cdf(D1) - 1)


def gamma(S, K, T, r, sigma, q=0.0):
    """
    Gamma — rate of change of delta with respect to stock price.

    Same for calls and puts.  e^(-qT) * N'(d1) / (S * σ * √T)
    """
    if T <= 0:
        return 0.0
    D1 = d1(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(D1) / (S * sigma * np.sqrt(T))


def theta(S, K, T, r, sigma, option_type='call', q=0.0):
    """
    Theta — option price decay per calendar day.

    Includes dividend yield term.
    """
    if T <= 0:
        return 0.0
    D1 = d1(S, K, T, r, sigma, q)
    D2 = d2(S, K, T, r, sigma, q)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)

    common = -(S * disc_q * norm.pdf(D1) * sigma) / (2 * np.sqrt(T))

    if option_type == 'call':
        return (
            common
            - r * K * disc_r * norm.cdf(D2)
            + q * S * disc_q * norm.cdf(D1)
        ) / 365
    else:
        return (
            common
            + r * K * disc_r * norm.cdf(-D2)
            - q * S * disc_q * norm.cdf(-D1)
        ) / 365


def vega(S, K, T, r, sigma, q=0.0):
    """
    Vega — option price change per 1% increase in volatility.

    S * e^(-qT) * N'(d1) * √T / 100
    """
    if T <= 0:
        return 0.0
    D1 = d1(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(D1) * np.sqrt(T) / 100


def rho(S, K, T, r, sigma, option_type='call', q=0.0):
    """
    Rho — option price change per 1% increase in risk-free rate.

    K * T * e^(-rT) * N(d2) / 100  (call)
    -K * T * e^(-rT) * N(-d2) / 100 (put)
    """
    if T <= 0:
        return 0.0
    D2 = d2(S, K, T, r, sigma, q)
    disc_r = np.exp(-r * T)
    if option_type == 'call':
        return K * T * disc_r * norm.cdf(D2) / 100
    return -K * T * disc_r * norm.cdf(-D2) / 100
