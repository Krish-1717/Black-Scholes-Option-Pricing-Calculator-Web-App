import numpy as np
from scipy.stats import norm
from black_scholes import d1, d2


def delta(S, K, T, r, sigma, option_type='call'):
    D1 = d1(S, K, T, r, sigma)
    if option_type == 'call':
        return norm.cdf(D1)
    return norm.cdf(D1) - 1


def gamma(S, K, T, r, sigma):
    D1 = d1(S, K, T, r, sigma)
    return norm.pdf(D1) / (S * sigma * np.sqrt(T))


def theta(S, K, T, r, sigma, option_type='call'):
    D1 = d1(S, K, T, r, sigma)
    D2 = d2(S, K, T, r, sigma)
    term1 = -(S * norm.pdf(D1) * sigma) / (2 * np.sqrt(T))
    if option_type == 'call':
        return (term1 - r * K * np.exp(-r * T) * norm.cdf(D2)) / 365
    return (term1 + r * K * np.exp(-r * T) * norm.cdf(-D2)) / 365


def vega(S, K, T, r, sigma):
    D1 = d1(S, K, T, r, sigma)
    return S * norm.pdf(D1) * np.sqrt(T) / 100


def rho(S, K, T, r, sigma, option_type='call'):
    D2 = d2(S, K, T, r, sigma)
    if option_type == 'call':
        return K * T * np.exp(-r * T) * norm.cdf(D2) / 100
    return -K * T * np.exp(-r * T) * norm.cdf(-D2) / 100
