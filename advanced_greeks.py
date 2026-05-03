"""
advanced_greeks.py — Second- and third-order option Greeks.

All functions assume:
  S     = current stock price
  K     = strike price
  T     = time to expiry in years
  r     = continuously-compounded risk-free rate (decimal)
  sigma = annualised volatility (decimal)
  option_type = 'call' | 'put'

Conventions
-----------
* Charm / Color are expressed PER CALENDAR DAY (divided by 365).
* Vanna  = ∂²V / (∂S ∂σ)  — shared by calls and puts.
* Vomma  = ∂²V / ∂σ²       — shared by calls and puts.
* Speed  = ∂Γ  / ∂S        — shared by calls and puts.
* Color  = ∂Γ  / ∂t        — per calendar day (negative = Gamma decays).
* Ultima = ∂³V / ∂σ³       — third-order vol sensitivity.
"""

import numpy as np
from scipy.stats import norm
from black_scholes import d1, d2


# ---------------------------------------------------------------------------
# Charm  (Delta decay / DdeltaDtime)
# ---------------------------------------------------------------------------

def charm(S: float, K: float, T: float, r: float, sigma: float,
          option_type: str = "call") -> float:
    """
    Rate of change of delta per calendar day.

    Charm = -∂Δ/∂τ  (where τ = time to expiry), divided by 365.
    A positive value means delta is rising as time passes.

    Analytical formula (q = 0):
        d(N(d1))/dτ = N'(d1) · (2rT - d2·σ√T) / (2T·σ√T)
        Charm = -d(N(d1))/dτ  / 365
    """
    if T <= 1e-8:
        return 0.0

    D1 = d1(S, K, T, r, sigma)
    D2 = d2(S, K, T, r, sigma)

    # dDelta/dT (T = time to expiry)
    d_delta_d_tau = (
        norm.pdf(D1) * (2.0 * r * T - D2 * sigma * np.sqrt(T))
        / (2.0 * T * sigma * np.sqrt(T))
    )
    # Charm = -dDelta/dTau per calendar day
    # (same sign for calls and puts — put delta = N(d1)-1, derivative unchanged)
    return -d_delta_d_tau / 365.0


# ---------------------------------------------------------------------------
# Vanna  (DdeltaDvol / DvegaDspot)
# ---------------------------------------------------------------------------

def vanna(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Sensitivity of delta to a change in volatility.
    Equivalently, sensitivity of vega to a change in spot.

    Vanna = -N'(d1) · d2 / σ
    """
    if T <= 1e-8:
        return 0.0

    D1 = d1(S, K, T, r, sigma)
    D2 = d2(S, K, T, r, sigma)
    return -norm.pdf(D1) * D2 / sigma


# ---------------------------------------------------------------------------
# Vomma / Volga  (DvegaDvol)
# ---------------------------------------------------------------------------

def vomma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Rate of change of vega with respect to volatility.
    Also called Volga.

    Vomma = Vega_raw · d1·d2 / σ
    where Vega_raw = S · N'(d1) · √T   (unscaled vega)
    """
    if T <= 1e-8:
        return 0.0

    D1 = d1(S, K, T, r, sigma)
    D2 = d2(S, K, T, r, sigma)
    raw_vega = S * norm.pdf(D1) * np.sqrt(T)
    return raw_vega * D1 * D2 / sigma


# ---------------------------------------------------------------------------
# Speed  (DgammaDspot)
# ---------------------------------------------------------------------------

def speed(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Rate of change of gamma with respect to the stock price.
    Third partial derivative of the option price with respect to spot.

    Speed = -(Γ / S) · (1 + d1 / (σ√T))
    """
    if T <= 1e-8:
        return 0.0

    D1 = d1(S, K, T, r, sigma)
    gamma_val = norm.pdf(D1) / (S * sigma * np.sqrt(T))
    return -gamma_val / S * (1.0 + D1 / (sigma * np.sqrt(T)))


# ---------------------------------------------------------------------------
# Color  (DgammaDtime)
# ---------------------------------------------------------------------------

def color(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Rate of change of gamma per calendar day (Gamma decay).

    Uses central-difference numerical differentiation for numerical stability.
    A negative value indicates gamma shrinks as time passes.
    """
    if T <= 1e-6:
        return 0.0

    eps = max(T * 0.005, 1e-5)

    def _gamma(T_val: float) -> float:
        if T_val <= 0:
            return 0.0
        D1 = d1(S, K, T_val, r, sigma)
        return norm.pdf(D1) / (S * sigma * np.sqrt(T_val))

    # dGamma/dT (T = time to expiry) via central differences
    d_gamma_d_tau = (_gamma(T + eps) - _gamma(T - eps)) / (2.0 * eps)

    # Color = -dGamma/dTau per calendar day
    return -d_gamma_d_tau / 365.0


# ---------------------------------------------------------------------------
# Ultima  (third-order vol sensitivity)
# ---------------------------------------------------------------------------

def ultima(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Third derivative of option price with respect to volatility.
    Rate of change of Vomma with respect to vol.

    Ultima = -(Vega_raw / σ²) · [d1·d2·(1 - d1·d2) + d1² + d2²]
    """
    if T <= 1e-8:
        return 0.0

    D1 = d1(S, K, T, r, sigma)
    D2 = d2(S, K, T, r, sigma)
    raw_vega = S * norm.pdf(D1) * np.sqrt(T)
    return -(raw_vega / sigma ** 2) * (
        D1 * D2 * (1.0 - D1 * D2) + D1 ** 2 + D2 ** 2
    )


# ---------------------------------------------------------------------------
# Convenience: return all advanced Greeks as a dict
# ---------------------------------------------------------------------------

def all_advanced_greeks(S: float, K: float, T: float, r: float, sigma: float,
                         option_type: str = "call") -> dict:
    """Return all second- and third-order Greeks in a single dict."""
    return {
        "Charm (per day)":  charm(S, K, T, r, sigma, option_type),
        "Vanna":            vanna(S, K, T, r, sigma),
        "Vomma":            vomma(S, K, T, r, sigma),
        "Speed":            speed(S, K, T, r, sigma),
        "Color (per day)":  color(S, K, T, r, sigma),
        "Ultima":           ultima(S, K, T, r, sigma),
    }
