"""
bond_pricer.py — Fixed income analytics: bond pricing, YTM, duration, convexity.

Supports:
* Fixed-coupon bonds (semi-annual or annual)
* Zero-coupon bonds
* Floating rate notes (simplified)
* Duration: Macaulay, Modified, Dollar (DV01)
* Convexity and dollar convexity
* Yield-to-maturity (Newton-Raphson + fallback bisection)
* Price sensitivity: full repricing and Taylor approximation
* Spread analysis: Z-spread, OAS stub
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Literal


# ─────────────────────────────────────────────────────────────────────────────
# Bond dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Bond:
    """
    Fixed-coupon bond.

    Parameters
    ----------
    face_value    : par / notional amount
    coupon_rate   : annual coupon rate (decimal)
    maturity      : years to maturity (can be fractional)
    frequency     : coupon payments per year (1 = annual, 2 = semi-annual)
    """
    face_value: float = 1000.0
    coupon_rate: float = 0.05
    maturity: float = 10.0
    frequency: int = 2          # semi-annual by default

    @property
    def coupon_payment(self) -> float:
        return self.face_value * self.coupon_rate / self.frequency

    @property
    def n_periods(self) -> int:
        return int(round(self.maturity * self.frequency))

    @property
    def period_length(self) -> float:
        return 1.0 / self.frequency

    def cash_flows(self) -> np.ndarray:
        """Return the array of cash flows (coupon + principal at maturity)."""
        n = self.n_periods
        cf = np.full(n, self.coupon_payment)
        cf[-1] += self.face_value
        return cf

    def time_to_cash_flows(self) -> np.ndarray:
        """Time (in years) to each cash flow."""
        return np.arange(1, self.n_periods + 1) * self.period_length


# ─────────────────────────────────────────────────────────────────────────────
# Pricing
# ─────────────────────────────────────────────────────────────────────────────

def bond_price(bond: Bond, ytm: float) -> float:
    """
    Price a fixed-coupon bond given a flat yield-to-maturity.

    ytm is the annual yield (decimal). Compounding matches coupon frequency.
    """
    y_per = ytm / bond.frequency
    times = np.arange(1, bond.n_periods + 1)
    df = (1 + y_per) ** (-times)
    cf = bond.cash_flows()
    return float(np.dot(cf, df))


def zero_coupon_price(face_value: float, ytm: float, maturity: float) -> float:
    """Price a zero-coupon bond."""
    return face_value / (1 + ytm) ** maturity


def bond_price_from_spread(bond: Bond, benchmark_ytm: float, spread: float) -> float:
    """Price using benchmark yield + spread (e.g. Treasury + OAS)."""
    return bond_price(bond, benchmark_ytm + spread)


# ─────────────────────────────────────────────────────────────────────────────
# Yield-to-Maturity
# ─────────────────────────────────────────────────────────────────────────────

def ytm(bond: Bond, price: float, tol: float = 1e-10, max_iter: int = 500) -> float:
    """
    Compute yield-to-maturity via Newton-Raphson.

    Falls back to bisection if Newton diverges.
    """
    # Newton-Raphson
    y = bond.coupon_rate  # initial guess = coupon rate
    for _ in range(max_iter):
        p = bond_price(bond, y)
        dp_dy = _price_derivative(bond, y)
        if abs(dp_dy) < 1e-12:
            break
        y_new = y - (p - price) / dp_dy
        if abs(y_new - y) < tol:
            return float(y_new)
        y = y_new

    # Bisection fallback
    lo, hi = -0.999, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if bond_price(bond, mid) > price:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return float(0.5 * (lo + hi))


def _price_derivative(bond: Bond, y: float) -> float:
    """dP/dy (analytic derivative)."""
    y_per = y / bond.frequency
    times = np.arange(1, bond.n_periods + 1)
    df = (1 + y_per) ** (-(times + 1))
    cf = bond.cash_flows()
    return float(-np.dot(cf * times, df) / bond.frequency)


# ─────────────────────────────────────────────────────────────────────────────
# Duration & Convexity
# ─────────────────────────────────────────────────────────────────────────────

def macaulay_duration(bond: Bond, ytm_: float) -> float:
    """
    Macaulay duration (in years): weighted average time to cash flows.
    """
    y_per = ytm_ / bond.frequency
    times = np.arange(1, bond.n_periods + 1)
    t_years = times * bond.period_length
    df = (1 + y_per) ** (-times)
    cf = bond.cash_flows()
    pv = cf * df
    p = pv.sum()
    return float(np.dot(pv, t_years) / p)


def modified_duration(bond: Bond, ytm_: float) -> float:
    """
    Modified duration = Macaulay duration / (1 + y/m).
    Measures % price change per 1 unit change in yield.
    """
    mac = macaulay_duration(bond, ytm_)
    return float(mac / (1 + ytm_ / bond.frequency))


def dollar_duration(bond: Bond, ytm_: float, price: float | None = None) -> float:
    """
    Dollar duration (DV01 * 10000): $ price change per 1% change in yield.
    DV01 = dollar_duration / 10000
    """
    if price is None:
        price = bond_price(bond, ytm_)
    return float(modified_duration(bond, ytm_) * price)


def dv01(bond: Bond, ytm_: float, price: float | None = None) -> float:
    """
    DV01 (Dollar Value of 1 basis point).
    Approximate $ price change for a 1bp increase in yield.
    """
    if price is None:
        price = bond_price(bond, ytm_)
    p_up = bond_price(bond, ytm_ + 0.0001)
    p_dn = bond_price(bond, ytm_ - 0.0001)
    return float((p_dn - p_up) / 2)


def convexity(bond: Bond, ytm_: float) -> float:
    """
    Convexity: second-order price sensitivity to yield.

    C = (1/P) * d²P/dy²
    """
    y_per = ytm_ / bond.frequency
    times = np.arange(1, bond.n_periods + 1)
    t_years = times * bond.period_length
    df = (1 + y_per) ** (-times)
    cf = bond.cash_flows()
    pv = cf * df
    p = pv.sum()
    # d²P/dy² = sum[ CF_t * t * (t+1/m) / (1+y/m)^(t+2) ] / m^2
    numer = np.dot(cf * t_years * (t_years + bond.period_length), df) / (1 + y_per) ** 2
    return float(numer / p)


def dollar_convexity(bond: Bond, ytm_: float, price: float | None = None) -> float:
    if price is None:
        price = bond_price(bond, ytm_)
    return float(convexity(bond, ytm_) * price)


# ─────────────────────────────────────────────────────────────────────────────
# Price sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def price_change_taylor(bond: Bond, ytm_: float, dy: float) -> dict:
    """
    Estimate price change for a yield shift dy using Taylor approximation.

    dP ≈ -MD * P * dy + 0.5 * C * P * dy²
    """
    p = bond_price(bond, ytm_)
    md = modified_duration(bond, ytm_)
    cx = convexity(bond, ytm_)
    dp_duration = -md * p * dy
    dp_convexity = 0.5 * cx * p * dy ** 2
    dp_total = dp_duration + dp_convexity
    p_new_approx = p + dp_total
    p_new_exact = bond_price(bond, ytm_ + dy)
    return {
        "price_initial": p,
        "yield_initial": ytm_,
        "yield_new": ytm_ + dy,
        "dp_duration": dp_duration,
        "dp_convexity": dp_convexity,
        "dp_total_approx": dp_total,
        "price_new_approx": p_new_approx,
        "price_new_exact": p_new_exact,
        "approximation_error": abs(p_new_approx - p_new_exact),
    }


def price_yield_curve(bond: Bond, ytm_range: np.ndarray) -> np.ndarray:
    """Return bond prices for an array of yields."""
    return np.array([bond_price(bond, y) for y in ytm_range])


# ─────────────────────────────────────────────────────────────────────────────
# Spread calculations
# ─────────────────────────────────────────────────────────────────────────────

def z_spread(
    bond: Bond,
    market_price: float,
    spot_rates: np.ndarray,
    spot_maturities: np.ndarray,
    tol: float = 1e-8,
) -> float:
    """
    Z-spread: constant spread added to each spot rate such that the
    discounted cash flows equal the market price.

    spot_rates      : array of spot rates at corresponding maturities
    spot_maturities : array of maturities matching spot_rates (years)
    """
    cf = bond.cash_flows()
    t = bond.time_to_cash_flows()

    def price_with_spread(s: float) -> float:
        interp_rates = np.interp(t, spot_maturities, spot_rates)
        df = (1 + interp_rates + s) ** (-t)
        return float(np.dot(cf, df))

    # Bisection
    lo, hi = -0.999, 5.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        p = price_with_spread(mid)
        if p > market_price:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return float(0.5 * (lo + hi))


# ─────────────────────────────────────────────────────────────────────────────
# Full analytics summary
# ─────────────────────────────────────────────────────────────────────────────

def bond_analytics(bond: Bond, ytm_: float | None = None,
                   market_price: float | None = None) -> dict:
    """
    Return a comprehensive analytics dictionary for a bond.

    Provide either ytm_ or market_price (not both).
    """
    if ytm_ is None and market_price is None:
        raise ValueError("Provide either ytm_ or market_price.")
    if market_price is not None and ytm_ is None:
        ytm_ = ytm(bond, market_price)
    if market_price is None:
        market_price = bond_price(bond, ytm_)

    mac_dur = macaulay_duration(bond, ytm_)
    mod_dur = modified_duration(bond, ytm_)
    dol_dur = dollar_duration(bond, ytm_, market_price)
    cx = convexity(bond, ytm_)
    dv = dv01(bond, ytm_, market_price)

    return {
        "price": market_price,
        "ytm": ytm_,
        "coupon_rate": bond.coupon_rate,
        "current_yield": bond.coupon_payment * bond.frequency / market_price,
        "macaulay_duration": mac_dur,
        "modified_duration": mod_dur,
        "dollar_duration": dol_dur,
        "dv01": dv,
        "convexity": cx,
        "dollar_convexity": dollar_convexity(bond, ytm_, market_price),
        "n_periods": bond.n_periods,
        "frequency": bond.frequency,
        "face_value": bond.face_value,
        "price_to_par": market_price / bond.face_value,
    }
