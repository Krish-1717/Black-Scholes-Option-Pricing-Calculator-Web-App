"""
yield_curve.py — Yield curve construction and interpolation.

Methods:
* Bootstrap from par rates
* Nelson-Siegel parametric fitting
* Interpolation: linear, cubic spline
* Forward rate extraction
* Treasury yield fetching via yfinance (^IRX, ^FVX, ^TNX, ^TYX)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline


# ─────────────────────────────────────────────────────────────────────────────
# US Treasury spot rates (live via yfinance)
# ─────────────────────────────────────────────────────────────────────────────

TREASURY_TICKERS = {
    0.25: "^IRX",   # 13-week T-bill
    2.0:  "^TXN",   # 2-year (proxy)
    5.0:  "^FVX",   # 5-year
    10.0: "^TNX",   # 10-year
    30.0: "^TYX",   # 30-year
}

_FALLBACK_CURVE = {
    0.25: 0.052,
    0.5:  0.051,
    1.0:  0.049,
    2.0:  0.047,
    3.0:  0.046,
    5.0:  0.045,
    7.0:  0.045,
    10.0: 0.044,
    20.0: 0.046,
    30.0: 0.047,
}


def get_treasury_yields() -> dict[float, float]:
    """
    Fetch current US Treasury yields.
    Returns a dict {maturity_years: yield_decimal}.
    Falls back to a reasonable static curve if yfinance is unavailable.
    """
    try:
        import yfinance as yf
        result = {}
        for mat, ticker in TREASURY_TICKERS.items():
            try:
                hist = yf.Ticker(ticker).history(period="5d")
                if not hist.empty:
                    rate = float(hist["Close"].iloc[-1]) / 100
                    result[mat] = rate
            except Exception:
                pass
        if len(result) >= 2:
            return dict(sorted(result.items()))
    except Exception:
        pass
    return _FALLBACK_CURVE.copy()


# ─────────────────────────────────────────────────────────────────────────────
# Nelson-Siegel model
# ─────────────────────────────────────────────────────────────────────────────

def nelson_siegel_rate(t: np.ndarray, beta0: float, beta1: float,
                        beta2: float, tau: float) -> np.ndarray:
    """
    Nelson-Siegel yield curve.

    y(t) = β₀ + β₁*(1-e^(-t/τ))/(t/τ) + β₂*[(1-e^(-t/τ))/(t/τ) - e^(-t/τ)]
    """
    t = np.asarray(t, dtype=float)
    t = np.where(t < 1e-8, 1e-8, t)
    x = t / tau
    factor1 = (1 - np.exp(-x)) / x
    factor2 = factor1 - np.exp(-x)
    return beta0 + beta1 * factor1 + beta2 * factor2


def fit_nelson_siegel(maturities: np.ndarray, yields: np.ndarray) -> dict:
    """
    Fit Nelson-Siegel parameters to observed (maturity, yield) pairs.

    Returns {'beta0', 'beta1', 'beta2', 'tau', 'rmse'}.
    """
    maturities = np.asarray(maturities, dtype=float)
    yields = np.asarray(yields, dtype=float)

    def objective(params):
        b0, b1, b2, tau = params
        if tau <= 0:
            return 1e10
        fitted = nelson_siegel_rate(maturities, b0, b1, b2, tau)
        return float(np.mean((fitted - yields) ** 2))

    best_res = None
    for tau0 in [1.0, 2.0, 5.0, 10.0]:
        x0 = [yields[-1], yields[0] - yields[-1], 0.0, tau0]
        res = minimize(objective, x0, method="Nelder-Mead",
                       options={"maxiter": 5000, "xatol": 1e-9, "fatol": 1e-9})
        if best_res is None or res.fun < best_res.fun:
            best_res = res

    b0, b1, b2, tau = best_res.x
    tau = abs(tau)
    fitted = nelson_siegel_rate(maturities, b0, b1, b2, tau)
    rmse = float(np.sqrt(np.mean((fitted - yields) ** 2)))

    return {"beta0": float(b0), "beta1": float(b1), "beta2": float(b2),
            "tau": float(tau), "rmse": rmse}


# ─────────────────────────────────────────────────────────────────────────────
# Interpolation
# ─────────────────────────────────────────────────────────────────────────────

class YieldCurve:
    """
    Yield curve with multiple interpolation methods.

    Parameters
    ----------
    maturities : array of maturities (years)
    yields     : array of spot yields (decimal)
    method     : 'linear' | 'cubic' | 'nelson-siegel'
    """

    def __init__(self,
                 maturities: np.ndarray,
                 yields: np.ndarray,
                 method: str = "cubic"):
        self.maturities = np.asarray(maturities, dtype=float)
        self.yields = np.asarray(yields, dtype=float)
        self.method = method
        self._ns_params = None

        if method == "nelson-siegel":
            self._ns_params = fit_nelson_siegel(self.maturities, self.yields)
        elif method == "cubic":
            self._cs = CubicSpline(self.maturities, self.yields,
                                   bc_type="not-a-knot", extrapolate=True)

    def spot_rate(self, t: float | np.ndarray) -> np.ndarray:
        """Interpolated spot rate at maturity t (years)."""
        t = np.asarray(t, dtype=float)
        if self.method == "nelson-siegel":
            p = self._ns_params
            return nelson_siegel_rate(t, p["beta0"], p["beta1"], p["beta2"], p["tau"])
        elif self.method == "cubic":
            return self._cs(t)
        else:  # linear
            return np.interp(t, self.maturities, self.yields)

    def discount_factor(self, t: float | np.ndarray) -> np.ndarray:
        """Continuous compounding discount factor: exp(-r(t)*t)."""
        t = np.asarray(t, dtype=float)
        r = self.spot_rate(t)
        return np.exp(-r * t)

    def forward_rate(self, t1: float, t2: float) -> float:
        """
        Implied forward rate between t1 and t2.
        f(t1, t2) = [r(t2)*t2 - r(t1)*t1] / (t2 - t1)
        """
        if t2 <= t1:
            raise ValueError("t2 must be greater than t1")
        r1 = float(self.spot_rate(t1))
        r2 = float(self.spot_rate(t2))
        return (r2 * t2 - r1 * t1) / (t2 - t1)

    def par_rate(self, maturity: float, frequency: int = 2) -> float:
        """
        Par coupon rate for a bond with given maturity (no-arbitrage coupon
        that prices the bond at par).
        """
        from bond_pricer import Bond, bond_price as bp
        from scipy.optimize import brentq

        def objective(c):
            b = Bond(face_value=100.0, coupon_rate=c, maturity=maturity, frequency=frequency)
            t = b.time_to_cash_flows()
            r = self.spot_rate(t)
            df = (1 + r / frequency) ** (-np.arange(1, b.n_periods + 1))
            return float(np.dot(b.cash_flows(), df)) - 100.0

        try:
            return float(brentq(objective, 0.0001, 0.3, xtol=1e-10))
        except Exception:
            return float(self.spot_rate(maturity))

    def forward_curve(self, tenors: np.ndarray, tenor_width: float = 0.25) -> np.ndarray:
        """Instantaneous forward rates at each tenor."""
        return np.array([self.forward_rate(t, t + tenor_width) for t in tenors])


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap from par rates
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_spot_rates(par_rates: dict[float, float],
                         frequency: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    Bootstrap spot rates from par coupon rates using the bootstrap method.

    par_rates : {maturity: par_rate} e.g. {0.5: 0.04, 1.0: 0.042, ...}
    Returns (maturities, spot_rates) arrays.
    """
    items = sorted(par_rates.items())
    maturities = np.array([m for m, _ in items])
    par = np.array([r for _, r in items])
    spot = np.zeros_like(par)

    for i, (mat, c) in enumerate(zip(maturities, par)):
        n = int(round(mat * frequency))
        period = 1.0 / frequency
        coupon = c * 100.0 / frequency

        if i == 0:
            # First period: simple discount
            spot[i] = (100.0 / (100.0 + coupon)) ** (1.0 / mat) - 1
            continue

        # Sum of discounted intermediate coupons using already-bootstrapped spots
        pv_coupons = 0.0
        for j in range(i):
            t_j = (j + 1) * period
            r_j = np.interp(t_j, maturities[: i + 1], spot[: i + 1])
            pv_coupons += coupon / (1 + r_j) ** (t_j)

        # Solve for spot[i]
        last_cf = 100.0 + coupon
        spot[i] = (last_cf / (100.0 - pv_coupons)) ** (1.0 / mat) - 1

    return maturities, spot
