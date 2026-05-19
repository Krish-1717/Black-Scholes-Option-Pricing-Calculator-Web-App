"""
volatility_surface.py — Build implied-volatility surfaces and vol smiles.

Functions
---------
build_surface_from_chain()  : IV surface from live options chain data
build_synthetic_surface()   : parametric skew/smile surface (no live data)
vol_term_structure()        : average ATM IV vs. time to expiry
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from black_scholes import call_price, put_price


# ---------------------------------------------------------------------------
# IV solver (shared helper)
# ---------------------------------------------------------------------------

def _calc_iv(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
) -> float:
    """Return implied volatility, or np.nan on failure."""
    if T <= 0 or market_price <= 0 or np.isnan(market_price):
        return np.nan

    pricer = call_price if option_type == "call" else put_price
    intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    if market_price < max(intrinsic - 0.01, 0):
        return np.nan

    try:
        lo, hi = 1e-4, 10.0
        f_lo = pricer(S, K, T, r, lo) - market_price
        f_hi = pricer(S, K, T, r, hi) - market_price
        if f_lo * f_hi > 0:
            return np.nan
        return float(brentq(
            lambda v: pricer(S, K, T, r, v) - market_price,
            lo, hi, maxiter=200, xtol=1e-6,
        ))
    except Exception:
        return np.nan


def _mid(row: pd.Series) -> float:
    """Best-effort mid price from a row."""
    bid = float(row.get("bid") or 0)
    ask = float(row.get("ask") or 0)
    last = float(row.get("lastPrice") or 0)
    if bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return last


# ---------------------------------------------------------------------------
# Surface from live chain data
# ---------------------------------------------------------------------------

def build_surface_from_chain(
    chain_by_expiry: dict,   # {expiry_str: (T_years, calls_df, puts_df)}
    S: float,
    r: float,
    atm_band: float = 0.40,
) -> pd.DataFrame:
    """
    Build an IV surface from multiple expiry slices.

    Parameters
    ----------
    chain_by_expiry : dict mapping expiry string → (T_years, calls_df, puts_df)
    S               : current spot price
    r               : risk-free rate
    atm_band        : half-width of strike filter (fraction of S)

    Returns
    -------
    DataFrame with columns:
        strike, expiry, T_days, moneyness, iv_pct, option_type
    """
    records = []

    for expiry, (T, calls_df, puts_df) in chain_by_expiry.items():
        if T <= 0:
            continue
        T_days = T * 365.0

        for df, otype in [(calls_df, "call"), (puts_df, "put")]:
            if df is None or df.empty:
                continue
            for _, row in df.iterrows():
                K = float(row["strike"])
                if not (S * (1 - atm_band) <= K <= S * (1 + atm_band)):
                    continue

                mid = _mid(row)
                if mid <= 0:
                    continue

                iv = _calc_iv(mid, S, K, T, r, otype)
                if np.isnan(iv) or iv < 0.01 or iv > 4.0:
                    continue

                records.append({
                    "strike":      K,
                    "expiry":      expiry,
                    "T_days":      round(T_days, 1),
                    "moneyness":   round(K / S, 4),
                    "iv_pct":      round(iv * 100, 2),
                    "option_type": otype,
                })

    if not records:
        return pd.DataFrame()

    df_out = pd.DataFrame(records)
    # De-duplicate: keep call for K >= S, put for K < S (reduces spread bias)
    call_mask = (df_out["moneyness"] >= 1.0) & (df_out["option_type"] == "call")
    put_mask  = (df_out["moneyness"]  < 1.0) & (df_out["option_type"] == "put")
    return df_out[call_mask | put_mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Synthetic surface (parametric equity vol skew)
# ---------------------------------------------------------------------------

def build_synthetic_surface(
    S: float,
    K_atm: float,
    T_values: list[float] | None = None,
    base_vol: float = 0.22,
    skew_slope: float = -0.15,
    smile_curv: float = 0.40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic IV surface using a simple parametric model.

    IV(K, T) = base_vol
               + skew_slope * ln(K/S)          ← negative skew (equity)
               + smile_curv * ln(K/S)^2         ← smile curvature
               + 0.02 * sqrt(T)                 ← upward term structure

    Returns
    -------
    strikes      : (n_strikes,) array
    T_values     : (n_T,) array of years
    vol_matrix   : (n_T, n_strikes) array of IV in percent
    """
    if T_values is None:
        T_values = [1/52, 1/12, 2/12, 3/12, 6/12, 1.0, 1.5, 2.0]

    strikes = np.linspace(K_atm * 0.65, K_atm * 1.35, 60)
    T_arr   = np.array(T_values)
    vol_mat = np.zeros((len(T_arr), len(strikes)))

    for i, T in enumerate(T_arr):
        for j, K in enumerate(strikes):
            lm = np.log(K / S)
            vol = base_vol + skew_slope * lm + smile_curv * lm ** 2 + 0.02 * np.sqrt(T)
            vol_mat[i, j] = max(vol * 100.0, 3.0)   # floor at 3 %

    return strikes, T_arr, vol_mat


# ---------------------------------------------------------------------------
# ATM term structure
# ---------------------------------------------------------------------------

def vol_term_structure(
    surface_df: pd.DataFrame,
    S: float,
    atm_band: float = 0.05,
) -> pd.DataFrame:
    """
    Compute average ATM IV for each expiry from a surface DataFrame.

    Returns DataFrame with columns: T_days, atm_iv_pct, expiry.
    """
    if surface_df.empty:
        return pd.DataFrame()

    atm = surface_df[
        (surface_df["moneyness"] >= 1.0 - atm_band) &
        (surface_df["moneyness"] <= 1.0 + atm_band)
    ]

    if atm.empty:
        return pd.DataFrame()

    result = (
        atm.groupby(["expiry", "T_days"])["iv_pct"]
        .mean()
        .reset_index()
        .rename(columns={"iv_pct": "atm_iv_pct"})
        .sort_values("T_days")
    )
    return result
