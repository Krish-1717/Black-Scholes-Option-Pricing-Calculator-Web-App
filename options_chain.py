"""
options_chain.py — Enrich a live options chain with BS theoretical prices,
implied volatility, and mispricing signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from black_scholes import call_price, put_price


# ---------------------------------------------------------------------------
# Implied volatility solver
# ---------------------------------------------------------------------------

def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    lo: float = 1e-4,
    hi: float = 10.0,
) -> float:
    """
    Solve for implied volatility via Brent's method.

    Returns np.nan if no solution exists or inputs are invalid.
    """
    if T <= 0 or market_price <= 0 or np.isnan(market_price):
        return np.nan

    intrinsic = (
        max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    )
    if market_price < max(intrinsic - 0.01, 0):
        return np.nan

    pricer = call_price if option_type == "call" else put_price

    try:
        f_lo = pricer(S, K, T, r, lo) - market_price
        f_hi = pricer(S, K, T, r, hi) - market_price
        if f_lo * f_hi > 0:
            return np.nan
        return float(brentq(lambda v: pricer(S, K, T, r, v) - market_price,
                            lo, hi, maxiter=200, xtol=1e-6))
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Chain enrichment
# ---------------------------------------------------------------------------

def enrich_chain(
    df: pd.DataFrame,
    S: float,
    T: float,
    r: float,
    option_type: str = "call",
    atm_band: float = 0.30,
) -> pd.DataFrame:
    """
    Add Black-Scholes theoretical price, BS-derived IV, and mispricing
    columns to a raw yfinance options DataFrame.

    Parameters
    ----------
    df          : raw calls or puts DataFrame from yfinance
    S           : current stock price
    T           : time to expiry in years
    r           : risk-free rate (decimal)
    option_type : 'call' | 'put'
    atm_band    : filter to strikes within ±atm_band of spot (e.g. 0.30 = ±30 %)

    New columns
    -----------
    mid_price      : (bid + ask) / 2, fallback to lastPrice
    bs_price       : Black-Scholes theoretical price (using yfinance IV)
    our_iv         : IV calculated from mid_price
    mispricing     : mid_price - bs_price
    mispricing_pct : mispricing / bs_price * 100
    signal         : "overpriced" | "underpriced" | "fair"
    """
    if df.empty:
        return df

    df = df.copy()

    # Compute mid price
    has_bid_ask = (df["bid"] > 0) & (df["ask"] > 0) & df["bid"].notna() & df["ask"].notna()
    df["mid_price"] = np.where(has_bid_ask, (df["bid"] + df["ask"]) / 2.0, df["lastPrice"])
    df["mid_price"] = df["mid_price"].fillna(df["lastPrice"])

    # Filter to near-the-money strikes
    lo_strike = S * (1.0 - atm_band)
    hi_strike = S * (1.0 + atm_band)
    df = df[(df["strike"] >= lo_strike) & (df["strike"] <= hi_strike)].copy()

    if df.empty:
        return df

    # Vectorised calculations
    pricer = call_price if option_type == "call" else put_price
    bs_prices = []
    our_ivs   = []

    for _, row in df.iterrows():
        K = float(row["strike"])
        # Use yfinance's IV for BS theoretical price (more stable than recalculating)
        raw_iv = row.get("impliedVolatility")
        yf_iv = 0.25 if (raw_iv is None or pd.isna(raw_iv) or float(raw_iv) <= 0) else float(raw_iv)
        yf_iv = max(yf_iv, 0.01)
        bs_p = pricer(S, K, T, r, yf_iv)
        bs_prices.append(bs_p)

        raw_mid = row["mid_price"]
        mid = 0.0 if (raw_mid is None or pd.isna(raw_mid)) else float(raw_mid)
        iv = implied_vol(mid, S, K, T, r, option_type) if mid > 0 else np.nan
        our_ivs.append(iv)

    df["bs_price"]     = bs_prices
    df["our_iv"]       = our_ivs
    df["yf_iv_pct"]    = (df["impliedVolatility"] * 100.0).round(2)
    df["our_iv_pct"]   = (df["our_iv"] * 100.0).round(2)
    df["mispricing"]   = (df["mid_price"] - df["bs_price"]).round(4)
    df["mispricing_pct"] = (
        (df["mispricing"] / df["bs_price"].replace(0, np.nan)) * 100
    ).round(2)

    def _signal(row) -> str:
        mp = row["mispricing_pct"]
        if np.isnan(mp):
            return "—"
        if mp > 5:
            return "overpriced"
        if mp < -5:
            return "underpriced"
        return "fair"

    df["signal"] = df.apply(_signal, axis=1)

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Volatility smile from a single expiry
# ---------------------------------------------------------------------------

def vol_smile(
    calls_df: pd.DataFrame,
    puts_df: pd.DataFrame,
    S: float,
    T: float,
    r: float,
    atm_band: float = 0.35,
) -> pd.DataFrame:
    """
    Extract implied volatility across strikes for a single expiry.

    Uses puts for K < S, calls for K >= S (standard convention for smile).
    Returns DataFrame with columns: strike, moneyness, iv_pct, option_type.
    """
    records = []

    for df, otype in [(puts_df, "put"), (calls_df, "call")]:
        if df.empty:
            continue
        for _, row in df.iterrows():
            K = float(row["strike"])
            if not (S * (1 - atm_band) <= K <= S * (1 + atm_band)):
                continue

            has_ba = row.get("bid", 0) > 0 and row.get("ask", 0) > 0
            mid = (row["bid"] + row["ask"]) / 2.0 if has_ba else float(row.get("lastPrice", 0))
            if mid <= 0:
                continue

            iv = implied_vol(mid, S, K, T, r, otype)
            if np.isnan(iv) or iv < 0.01 or iv > 5.0:
                continue

            records.append({
                "strike":      K,
                "moneyness":   K / S,
                "iv_pct":      iv * 100.0,
                "option_type": otype,
            })

    return pd.DataFrame(records).sort_values("strike").reset_index(drop=True)
