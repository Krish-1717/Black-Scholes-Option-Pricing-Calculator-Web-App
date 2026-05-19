"""
strategy_builder.py — Multi-leg options strategy P&L engine.

Provides
--------
PREDEFINED_STRATEGIES : dict of strategy name → leg template list
get_strategy_legs()   : convert template + market params → concrete legs
strategy_pnl()        : P&L array across a spot price range at expiry
strategy_pnl_today()  : P&L array at current time (using BS pricing)
pnl_heatmap()         : 2D P&L array (price × time)
strategy_summary()    : max profit, max loss, breakeven points

Each leg is a dict:
    type        : 'call' | 'put' | 'stock'
    action      : 'long' | 'short'
    strike      : absolute strike price (K)
    qty         : number of contracts (1 contract = 100 shares)
    entry_price : premium paid/received per share
    T           : time to expiry (years) — may differ between legs
"""

from __future__ import annotations

import numpy as np
from black_scholes import call_price, put_price


# ---------------------------------------------------------------------------
# Strategy templates
# K_pct_offset : strike offset as fraction of ATM strike
# T_offset     : additional time on this leg vs. main T (for calendars)
# ---------------------------------------------------------------------------

PREDEFINED_STRATEGIES: dict[str, list[dict]] = {
    "Long Call": [
        {"type": "call", "action": "long",  "K_pct_offset": 0.00, "qty": 1},
    ],
    "Long Put": [
        {"type": "put",  "action": "long",  "K_pct_offset": 0.00, "qty": 1},
    ],
    "Short Call (Naked)": [
        {"type": "call", "action": "short", "K_pct_offset": 0.05, "qty": 1},
    ],
    "Short Put (Cash-Secured)": [
        {"type": "put",  "action": "short", "K_pct_offset": -0.05, "qty": 1},
    ],
    "Covered Call": [
        {"type": "stock", "action": "long",  "K_pct_offset": 0.00, "qty": 100},
        {"type": "call",  "action": "short", "K_pct_offset": 0.05, "qty": 1},
    ],
    "Protective Put": [
        {"type": "stock", "action": "long",  "K_pct_offset": 0.00, "qty": 100},
        {"type": "put",   "action": "long",  "K_pct_offset": 0.00, "qty": 1},
    ],
    "Bull Call Spread": [
        {"type": "call", "action": "long",  "K_pct_offset": -0.05, "qty": 1},
        {"type": "call", "action": "short", "K_pct_offset":  0.05, "qty": 1},
    ],
    "Bear Put Spread": [
        {"type": "put", "action": "long",  "K_pct_offset":  0.05, "qty": 1},
        {"type": "put", "action": "short", "K_pct_offset": -0.05, "qty": 1},
    ],
    "Long Straddle": [
        {"type": "call", "action": "long", "K_pct_offset": 0.00, "qty": 1},
        {"type": "put",  "action": "long", "K_pct_offset": 0.00, "qty": 1},
    ],
    "Short Straddle": [
        {"type": "call", "action": "short", "K_pct_offset": 0.00, "qty": 1},
        {"type": "put",  "action": "short", "K_pct_offset": 0.00, "qty": 1},
    ],
    "Long Strangle": [
        {"type": "call", "action": "long", "K_pct_offset":  0.07, "qty": 1},
        {"type": "put",  "action": "long", "K_pct_offset": -0.07, "qty": 1},
    ],
    "Short Strangle": [
        {"type": "call", "action": "short", "K_pct_offset":  0.07, "qty": 1},
        {"type": "put",  "action": "short", "K_pct_offset": -0.07, "qty": 1},
    ],
    "Iron Condor": [
        {"type": "put",  "action": "long",  "K_pct_offset": -0.15, "qty": 1},
        {"type": "put",  "action": "short", "K_pct_offset": -0.05, "qty": 1},
        {"type": "call", "action": "short", "K_pct_offset":  0.05, "qty": 1},
        {"type": "call", "action": "long",  "K_pct_offset":  0.15, "qty": 1},
    ],
    "Iron Butterfly": [
        {"type": "put",  "action": "long",  "K_pct_offset": -0.10, "qty": 1},
        {"type": "put",  "action": "short", "K_pct_offset":  0.00, "qty": 1},
        {"type": "call", "action": "short", "K_pct_offset":  0.00, "qty": 1},
        {"type": "call", "action": "long",  "K_pct_offset":  0.10, "qty": 1},
    ],
    "Butterfly Spread": [
        {"type": "call", "action": "long",  "K_pct_offset": -0.10, "qty": 1},
        {"type": "call", "action": "short", "K_pct_offset":  0.00, "qty": 2},
        {"type": "call", "action": "long",  "K_pct_offset":  0.10, "qty": 1},
    ],
    "Calendar Spread": [
        {"type": "call", "action": "short", "K_pct_offset": 0.00, "qty": 1, "T_offset": 0.0},
        {"type": "call", "action": "long",  "K_pct_offset": 0.00, "qty": 1, "T_offset": 1/12},
    ],
    "Ratio Call Spread (1x2)": [
        {"type": "call", "action": "long",  "K_pct_offset": 0.00, "qty": 1},
        {"type": "call", "action": "short", "K_pct_offset": 0.05, "qty": 2},
    ],
}


# ---------------------------------------------------------------------------
# Build concrete legs from template
# ---------------------------------------------------------------------------

def get_strategy_legs(
    strategy_name: str,
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> list[dict]:
    """
    Convert a predefined strategy template into concrete legs with
    absolute strikes, entry prices, and times to expiry.
    """
    template = PREDEFINED_STRATEGIES.get(strategy_name, [])
    legs = []

    for tmpl in template:
        strike = K * (1.0 + tmpl.get("K_pct_offset", 0.0))
        leg_T  = T + tmpl.get("T_offset", 0.0)
        qty    = tmpl.get("qty", 1)
        ltype  = tmpl["type"]
        action = tmpl["action"]

        if ltype == "stock":
            entry_price = S
        elif ltype == "call":
            entry_price = call_price(S, strike, max(leg_T, 1e-6), r, sigma)
        else:
            entry_price = put_price(S, strike, max(leg_T, 1e-6), r, sigma)

        legs.append({
            "type":        ltype,
            "action":      action,
            "strike":      strike,
            "qty":         qty,
            "entry_price": entry_price,
            "T":           leg_T,
        })

    return legs


# ---------------------------------------------------------------------------
# P&L at expiration
# ---------------------------------------------------------------------------

def strategy_pnl(
    legs: list[dict],
    S: float,
    spot_range: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """
    P&L at expiration across a range of stock prices.

    Returns
    -------
    pnl       : (n_spots,) array of P&L in dollars
    net_debit : total net premium paid (positive = debit, negative = credit)
    """
    if spot_range is None:
        # centre on current S, ±50 %
        spot_range = np.linspace(S * 0.50, S * 1.50, 400)

    total_pnl  = np.zeros(len(spot_range))
    net_debit  = 0.0
    multiplier = 100  # 1 contract = 100 shares

    for leg in legs:
        sign  = 1.0 if leg["action"] == "long" else -1.0
        qty   = leg["qty"]
        entry = leg["entry_price"]
        ltype = leg["type"]

        if ltype == "stock":
            pnl = sign * qty * (spot_range - entry)
            net_debit += sign * qty * entry
        else:
            if ltype == "call":
                payoff = np.maximum(spot_range - leg["strike"], 0.0)
            else:
                payoff = np.maximum(leg["strike"] - spot_range, 0.0)
            pnl = sign * qty * multiplier * (payoff - entry)
            net_debit += sign * qty * multiplier * entry

        total_pnl += pnl

    return total_pnl, net_debit


# ---------------------------------------------------------------------------
# P&L at current time (before expiry)
# ---------------------------------------------------------------------------

def strategy_pnl_now(
    legs: list[dict],
    S_range: np.ndarray,
    r: float,
    sigma: float,
) -> np.ndarray:
    """
    P&L at the CURRENT moment (not at expiry) for each spot in S_range.
    Uses BS pricing for remaining option value.
    """
    total_pnl  = np.zeros(len(S_range))
    multiplier = 100

    for leg in legs:
        sign  = 1.0 if leg["action"] == "long" else -1.0
        qty   = leg["qty"]
        entry = leg["entry_price"]
        ltype = leg["type"]
        T_rem = max(float(leg.get("T", 0)), 1e-6)

        if ltype == "stock":
            total_pnl += sign * qty * (S_range - entry)
            continue

        current_vals = np.array([
            (call_price(s, leg["strike"], T_rem, r, sigma)
             if ltype == "call"
             else put_price(s, leg["strike"], T_rem, r, sigma))
            for s in S_range
        ])
        total_pnl += sign * qty * multiplier * (current_vals - entry)

    return total_pnl


# ---------------------------------------------------------------------------
# P&L heatmap (price × time)
# ---------------------------------------------------------------------------

def pnl_heatmap(
    legs: list[dict],
    S: float,
    T: float,
    r: float,
    sigma: float,
    n_price: int = 60,
    n_time:  int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a P&L grid across stock prices and time remaining.

    Returns
    -------
    spot_range : (n_price,) array
    days_range : (n_time,) array — days remaining to expiry
    pnl_matrix : (n_time, n_price) array — P&L in dollars
    """
    spot_range = np.linspace(S * 0.60, S * 1.40, n_price)
    days_range = np.linspace(T * 365, 0, n_time)  # from full DTE → 0
    t_range    = days_range / 365.0

    pnl_matrix = np.zeros((n_time, n_price))
    multiplier = 100

    for i, t_rem in enumerate(t_range):
        for j, spot in enumerate(spot_range):
            total = 0.0
            for leg in legs:
                sign    = 1.0 if leg["action"] == "long" else -1.0
                qty     = leg["qty"]
                entry   = leg["entry_price"]
                ltype   = leg["type"]
                leg_T   = float(leg.get("T", T))
                t_left  = max(leg_T - (T - t_rem), 0.0)

                if ltype == "stock":
                    total += sign * qty * (spot - entry)
                    continue

                if t_left <= 0:
                    if ltype == "call":
                        current = max(spot - leg["strike"], 0.0)
                    else:
                        current = max(leg["strike"] - spot, 0.0)
                else:
                    if ltype == "call":
                        current = call_price(spot, leg["strike"], t_left, r, sigma)
                    else:
                        current = put_price(spot, leg["strike"], t_left, r, sigma)

                total += sign * qty * multiplier * (current - entry)

            pnl_matrix[i, j] = total

    return spot_range, days_range, pnl_matrix


# ---------------------------------------------------------------------------
# Strategy summary metrics
# ---------------------------------------------------------------------------

def strategy_summary(
    legs: list[dict],
    S: float,
    pnl: np.ndarray,
    spot_range: np.ndarray,
    net_debit: float,
) -> dict:
    """
    Compute max profit, max loss, and breakeven prices.

    Returns dict with:
        max_profit, max_loss, breakevens, net_debit
    """
    max_profit = float(np.max(pnl))
    max_loss   = float(np.min(pnl))

    # Breakevens: zero-crossings of the P&L curve
    sign_changes = np.where(np.diff(np.sign(pnl)))[0]
    breakevens = []
    for idx in sign_changes:
        # Linear interpolation between idx and idx+1
        x0, x1 = spot_range[idx], spot_range[idx + 1]
        y0, y1 = pnl[idx], pnl[idx + 1]
        if y1 - y0 != 0:
            be = x0 - y0 * (x1 - x0) / (y1 - y0)
            breakevens.append(round(be, 2))

    return {
        "max_profit": max_profit if max_profit < 1e8 else float("inf"),
        "max_loss":   max_loss   if max_loss   > -1e8 else float("-inf"),
        "breakevens": breakevens,
        "net_debit":  net_debit,
    }
