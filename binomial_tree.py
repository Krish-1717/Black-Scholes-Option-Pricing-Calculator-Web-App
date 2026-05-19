"""
binomial_tree.py — Cox-Ross-Rubinstein binomial tree pricer.

Supports both American and European options. Includes a helper that
returns a small tree (≤8 steps) for on-screen visualisation.
"""

import numpy as np


def binomial_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int = 200,
    option_type: str = "call",
    american: bool = False,
    q: float = 0.0,
) -> float:
    """
    Price an option via the CRR binomial tree.

    Parameters
    ----------
    S         : current stock price
    K         : strike price
    T         : time to expiry (years)
    r         : risk-free rate (decimal)
    sigma     : annualised volatility (decimal)
    n_steps   : number of time steps (more = more accurate)
    option_type : 'call' | 'put'
    american  : True to allow early exercise (American option)
    q         : continuous dividend yield (decimal), default 0.0

    Returns
    -------
    option price (float)
    """
    if T <= 0:
        return max(S - K, 0) if option_type == "call" else max(K - S, 0)

    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    # Terminal stock prices (vectorised)
    j = np.arange(n_steps + 1)
    ST = S * (u ** j) * (d ** (n_steps - j))

    # Terminal payoffs
    if option_type == "call":
        option_vals = np.maximum(ST - K, 0.0)
    else:
        option_vals = np.maximum(K - ST, 0.0)

    # Backward induction
    for i in range(n_steps - 1, -1, -1):
        # Stock prices at this step
        j = np.arange(i + 1)
        stock = S * (u ** j) * (d ** (i - j))

        option_vals = discount * (p * option_vals[1 : i + 2] + (1 - p) * option_vals[: i + 1])

        if american:
            if option_type == "call":
                intrinsic = np.maximum(stock - K, 0.0)
            else:
                intrinsic = np.maximum(K - stock, 0.0)
            option_vals = np.maximum(option_vals, intrinsic)

    return float(option_vals[0])


def tree_for_viz(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int = 7,
    option_type: str = "call",
    american: bool = False,
    q: float = 0.0,
) -> dict:
    """
    Build a small binomial tree (≤ n_steps) and return node data
    suitable for Plotly scatter/annotation rendering.

    Returns
    -------
    dict with keys:
        price        : option price (full-precision, 200-step tree)
        nodes        : list of {step, node_j, stock, option, early_ex}
        u, d, p, dt  : tree parameters
        n_steps      : steps used for visualisation
    """
    n_viz = min(n_steps, 8)

    dt = T / n_viz
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    # Build stock and option trees (2D arrays indexed [step][j])
    stock_tree = [[0.0] * (i + 1) for i in range(n_viz + 1)]
    option_tree = [[0.0] * (i + 1) for i in range(n_viz + 1)]

    for i in range(n_viz + 1):
        for j in range(i + 1):
            stock_tree[i][j] = S * (u ** j) * (d ** (i - j))

    # Terminal payoffs
    for j in range(n_viz + 1):
        S_T = stock_tree[n_viz][j]
        option_tree[n_viz][j] = (
            max(S_T - K, 0.0) if option_type == "call" else max(K - S_T, 0.0)
        )

    early_exercise = [[False] * (i + 1) for i in range(n_viz + 1)]

    # Backward induction
    for i in range(n_viz - 1, -1, -1):
        for j in range(i + 1):
            hold = discount * (
                p * option_tree[i + 1][j + 1] + (1 - p) * option_tree[i + 1][j]
            )
            S_node = stock_tree[i][j]
            intrinsic = (
                max(S_node - K, 0.0) if option_type == "call" else max(K - S_node, 0.0)
            )
            if american and intrinsic > hold:
                option_tree[i][j] = intrinsic
                early_exercise[i][j] = True
            else:
                option_tree[i][j] = hold

    # Build flat node list for Plotly
    nodes = []
    for i in range(n_viz + 1):
        for j in range(i + 1):
            nodes.append(
                {
                    "step": i,
                    "node_j": j,
                    "stock": round(stock_tree[i][j], 4),
                    "option": round(option_tree[i][j], 4),
                    "early_ex": early_exercise[i][j],
                }
            )

    # High-precision price from a 200-step tree
    price = binomial_price(S, K, T, r, sigma, 200, option_type, american, q)

    return {
        "price": price,
        "nodes": nodes,
        "u": u,
        "d": d,
        "p": p,
        "dt": dt,
        "n_steps": n_viz,
    }
