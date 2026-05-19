"""
app.py — Black-Scholes Option Pricing Calculator (Premium Edition)

Six-tab Streamlit application:
  1. Pricer         — BS price, all Greeks, IV, payoff, heatmap, binomial tree
  2. Live Market    — Ticker lookup, auto-fill, historical charts
  3. Monte Carlo    — GBM simulation, distribution, price comparison
  4. Strategy Builder — Multi-leg P&L, heatmap, scenario analysis
  5. Options Chain  — Live chain vs BS model, IV smile
  6. Vol Surface    — 3D IV surface, term structure, vol smile
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import brentq

# ── Core modules ─────────────────────────────────────────────────────────────
from black_scholes import (
    call_price, put_price,
    prob_itm_call, prob_itm_put, breakeven_call, breakeven_put,
)
from greeks import delta, gamma, theta, vega, rho
from advanced_greeks import charm, vanna, vomma, speed, color, ultima
from monte_carlo import mc_convergence
from stochastic_models import (
    GBMParams, HestonParams, JumpDiffusionParams,
    simulate,
)
from options_chain import enrich_chain, vol_smile
from strategy_builder import (
    PREDEFINED_STRATEGIES, get_strategy_legs,
    strategy_pnl, strategy_pnl_now, pnl_heatmap, strategy_summary,
)
from market_data import (
    get_stock_info, get_historical_volatility, get_risk_free_rate,
    get_options_expirations, get_options_chain, get_price_history,
    format_market_cap, days_to_expiry,
)
from volatility_surface import (
    build_surface_from_chain, build_synthetic_surface, vol_term_structure,
)
from binomial_tree import binomial_price, tree_for_viz

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BS Option Pricer Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Tighten metric cards */
[data-testid="metric-container"] {
    background: var(--secondary-background-color);
    border: 1px solid var(--border-color, #2d3748);
    border-radius: 10px;
    padding: 12px 16px;
}
/* Tab label styling */
button[data-baseweb="tab"] {
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
/* Table headers */
thead tr th { font-weight: 700; }
/* Smooth colour transitions */
* { transition: background-color 0.2s ease; }
/* Tooltip-style info boxes */
.info-box {
    background: rgba(0, 212, 170, 0.08);
    border-left: 3px solid #00d4aa;
    border-radius: 4px;
    padding: 8px 12px;
    font-size: 0.85rem;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)


# ── Session-state defaults ────────────────────────────────────────────────────
_defaults = {
    "S": 100.0, "K": 100.0, "T": 1.0, "r": 0.053, "sigma": 0.20, "q": 0.0,
    "ticker": "", "live_fetched": False, "live_info": {},
    "mc_result": None, "mc_sim_result": None,
    "mc_ticker": "", "mc_live_info": {}, "mc_live_fetched": False,
    "chain_calls": None, "chain_puts": None,
    "chain_ticker": "", "chain_expiry": "", "vol_surface_df": None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 BS Option Pricer")
    st.markdown("---")
    st.markdown("### Parameters")

    S = st.number_input(
        "Stock Price  S ($)", min_value=0.01, max_value=100_000.0,
        value=float(st.session_state.S), step=1.0, format="%.2f", key="S_input",
    )
    K = st.number_input(
        "Strike Price  K ($)", min_value=0.01, max_value=100_000.0,
        value=float(st.session_state.K), step=1.0, format="%.2f", key="K_input",
    )
    T = st.slider(
        "Time to Expiry (years)", 0.01, 3.0,
        value=float(st.session_state.T), step=0.01, key="T_input",
    )
    r = st.slider(
        "Risk-Free Rate (%)", 0.0, 15.0,
        value=float(st.session_state.r * 100), step=0.05, key="r_input",
    ) / 100.0
    sigma = st.slider(
        "Volatility σ (%)", 1.0, 150.0,
        value=float(st.session_state.sigma * 100), step=0.5, key="sigma_input",
    ) / 100.0
    q = st.slider(
        "Dividend Yield q (%)", 0.0, 10.0,
        value=float(st.session_state.q * 100), step=0.05, key="q_input",
        help="Continuous dividend yield (Merton extension). Set to 0 for non-dividend stocks.",
    ) / 100.0

    st.markdown("---")
    option_type = st.radio("Option Type", ["Call", "Put", "Both"], horizontal=True)

    st.markdown("---")
    st.caption("Prices and Greeks update live.")
    st.caption("Educational purposes only.")

# persist to session_state
st.session_state.S = S
st.session_state.K = K
st.session_state.T = T
st.session_state.r = r
st.session_state.sigma = sigma
st.session_state.q = q


# ── Cached data-fetching helpers ──────────────────────────────────────────────
@st.cache_data(ttl=300)
def _stock_info(ticker: str) -> dict:
    return get_stock_info(ticker)

@st.cache_data(ttl=300)
def _hist_vol(ticker: str, window: int = 30) -> float | None:
    return get_historical_volatility(ticker, window)

@st.cache_data(ttl=300)
def _rfr() -> float:
    return get_risk_free_rate()

@st.cache_data(ttl=300)
def _expiries(ticker: str) -> list:
    return get_options_expirations(ticker)

@st.cache_data(ttl=300)
def _chain(ticker: str, expiry: str):
    return get_options_chain(ticker, expiry)

@st.cache_data(ttl=300)
def _price_hist(ticker: str, period: str = "1y"):
    return get_price_history(ticker, period)

@st.cache_data(ttl=120)
def _run_simulate(model_str, style_str, otype, n_paths, n_display,
                  seed, barrier, btype, bdir, dig_payout,
                  mc_S, mc_K, mc_T, mc_r2, mc_sigma, mc_q2,
                  heston_kappa, heston_xi, heston_rho,
                  jump_lam, jump_mu_j, jump_sig_j):
    model_key = model_str.lower().replace("-", "").replace(" ", "")
    style_key = style_str.lower()

    if model_key == "gbm":
        params = GBMParams(S=mc_S, K=mc_K, T=mc_T, r=mc_r2, sigma=mc_sigma, q=mc_q2)
    elif model_key == "heston":
        params = HestonParams(
            S=mc_S, K=mc_K, T=mc_T, r=mc_r2, q=mc_q2,
            v0=mc_sigma**2, kappa=heston_kappa,
            theta=mc_sigma**2, xi=heston_xi, rho=heston_rho,
        )
    else:  # jumpdiffusion
        params = JumpDiffusionParams(
            S=mc_S, K=mc_K, T=mc_T, r=mc_r2, sigma=mc_sigma, q=mc_q2,
            lam=jump_lam, mu_j=jump_mu_j, sigma_j=jump_sig_j,
        )

    return simulate(
        model=model_key if model_key != "jumpdiffusion" else "jump",
        model_params=params,
        option_style=style_key,
        option_type=otype,
        n_paths=n_paths,
        n_steps=252,
        seed=seed,
        barrier=barrier,
        barrier_type=btype,
        barrier_direction=bdir,
        digital_payout=dig_payout,
        n_display_paths=n_display,
    )


# ── Shared Plotly layout ──────────────────────────────────────────────────────
_layout = dict(
    margin=dict(l=20, r=20, t=40, b=20),
    hovermode="x unified",
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)


# ── Helper: Greek tooltip ─────────────────────────────────────────────────────
_GREEK_HELP = {
    "Delta":        "Rate of change of option price per $1 move in stock.",
    "Gamma":        "Rate of change of delta per $1 move in stock.",
    "Theta":        "Option price decay per calendar day (time erosion).",
    "Vega":         "Price change per 1% increase in volatility.",
    "Rho":          "Price change per 1% increase in risk-free rate.",
    "Charm":        "Rate of change of delta per calendar day (delta decay).",
    "Vanna":        "How delta changes when volatility changes.",
    "Vomma":        "Rate of change of vega with respect to volatility.",
    "Speed":        "Rate of change of gamma with respect to stock price.",
    "Color":        "Rate of change of gamma per calendar day.",
    "Ultima":       "Third derivative w.r.t. volatility.",
}


# ─────────────────────────────────────────────────────────────────────────────
# TAB DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
tab_pricer, tab_market, tab_mc, tab_strategy, tab_chain, tab_surface = st.tabs([
    "📊 Pricer",
    "🌐 Live Market",
    "🎲 Monte Carlo",
    "⚙️ Strategy Builder",
    "🔗 Options Chain",
    "📈 Vol Surface",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PRICER
# ══════════════════════════════════════════════════════════════════════════════
with tab_pricer:
    st.markdown("### Black-Scholes Option Prices")

    c_price = call_price(S, K, T, r, sigma, q)
    p_price = put_price(S, K, T, r, sigma, q)
    moneyness = "ATM" if abs(S - K) / K < 0.01 else ("ITM" if S > K else "OTM")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Call Price", f"${c_price:.4f}")
    with col2:
        st.metric("Put Price", f"${p_price:.4f}")
    with col3:
        parity_diff = c_price - p_price - (S * np.exp(-q * T) - K * np.exp(-r * T))
        st.metric("Put-Call Parity Δ", f"${parity_diff:.6f}",
                  help="Should be ~0 for European options. Measures model consistency.")
    with col4:
        st.metric("Moneyness", moneyness,
                  delta=f"S/K = {S/K:.3f}")

    # ── Probability & Risk Metrics ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Probability & Risk Metrics")

    prob_call_itm = prob_itm_call(S, K, T, r, sigma, q)
    prob_put_itm  = prob_itm_put(S, K, T, r, sigma, q)
    be_call = breakeven_call(K, c_price)
    be_put  = breakeven_put(K, p_price)

    pm1, pm2, pm3, pm4, pm5, pm6 = st.columns(6)
    with pm1:
        st.metric("P(Call ITM)", f"{prob_call_itm*100:.1f}%",
                  help="Risk-neutral probability the call expires in the money (N(d2)).")
    with pm2:
        st.metric("P(Put ITM)",  f"{prob_put_itm*100:.1f}%",
                  help="Risk-neutral probability the put expires in the money (N(-d2)).")
    with pm3:
        st.metric("Call Breakeven", f"${be_call:.2f}",
                  help="Stock price at expiry where long call breaks even.")
    with pm4:
        st.metric("Put Breakeven",  f"${be_put:.2f}",
                  help="Stock price at expiry where long put breaks even.")
    with pm5:
        st.metric("Max Loss (Call)", f"-${c_price:.4f}",
                  help="Maximum possible loss on a long call = premium paid.")
    with pm6:
        st.metric("Max Loss (Put)",  f"-${p_price:.4f}",
                  help="Maximum possible loss on a long put = premium paid.")

    st.markdown("---")

    # ── Greeks ────────────────────────────────────────────────────────────────
    st.markdown("#### First-Order Greeks")

    g_cols = st.columns(5)
    greeks_vals = {
        "Delta": (delta(S, K, T, r, sigma, "call", q), delta(S, K, T, r, sigma, "put", q)),
        "Gamma": (gamma(S, K, T, r, sigma, q),) * 2,
        "Theta": (theta(S, K, T, r, sigma, "call", q), theta(S, K, T, r, sigma, "put", q)),
        "Vega":  (vega(S, K, T, r, sigma, q),) * 2,
        "Rho":   (rho(S, K, T, r, sigma, "call", q), rho(S, K, T, r, sigma, "put", q)),
    }
    for col, (name, (cv, pv)) in zip(g_cols, greeks_vals.items()):
        with col:
            st.metric(f"{name} (Call)", f"{cv:.4f}", help=_GREEK_HELP[name])
            st.metric(f"{name} (Put)",  f"{pv:.4f}")

    # ── Advanced Greeks ────────────────────────────────────────────────────────
    with st.expander("Second & Third-Order Greeks", expanded=False):
        adv_names = ["Charm", "Vanna", "Vomma", "Speed", "Color", "Ultima"]
        adv_fns   = [
            lambda: charm(S, K, T, r, sigma, "call"),
            lambda: vanna(S, K, T, r, sigma),
            lambda: vomma(S, K, T, r, sigma),
            lambda: speed(S, K, T, r, sigma),
            lambda: color(S, K, T, r, sigma),
            lambda: ultima(S, K, T, r, sigma),
        ]
        adv_cols = st.columns(6)
        for col, name, fn in zip(adv_cols, adv_names, adv_fns):
            with col:
                val = fn()
                st.metric(name, f"{val:.6f}", help=_GREEK_HELP[name])

    st.markdown("---")

    # ── Implied Volatility ────────────────────────────────────────────────────
    st.markdown("#### Implied Volatility Calculator")
    iv_col1, iv_col2, iv_col3 = st.columns([2, 1, 3])
    with iv_col1:
        market_price_input = st.number_input(
            "Market Option Price ($)", min_value=0.001,
            value=round(c_price, 2), step=0.01,
        )
    with iv_col2:
        iv_type = st.radio("Type", ["call", "put"], key="iv_type_radio")
    with iv_col3:
        pricer_fn = call_price if iv_type == "call" else put_price
        try:
            f_lo = pricer_fn(S, K, T, r, 1e-4, q) - market_price_input
            f_hi = pricer_fn(S, K, T, r, 10.0, q)  - market_price_input
            if f_lo * f_hi > 0:
                raise ValueError("No solution")
            iv_result = brentq(
                lambda v: pricer_fn(S, K, T, r, v, q) - market_price_input,
                1e-4, 10.0, maxiter=200,
            )
            st.success(f"Implied Volatility: **{iv_result*100:.2f}%**  "
                       f"({'above' if iv_result > sigma else 'below'} σ input of {sigma*100:.1f}%)")
        except Exception:
            st.error("Cannot compute IV — check that market price is within no-arbitrage bounds.")

    st.markdown("---")

    # ── Payoff at Expiration ──────────────────────────────────────────────────
    st.markdown("#### Payoff at Expiration")
    spot_range = np.linspace(max(1.0, S * 0.50), S * 1.50, 400)
    fig_pnl = go.Figure()
    if option_type in ("Call", "Both"):
        call_pnl = np.maximum(spot_range - K, 0.0) - c_price
        fig_pnl.add_trace(go.Scatter(
            x=spot_range, y=call_pnl, name="Call P&L",
            line=dict(color="#00d4aa", width=2.5),
            fill="tozeroy", fillcolor="rgba(0,212,170,0.08)",
        ))
    if option_type in ("Put", "Both"):
        put_pnl = np.maximum(K - spot_range, 0.0) - p_price
        fig_pnl.add_trace(go.Scatter(
            x=spot_range, y=put_pnl, name="Put P&L",
            line=dict(color="#ff6b6b", width=2.5),
            fill="tozeroy", fillcolor="rgba(255,107,107,0.08)",
        ))
    fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_pnl.add_vline(x=K, line_dash="dot", line_color="#ffd700",
                      annotation_text=f"Strike ${K:.0f}", annotation_font_color="#ffd700")
    fig_pnl.add_vline(x=S, line_dash="dash", line_color="#a0a0a0",
                      annotation_text=f"Spot ${S:.0f}", annotation_font_color="#a0a0a0")
    fig_pnl.update_layout(
        **_layout,
        title="P&L at Expiration",
        xaxis_title="Stock Price at Expiry ($)",
        yaxis_title="Profit / Loss ($)",
        height=400,
    )
    st.plotly_chart(fig_pnl, use_container_width=True)

    # ── P&L Scenario Heatmap ──────────────────────────────────────────────────
    st.markdown("#### P&L Scenario Heatmap — Price × Time")
    st.markdown('<div class="info-box">Shows how a single-option P&L changes across stock prices '
                'AND time remaining. Green = profit, red = loss.</div>', unsafe_allow_html=True)

    hm_otype = st.radio("Option for Heatmap", ["call", "put"], horizontal=True, key="hm_otype")
    entry_price_hm = c_price if hm_otype == "call" else p_price
    spot_hm = np.linspace(S * 0.60, S * 1.40, 60)
    days_hm = np.linspace(T * 365, 0, 30)

    pnl_mat = np.zeros((len(days_hm), len(spot_hm)))
    pricer_hm = call_price if hm_otype == "call" else put_price
    for i, d_rem in enumerate(days_hm):
        t_rem = d_rem / 365.0
        for j, s in enumerate(spot_hm):
            if t_rem <= 0:
                cv = max(s - K, 0) if hm_otype == "call" else max(K - s, 0)
            else:
                cv = pricer_hm(s, K, t_rem, r, sigma, q)
            pnl_mat[i, j] = cv - entry_price_hm

    fig_hm = go.Figure(go.Heatmap(
        z=pnl_mat,
        x=np.round(spot_hm, 2),
        y=np.round(days_hm, 1),
        colorscale="RdYlGn",
        zmid=0,
        colorbar=dict(title="P&L ($)"),
        hovertemplate="Spot: $%{x:.2f}<br>Days left: %{y:.0f}<br>P&L: $%{z:.4f}<extra></extra>",
    ))
    fig_hm.add_vline(x=S, line_dash="dash", line_color="white", opacity=0.6,
                     annotation_text="Current Spot")
    fig_hm.add_hline(y=T * 365, line_dash="dash", line_color="white", opacity=0.6,
                     annotation_text="Today")
    fig_hm.update_layout(
        **{**_layout, "hovermode": "closest"},
        title=f"P&L Heatmap — Long {hm_otype.capitalize()} (Entry: ${entry_price_hm:.4f})",
        xaxis_title="Stock Price ($)",
        yaxis_title="Days to Expiry",
        height=420,
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    # ── 3-D Option Price Surface ──────────────────────────────────────────────
    st.markdown("#### Option Price Surface (Call)")
    vol_ax  = np.linspace(0.05, 0.80, 35)
    spot_ax = np.linspace(S * 0.60, S * 1.40, 35)
    Z = np.array([[call_price(s, K, T, r, v, q) for v in vol_ax] for s in spot_ax])
    fig3d = go.Figure(go.Surface(
        z=Z, x=vol_ax * 100, y=spot_ax,
        colorscale="Viridis",
        colorbar=dict(title="Call Price ($)"),
        hovertemplate="Vol: %{x:.1f}%<br>Spot: $%{y:.2f}<br>Price: $%{z:.4f}<extra></extra>",
    ))
    fig3d.update_layout(
        title="Call Price Surface — Spot vs Volatility",
        scene=dict(
            xaxis_title="Volatility (%)",
            yaxis_title="Stock Price ($)",
            zaxis_title="Call Price ($)",
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=40, b=0),
        height=500,
    )
    st.plotly_chart(fig3d, use_container_width=True)

    # ── Binomial Tree ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Binomial Tree — American vs European")
    bt_col1, bt_col2, bt_col3 = st.columns(3)
    with bt_col1:
        bt_type = st.radio("Option Type", ["call", "put"], key="bt_type", horizontal=True)
    with bt_col2:
        bt_steps = st.slider("Visualisation Steps", 3, 8, 6, key="bt_steps")
    with bt_col3:
        bt_200_eu = binomial_price(S, K, T, r, sigma, 200, bt_type, american=False)
        bt_200_am = binomial_price(S, K, T, r, sigma, 200, bt_type, american=True)
        bs_bench  = call_price(S, K, T, r, sigma) if bt_type == "call" else put_price(S, K, T, r, sigma)
        st.metric("European (Binomial)", f"${bt_200_eu:.4f}")
        st.metric("American (Binomial)", f"${bt_200_am:.4f}")
        st.metric("Black-Scholes",       f"${bs_bench:.4f}")

    tree_data = tree_for_viz(S, K, T, r, sigma, bt_steps, bt_type, american=False)
    tree_am   = tree_for_viz(S, K, T, r, sigma, bt_steps, bt_type, american=True)

    show_am = st.checkbox("Show American early-exercise nodes", value=True)
    chosen_tree = tree_am if show_am else tree_data

    # Build Plotly scatter from nodes
    node_x, node_y, node_text, node_color = [], [], [], []
    edge_x, edge_y = [], []
    n = chosen_tree["n_steps"]

    # Map (step, j) → position: x = step, y = j - step/2
    def _node_pos(step, j):
        return step, j - step / 2.0

    nodes_map = {(nd["step"], nd["node_j"]): nd for nd in chosen_tree["nodes"]}

    for nd in chosen_tree["nodes"]:
        xi, yi = _node_pos(nd["step"], nd["node_j"])
        node_x.append(xi)
        node_y.append(yi)
        node_text.append(
            f"Step {nd['step']}<br>Stock: ${nd['stock']:.2f}<br>"
            f"Option: ${nd['option']:.4f}"
            + ("<br><b>Early Exercise</b>" if nd.get("early_ex") else "")
        )
        node_color.append("#ff4b4b" if nd.get("early_ex") else "#00d4aa")

        # Add edges to children
        step_i, j_i = nd["step"], nd["node_j"]
        if step_i < n:
            for dj in [0, 1]:
                child = nodes_map.get((step_i + 1, j_i + dj))
                if child:
                    cx, cy = _node_pos(child["step"], child["node_j"])
                    edge_x += [xi, cx, None]
                    edge_y += [yi, cy, None]

    fig_tree = go.Figure()
    fig_tree.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color="rgba(150,150,150,0.4)", width=1),
        hoverinfo="none", showlegend=False,
    ))
    fig_tree.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(size=16, color=node_color, line=dict(width=1, color="white")),
        text=[f"${nd['option']:.2f}" for nd in chosen_tree["nodes"]],
        textposition="top center",
        textfont=dict(size=8),
        hovertext=node_text,
        hoverinfo="text",
        showlegend=False,
    ))
    fig_tree.update_layout(
        **_layout,
        title=f"{'American' if show_am else 'European'} {bt_type.capitalize()} Tree "
              f"(u={chosen_tree['u']:.4f}, d={chosen_tree['d']:.4f}, p={chosen_tree['p']:.4f})",
        xaxis=dict(title="Time Step", showgrid=False),
        yaxis=dict(title="", showgrid=False, showticklabels=False),
        height=420,
    )
    st.plotly_chart(fig_tree, use_container_width=True)
    if show_am and bt_200_am > bt_200_eu:
        diff = bt_200_am - bt_200_eu
        st.info(f"Early-exercise premium: ${diff:.4f}  "
                f"({diff/bt_200_eu*100:.2f}% above European price)")

    # ── Greeks Profiles ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Greeks Profiles — vs Stock Price")
    st.markdown('<div class="info-box">Shows how each Greek changes as the underlying '
                'stock price moves. Vertical dashed line = current spot.</div>',
                unsafe_allow_html=True)

    gp_otype = st.radio("Option for Greeks Profiles", ["call", "put"],
                        horizontal=True, key="gp_otype")

    s_range = np.linspace(max(1.0, S * 0.50), S * 1.50, 300)
    gp_delta  = [delta(s, K, T, r, sigma, gp_otype, q) for s in s_range]
    gp_gamma  = [gamma(s, K, T, r, sigma, q)            for s in s_range]
    gp_theta  = [theta(s, K, T, r, sigma, gp_otype, q)  for s in s_range]
    gp_vega   = [vega(s, K, T, r, sigma, q)              for s in s_range]

    fig_gp = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Delta", "Gamma", "Theta (per day)", "Vega (per 1% vol)"),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )
    _gp_data = [
        (gp_delta, "#00d4aa", 1, 1),
        (gp_gamma, "#ffd700", 1, 2),
        (gp_theta, "#ff6b6b", 2, 1),
        (gp_vega,  "#a29bfe", 2, 2),
    ]
    for vals, clr, row, col in _gp_data:
        fig_gp.add_trace(go.Scatter(
            x=s_range, y=vals,
            line=dict(color=clr, width=2.0),
            showlegend=False,
            hovertemplate="Spot: $%{x:.2f}<br>Value: %{y:.5f}<extra></extra>",
        ), row=row, col=col)
        fig_gp.add_vline(x=S, line_dash="dash", line_color="white",
                         opacity=0.45, row=row, col=col)
        fig_gp.add_vline(x=K, line_dash="dot", line_color="#ffd700",
                         opacity=0.35, row=row, col=col)

    fig_gp.update_layout(
        **_layout,
        height=540,
        title=f"Greeks Profiles — Long {gp_otype.capitalize()}  (K=${K:.0f}, T={T:.2f}y, σ={sigma*100:.1f}%)",
    )
    st.plotly_chart(fig_gp, use_container_width=True)

    # ── Comparative Strike Analysis ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Comparative Strike Analysis")
    st.markdown('<div class="info-box">Prices and Greeks for strikes around the current spot price. '
                'Green = in the money, red = out of the money.</div>',
                unsafe_allow_html=True)

    strike_spacing = st.number_input(
        "Strike spacing ($)", min_value=0.5, value=round(S * 0.05, 1), step=0.5,
        key="cs_spacing", help="Dollar distance between each strike in the table.",
    )
    cs_otype = st.radio("Show", ["call", "put", "both"], horizontal=True, key="cs_otype")

    strikes_cs = [K + (i - 4) * strike_spacing for i in range(9)]
    rows_cs = []
    for k_cs in strikes_cs:
        if k_cs <= 0:
            continue
        cp = call_price(S, k_cs, T, r, sigma, q)
        pp = put_price(S, k_cs, T, r, sigma, q)
        itm_call = S > k_cs
        itm_put  = S < k_cs
        row = {
            "Strike": f"${k_cs:.2f}",
            "Moneyness": "ITM" if itm_call else ("ATM" if abs(S - k_cs) / k_cs < 0.01 else "OTM"),
        }
        if cs_otype in ("call", "both"):
            row["Call Price"]  = f"${cp:.4f}"
            row["Call Δ"]      = f"{delta(S, k_cs, T, r, sigma, 'call', q):.4f}"
            row["Call Θ/day"]  = f"{theta(S, k_cs, T, r, sigma, 'call', q):.4f}"
        if cs_otype in ("put", "both"):
            row["Put Price"]   = f"${pp:.4f}"
            row["Put Δ"]       = f"{delta(S, k_cs, T, r, sigma, 'put', q):.4f}"
            row["Put Θ/day"]   = f"{theta(S, k_cs, T, r, sigma, 'put', q):.4f}"
        row["Gamma"]  = f"{gamma(S, k_cs, T, r, sigma, q):.5f}"
        row["Vega"]   = f"{vega(S, k_cs, T, r, sigma, q):.4f}"
        rows_cs.append(row)

    df_cs = pd.DataFrame(rows_cs)

    def _color_moneyness(val):
        if val == "ITM":
            return "background-color: rgba(0,212,170,0.15); color: #00d4aa; font-weight:600"
        if val == "OTM":
            return "background-color: rgba(255,107,107,0.10); color: #ff6b6b"
        return "font-weight:700; color:#ffd700"

    styled_cs = df_cs.style.map(_color_moneyness, subset=["Moneyness"])
    st.dataframe(styled_cs, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE MARKET
# ══════════════════════════════════════════════════════════════════════════════
with tab_market:
    st.markdown("### Live Ticker Lookup")
    st.markdown("Enter a ticker to auto-fill parameters and see live market data.")

    lm_col1, lm_col2 = st.columns([3, 1])
    with lm_col1:
        ticker_input = st.text_input(
            "Ticker Symbol", value=st.session_state.ticker,
            placeholder="e.g. AAPL, SPY, TSLA",
        ).upper().strip()
    with lm_col2:
        fetch_btn = st.button("Fetch Data", type="primary", use_container_width=True)

    if fetch_btn and ticker_input:
        with st.spinner(f"Fetching {ticker_input}…"):
            info = _stock_info(ticker_input)
            hv   = _hist_vol(ticker_input)
            rfr  = _rfr()
        st.session_state.ticker        = ticker_input
        st.session_state.live_info     = info
        st.session_state.live_fetched  = True
        if hv:
            st.session_state.sigma = hv
        st.session_state.r = rfr
        if info.get("success") and info.get("price"):
            st.session_state.S = float(info["price"])
            st.session_state.K = float(info["price"])

    info = st.session_state.get("live_info", {})

    if info.get("success"):
        st.markdown("---")
        # Company header
        st.markdown(f"## {info['name']}  `{info['ticker']}`")
        st.markdown(f"**{info.get('sector', 'N/A')}** · {info.get('industry', 'N/A')} · {info.get('exchange', 'N/A')}")

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Current Price", f"${info['price']:.2f}" if info.get('price') else "N/A")
        with m2:
            st.metric("52-Wk High", f"${info.get('52_week_high', 0):.2f}" if info.get('52_week_high') else "N/A")
        with m3:
            st.metric("52-Wk Low",  f"${info.get('52_week_low', 0):.2f}"  if info.get('52_week_low')  else "N/A")
        with m4:
            st.metric("Market Cap", format_market_cap(info.get("market_cap")))
        with m5:
            hv_val = st.session_state.sigma
            st.metric("30-Day HV", f"{hv_val*100:.1f}%" if hv_val else "N/A")

        extra1, extra2, extra3 = st.columns(3)
        with extra1:
            st.metric("Beta",      f"{info.get('beta', 'N/A'):.2f}" if info.get('beta') else "N/A")
        with extra2:
            st.metric("P/E Ratio", f"{info.get('pe_ratio', 'N/A'):.1f}" if info.get('pe_ratio') else "N/A")
        with extra3:
            dy = info.get("dividend_yield")
            st.metric("Div. Yield", f"{dy*100:.2f}%" if dy else "0.00%")

        # Auto-fill button
        st.markdown("---")
        if st.button("Auto-fill Parameters from Live Data", type="secondary"):
            if info.get("price"):
                st.session_state.S = float(info["price"])
                st.session_state.K = float(info["price"])
            if st.session_state.sigma:
                pass  # already set on fetch
            st.rerun()

        # Historical price chart
        st.markdown("#### Historical Price")
        period_sel = st.select_slider("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], value="1y")
        hist_df = _price_hist(info["ticker"], period_sel)

        if not hist_df.empty:
            fig_price = go.Figure()
            fig_price.add_trace(go.Candlestick(
                x=hist_df.index,
                open=hist_df["Open"],
                high=hist_df["High"],
                low=hist_df["Low"],
                close=hist_df["Close"],
                name="Price",
                increasing_line_color="#00d4aa",
                decreasing_line_color="#ff6b6b",
            ))
            fig_price.update_layout(
                **_layout,
                title=f"{info['ticker']} — {period_sel} Price History",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=420,
                xaxis_rangeslider_visible=False,
            )
            st.plotly_chart(fig_price, use_container_width=True)

            # Rolling historical volatility chart
            st.markdown("#### Rolling Annualised Volatility")
            closes = hist_df["Close"].dropna()
            log_ret = np.log(closes / closes.shift(1)).dropna()
            fig_rv = go.Figure()
            for win, col in [(10, "#ff6b6b"), (20, "#ffd700"), (30, "#00d4aa"), (60, "#a29bfe")]:
                if len(log_ret) > win:
                    rv = log_ret.rolling(win).std() * np.sqrt(252) * 100
                    fig_rv.add_trace(go.Scatter(
                        x=rv.index, y=rv, name=f"{win}-day HV",
                        line=dict(color=col, width=1.8),
                    ))
            fig_rv.add_hline(y=st.session_state.sigma * 100,
                             line_dash="dash", line_color="white",
                             annotation_text="Current σ input",
                             annotation_font_color="white")
            fig_rv.update_layout(
                **_layout,
                title="Rolling Historical Volatility (%)",
                xaxis_title="Date", yaxis_title="Annualised Vol (%)",
                height=360,
            )
            st.plotly_chart(fig_rv, use_container_width=True)

    elif st.session_state.live_fetched:
        st.error(f"Could not fetch data for '{ticker_input}'. Check the ticker and try again.")
    else:
        st.info("Enter a ticker symbol above and click 'Fetch Data'.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MONTE CARLO
# ══════════════════════════════════════════════════════════════════════════════
with tab_mc:
    st.markdown("### Monte Carlo Simulation Engine")

    # ── Live ticker lookup inside MC tab ──────────────────────────────────────
    st.markdown("#### Stock Lookup")
    st.markdown('<div class="info-box">Type any ticker to auto-fetch live price, '
                'volatility, and dividend yield — then run the simulation.</div>',
                unsafe_allow_html=True)

    mc_ticker_col, mc_fetch_col = st.columns([4, 1])
    with mc_ticker_col:
        mc_ticker_in = st.text_input(
            "Ticker Symbol", value=st.session_state.mc_ticker,
            placeholder="e.g. AAPL, TSLA, SPY, QQQ",
            key="mc_ticker_input",
        ).upper().strip()
    with mc_fetch_col:
        mc_fetch_btn = st.button("Fetch", type="primary", use_container_width=True, key="mc_fetch")

    if mc_fetch_btn and mc_ticker_in:
        with st.spinner(f"Fetching {mc_ticker_in}…"):
            _mc_info = _stock_info(mc_ticker_in)
            _mc_hv   = _hist_vol(mc_ticker_in, 30)
            _mc_rfr  = _rfr()
        st.session_state.mc_ticker     = mc_ticker_in
        st.session_state.mc_live_info  = _mc_info
        st.session_state.mc_live_fetched = True
        if _mc_hv:
            st.session_state["mc_sigma_live"] = _mc_hv
        st.session_state["mc_r_live"] = _mc_rfr
        if _mc_info.get("success") and _mc_info.get("price"):
            st.session_state["mc_S_live"] = float(_mc_info["price"])
            dy = _mc_info.get("dividend_yield") or 0.0
            st.session_state["mc_q_live"] = float(dy)

    mc_info = st.session_state.get("mc_live_info", {})
    mc_live_ok = mc_info.get("success", False)

    if mc_live_ok:
        mi1, mi2, mi3, mi4, mi5 = st.columns(5)
        mi1.metric("Company", mc_info.get("name", "N/A")[:20])
        mi2.metric("Live Price", f"${mc_info.get('price', 0):.2f}")
        mi3.metric("30d HV", f"{st.session_state.get('mc_sigma_live', sigma)*100:.1f}%")
        mi4.metric("Div Yield", f"{st.session_state.get('mc_q_live', 0)*100:.2f}%")
        mi5.metric("Risk-Free", f"{st.session_state.get('mc_r_live', r)*100:.2f}%")
        st.markdown("---")

    # ── Model & option parameters ─────────────────────────────────────────────
    st.markdown("#### Simulation Parameters")

    # Resolve live or sidebar params
    _mc_S     = st.session_state.get("mc_S_live", S)     if mc_live_ok else S
    _mc_sigma = st.session_state.get("mc_sigma_live", sigma) if mc_live_ok else sigma
    _mc_r     = st.session_state.get("mc_r_live", r)     if mc_live_ok else r
    _mc_q     = st.session_state.get("mc_q_live", q)     if mc_live_ok else q

    sp1, sp2, sp3 = st.columns(3)
    with sp1:
        mc_S = st.number_input("Stock Price ($)", value=float(_mc_S),
                               min_value=0.01, step=1.0, format="%.2f", key="mc_S")
        mc_K = st.number_input("Strike ($)", value=float(round(_mc_S, 2)),
                               min_value=0.01, step=1.0, format="%.2f", key="mc_K")
    with sp2:
        mc_T = st.slider("Expiry (years)", 0.02, 3.0, float(T), 0.01, key="mc_T")
        mc_sigma = st.slider("Volatility σ (%)", 1.0, 150.0,
                             float(_mc_sigma * 100), 0.5, key="mc_sigma") / 100
    with sp3:
        mc_r2 = st.slider("Risk-free rate (%)", 0.0, 15.0,
                          float(_mc_r * 100), 0.05, key="mc_r2") / 100
        mc_q2 = st.slider("Div yield q (%)", 0.0, 10.0,
                          float(_mc_q * 100), 0.05, key="mc_q2") / 100

    st.markdown("---")

    # ── Model and option style selectors ─────────────────────────────────────
    mod1, mod2, mod3, mod4 = st.columns(4)
    with mod1:
        mc_model = st.selectbox("Process Model", ["GBM", "Heston", "Jump-Diffusion"], key="mc_model")
    with mod2:
        mc_style = st.selectbox("Option Style", ["European", "Asian", "Barrier", "Digital"], key="mc_style")
    with mod3:
        mc_otype = st.radio("Call / Put", ["call", "put"], horizontal=True, key="mc_otype")
    with mod4:
        mc_n_paths = st.select_slider("Paths", [10_000, 50_000, 100_000, 200_000], value=100_000, key="mc_npaths")

    # Barrier config
    mc_barrier = None
    mc_barrier_type = "knock-out"
    mc_barrier_dir  = "up"
    if mc_style == "Barrier":
        b1, b2, b3 = st.columns(3)
        with b1:
            mc_barrier = st.number_input("Barrier Level ($)", value=float(mc_S * 1.20),
                                         min_value=0.01, step=1.0, key="mc_barrier")
        with b2:
            mc_barrier_type = st.radio("Barrier Type", ["knock-out", "knock-in"],
                                       horizontal=True, key="mc_btype")
        with b3:
            mc_barrier_dir = st.radio("Direction", ["up", "down"],
                                      horizontal=True, key="mc_bdir")

    digital_payout = 1.0
    if mc_style == "Digital":
        digital_payout = st.number_input("Cash Payout ($)", value=1.0,
                                         min_value=0.01, step=0.1, key="mc_dig_payout")

    # Heston extra params
    mc_heston_kappa = 2.0
    mc_heston_xi    = 0.30
    mc_heston_rho   = -0.70
    if mc_model == "Heston":
        with st.expander("Heston Parameters (κ, ξ, ρ)", expanded=False):
            hc1, hc2, hc3 = st.columns(3)
            with hc1:
                mc_heston_kappa = st.slider("κ (mean-reversion)", 0.1, 10.0, 2.0, 0.1, key="heston_kappa")
            with hc2:
                mc_heston_xi = st.slider("ξ (vol-of-vol)", 0.05, 1.5, 0.30, 0.05, key="heston_xi")
            with hc3:
                mc_heston_rho = st.slider("ρ (correlation)", -0.99, 0.99, -0.70, 0.01, key="heston_rho")

    mc_jump_lam   = 5.0
    mc_jump_mu_j  = -0.02
    mc_jump_sig_j = 0.06
    if mc_model == "Jump-Diffusion":
        with st.expander("Jump Parameters (λ, μ_J, σ_J)", expanded=False):
            jc1, jc2, jc3 = st.columns(3)
            with jc1:
                mc_jump_lam  = st.slider("λ (jumps/year)", 0.5, 20.0, 5.0, 0.5, key="jump_lam")
            with jc2:
                mc_jump_mu_j = st.slider("μ_J (avg log jump)", -0.30, 0.20, -0.02, 0.01, key="jump_mu")
            with jc3:
                mc_jump_sig_j = st.slider("σ_J (jump vol)", 0.01, 0.50, 0.06, 0.01, key="jump_sig")

    mc_seed = st.number_input("Random Seed", value=42, step=1, key="mc_seed")
    mc_n_display = st.slider("Display Paths", 50, 300, 150, 25, key="mc_ndisplay")

    run_mc = st.button("Run Simulation", type="primary", use_container_width=True, key="run_mc_btn")

    if run_mc:
        with st.spinner(f"Running {mc_n_paths:,}-path {mc_model} simulation…"):
            sim = _run_simulate(
                mc_model, mc_style, mc_otype, mc_n_paths, mc_n_display,
                int(mc_seed), mc_barrier, mc_barrier_type, mc_barrier_dir, digital_payout,
                mc_S, mc_K, mc_T, mc_r2, mc_sigma, mc_q2,
                mc_heston_kappa, mc_heston_xi, mc_heston_rho,
                mc_jump_lam, mc_jump_mu_j, mc_jump_sig_j,
            )
            st.session_state.mc_sim_result = sim

    sim = st.session_state.get("mc_sim_result")

    if sim:
        st.markdown("---")
        st.markdown("#### Results")

        # Compare with BS if European GBM
        bs_ref = (call_price(mc_S, mc_K, mc_T, mc_r2, mc_sigma, mc_q2)
                  if mc_otype == "call"
                  else put_price(mc_S, mc_K, mc_T, mc_r2, mc_sigma, mc_q2))

        rm1, rm2, rm3, rm4, rm5 = st.columns(5)
        rm1.metric("MC Price", f"${sim['price']:.4f}")
        rm2.metric("BS Reference", f"${bs_ref:.4f}")
        diff_pct = (sim['price'] - bs_ref) / max(bs_ref, 1e-8) * 100
        rm3.metric("MC vs BS", f"{diff_pct:+.2f}%",
                   help="Difference vs Black-Scholes European. Non-zero for exotic styles or non-GBM models.")
        rm4.metric("Std Error", f"${sim['std_error']:.5f}")
        rm5.metric("P(ITM)", f"{sim['itm_fraction']*100:.1f}%")

        st.markdown(f"**95% CI:** [${sim['ci_low']:.4f} — ${sim['ci_high']:.4f}]  "
                    f"| **Paths:** {mc_n_paths:,} | **Model:** {mc_model} | **Style:** {mc_style}")

        # ── Path chart ────────────────────────────────────────────────────────
        paths  = sim["paths"]
        t_ax   = sim["time_axis"]
        fig_paths = go.Figure()

        # Individual paths (thin, semi-transparent)
        n_show = min(80, paths.shape[1])
        for i in range(n_show):
            fig_paths.add_trace(go.Scatter(
                x=t_ax, y=paths[:, i], mode="lines",
                line=dict(width=0.5, color="rgba(0,212,170,0.12)"),
                hoverinfo="skip", showlegend=False,
            ))

        # Percentile bands (filled regions)
        fig_paths.add_trace(go.Scatter(
            x=np.concatenate([t_ax, t_ax[::-1]]),
            y=np.concatenate([sim["pct95"], sim["pct5"][::-1]]),
            fill="toself", fillcolor="rgba(162,155,254,0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            name="5–95th pct", showlegend=True,
        ))
        fig_paths.add_trace(go.Scatter(
            x=np.concatenate([t_ax, t_ax[::-1]]),
            y=np.concatenate([sim["pct75"], sim["pct25"][::-1]]),
            fill="toself", fillcolor="rgba(162,155,254,0.20)",
            line=dict(color="rgba(0,0,0,0)"),
            name="25–75th pct", showlegend=True,
        ))
        # Mean path
        fig_paths.add_trace(go.Scatter(
            x=t_ax, y=sim["mean_path"], name="Mean",
            line=dict(color="#ffd700", width=2.5),
        ))
        fig_paths.add_hline(y=mc_K, line_dash="dot", line_color="#ff6b6b",
                            annotation_text=f"Strike ${mc_K:.0f}")
        if mc_barrier is not None:
            fig_paths.add_hline(y=mc_barrier, line_dash="dash",
                                line_color="#ffa500",
                                annotation_text=f"Barrier ${mc_barrier:.0f}")

        ticker_label = f" — {st.session_state.mc_ticker}" if st.session_state.mc_ticker else ""
        fig_paths.update_layout(
            **_layout,
            title=f"{mc_model} Paths{ticker_label} ({mc_style} {mc_otype})",
            xaxis_title="Time (years)", yaxis_title="Stock Price ($)",
            height=430, legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig_paths, use_container_width=True)

        # ── Terminal distribution ──────────────────────────────────────────────
        col_hist, col_stats = st.columns([3, 2])
        with col_hist:
            ST = sim["final_prices"]
            itm_mask = ST > mc_K if mc_otype == "call" else ST < mc_K
            fig_hist = go.Figure()
            bin_w = (ST.max() - ST.min()) / 70
            fig_hist.add_trace(go.Histogram(
                x=ST[~itm_mask], name="OTM",
                marker_color="rgba(255,107,107,0.65)",
                xbins=dict(size=bin_w), showlegend=True,
            ))
            fig_hist.add_trace(go.Histogram(
                x=ST[itm_mask], name="ITM",
                marker_color="rgba(0,212,170,0.75)",
                xbins=dict(size=bin_w), showlegend=True,
            ))
            fig_hist.add_vline(x=mc_K, line_dash="dot", line_color="#ffd700",
                               annotation_text=f"K=${mc_K:.0f}")
            fig_hist.add_vline(x=float(ST.mean()), line_dash="dash",
                               line_color="white",
                               annotation_text=f"E[S]=${ST.mean():.1f}")
            fig_hist.update_layout(
                **_layout, barmode="overlay",
                title=f"Terminal Distribution (ITM: {sim['itm_fraction']*100:.1f}%)",
                xaxis_title="Terminal Stock Price ($)", yaxis_title="Count",
                height=380,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_stats:
            st.markdown("**Distribution Stats**")
            st.markdown(f"Mean terminal price: **${ST.mean():.2f}**")
            st.markdown(f"Std dev: **${ST.std():.2f}**")
            st.markdown(f"Skewness: **{float(((ST-ST.mean())**3).mean()/ST.std()**3):.3f}**")
            st.markdown(f"Kurtosis: **{float(((ST-ST.mean())**4).mean()/ST.std()**4 - 3):.3f}**")
            st.markdown(f"5th pct: **${np.percentile(ST, 5):.2f}**")
            st.markdown(f"95th pct: **${np.percentile(ST, 95):.2f}**")
            st.markdown("---")
            st.markdown("**Option Stats**")
            st.markdown(f"MC Price: **${sim['price']:.4f}**")
            st.markdown(f"Std error: **${sim['std_error']:.5f}**")
            st.markdown(f"95% CI: **[${sim['ci_low']:.4f}, ${sim['ci_high']:.4f}]**")
            st.markdown(f"BS Reference: **${bs_ref:.4f}**")
            if mc_style == "European":
                st.markdown(f"MC vs BS: **{diff_pct:+.2f}%**")

        # ── Convergence (European GBM only) ───────────────────────────────────
        if mc_style == "European" and mc_model == "GBM":
            with st.expander("Convergence Analysis — Price vs Number of Paths"):
                with st.spinner("Computing convergence…"):
                    p_counts, p_prices, p_errs = mc_convergence(
                        mc_S, mc_K, mc_T, mc_r2, mc_sigma, mc_otype,
                        max_paths=50_000, seed=int(mc_seed),
                    )
                fig_conv = go.Figure()
                fig_conv.add_trace(go.Scatter(
                    x=p_counts, y=p_prices, name="MC Price",
                    line=dict(color="#00d4aa", width=2),
                    mode="lines+markers", marker_size=4,
                ))
                fig_conv.add_trace(go.Scatter(
                    x=p_counts,
                    y=p_prices + 1.96 * p_errs,
                    line=dict(color="rgba(0,212,170,0.3)", dash="dot"),
                    showlegend=False,
                ))
                fig_conv.add_trace(go.Scatter(
                    x=p_counts,
                    y=p_prices - 1.96 * p_errs,
                    fill="tonexty", fillcolor="rgba(0,212,170,0.08)",
                    line=dict(color="rgba(0,212,170,0.3)", dash="dot"),
                    showlegend=False, name="95% CI",
                ))
                fig_conv.add_hline(y=bs_ref, line_dash="dash", line_color="#ffd700",
                                   annotation_text=f"BS ${bs_ref:.4f}")
                fig_conv.update_layout(
                    **_layout,
                    title="MC Convergence — Price vs Path Count",
                    xaxis_title="Paths (log scale)", yaxis_title="Price ($)",
                    xaxis_type="log", height=360,
                )
                st.plotly_chart(fig_conv, use_container_width=True)
    else:
        st.info("Configure parameters above and click **Run Simulation**.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — STRATEGY BUILDER
# ══════════════════════════════════════════════════════════════════════════════
with tab_strategy:
    st.markdown("### Multi-Leg Strategy Builder")

    sb_c1, sb_c2 = st.columns([2, 1])
    with sb_c1:
        strategy_name = st.selectbox(
            "Strategy",
            list(PREDEFINED_STRATEGIES.keys()),
            index=list(PREDEFINED_STRATEGIES.keys()).index("Iron Condor"),
        )
    with sb_c2:
        use_atm = st.checkbox("Force ATM strikes (K = S)", value=True)
        K_sb = S if use_atm else K

    legs = get_strategy_legs(strategy_name, S, K_sb, T, r, sigma)

    # Leg table
    if legs:
        leg_rows = []
        for lg in legs:
            ltype = lg["type"]
            if ltype == "stock":
                desc = f"{'Long' if lg['action']=='long' else 'Short'} {int(lg['qty'])} shares @ ${lg['entry_price']:.2f}"
            else:
                desc = (f"{'Long' if lg['action']=='long' else 'Short'} "
                        f"{int(lg['qty'])}x {ltype.upper()} "
                        f"K=${lg['strike']:.2f}  "
                        f"T={lg['T']*365:.0f}d  "
                        f"Premium=${lg['entry_price']:.4f}")
            leg_rows.append(desc)

        st.markdown("**Legs:**")
        for row in leg_rows:
            st.markdown(f"- {row}")

        st.markdown("---")
        spot_range_sb = np.linspace(S * 0.50, S * 1.50, 400)
        pnl_arr, net_debit = strategy_pnl(legs, S, spot_range_sb)
        pnl_now = strategy_pnl_now(legs, spot_range_sb, r, sigma)
        summary  = strategy_summary(legs, S, pnl_arr, spot_range_sb, net_debit)

        # Summary metrics
        sm1, sm2, sm3, sm4 = st.columns(4)
        with sm1:
            mp = summary["max_profit"]
            st.metric("Max Profit",  f"${mp:.2f}" if mp != float("inf") else "Unlimited")
        with sm2:
            ml = summary["max_loss"]
            st.metric("Max Loss",  f"${ml:.2f}" if ml != float("-inf") else "Unlimited")
        with sm3:
            nd = summary["net_debit"]
            st.metric("Net Debit/Credit", f"{'Debit' if nd>0 else 'Credit'} ${abs(nd):.2f}")
        with sm4:
            bes = summary["breakevens"]
            st.metric("Breakeven(s)", ", ".join(f"${b:.2f}" for b in bes) if bes else "N/A")

        # P&L diagram
        fig_strat = go.Figure()
        fig_strat.add_trace(go.Scatter(
            x=spot_range_sb, y=pnl_arr, name="At Expiry",
            line=dict(color="#ffd700", width=3),
            fill="tozeroy",
            fillcolor="rgba(255,215,0,0.08)",
        ))
        fig_strat.add_trace(go.Scatter(
            x=spot_range_sb, y=pnl_now, name="Today (BS)",
            line=dict(color="#00d4aa", width=2, dash="dot"),
        ))
        fig_strat.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_strat.add_vline(x=S, line_dash="dash", line_color="#a0a0a0",
                            annotation_text=f"Spot ${S:.0f}")
        for be in summary["breakevens"]:
            fig_strat.add_vline(x=be, line_dash="dot", line_color="#ff6b6b",
                                annotation_text=f"BE ${be:.0f}", opacity=0.7)
        for lg in legs:
            if lg["type"] != "stock":
                fig_strat.add_vline(x=lg["strike"], line_dash="dot",
                                    line_color="rgba(255,255,255,0.25)")
        fig_strat.update_layout(
            **_layout,
            title=f"{strategy_name} — P&L Diagram",
            xaxis_title="Stock Price at Expiry ($)",
            yaxis_title="P&L ($)",
            height=430,
        )
        st.plotly_chart(fig_strat, use_container_width=True)

        # P&L Heatmap
        st.markdown("#### P&L Heatmap — Price × Time Remaining")
        with st.spinner("Computing heatmap…"):
            sp_hm, d_hm, pm_hm = pnl_heatmap(legs, S, T, r, sigma, n_price=55, n_time=25)

        max_abs = max(abs(pm_hm.max()), abs(pm_hm.min())) or 1
        fig_hm2 = go.Figure(go.Heatmap(
            z=pm_hm,
            x=np.round(sp_hm, 2),
            y=np.round(d_hm, 1),
            colorscale="RdYlGn",
            zmid=0,
            zmin=-max_abs, zmax=max_abs,
            colorbar=dict(title="P&L ($)"),
            hovertemplate="Spot: $%{x:.2f}<br>Days left: %{y:.0f}<br>P&L: $%{z:.2f}<extra></extra>",
        ))
        fig_hm2.add_vline(x=S, line_dash="dash", line_color="white", opacity=0.7,
                          annotation_text="Current Spot")
        fig_hm2.update_layout(
            **{**_layout, "hovermode": "closest"},
            title=f"{strategy_name} — P&L Heatmap",
            xaxis_title="Stock Price ($)",
            yaxis_title="Days to Expiry",
            height=430,
        )
        st.plotly_chart(fig_hm2, use_container_width=True)

        # Strategy Greeks
        with st.expander("Strategy-Level Greeks"):
            total_delta = sum(
                (1 if lg["action"] == "long" else -1) * lg["qty"] *
                (delta(S, lg["strike"], lg["T"], r, sigma, lg["type"], q) if lg["type"] != "stock"
                 else (1.0 if lg["action"] == "long" else -1.0))
                for lg in legs
            )
            total_gamma = sum(
                (1 if lg["action"] == "long" else -1) * lg["qty"] * 100 *
                gamma(S, lg["strike"], lg["T"], r, sigma, q)
                for lg in legs if lg["type"] != "stock"
            )
            total_theta = sum(
                (1 if lg["action"] == "long" else -1) * lg["qty"] * 100 *
                theta(S, lg["strike"], lg["T"], r, sigma, lg.get("type", "call"), q)
                for lg in legs if lg["type"] != "stock"
            )
            total_vega = sum(
                (1 if lg["action"] == "long" else -1) * lg["qty"] * 100 *
                vega(S, lg["strike"], lg["T"], r, sigma, q)
                for lg in legs if lg["type"] != "stock"
            )
            gc1, gc2, gc3, gc4 = st.columns(4)
            gc1.metric("Net Delta",  f"{total_delta:.4f}")
            gc2.metric("Net Gamma",  f"{total_gamma:.4f}")
            gc3.metric("Net Theta",  f"${total_theta:.4f}/day")
            gc4.metric("Net Vega",   f"${total_vega:.4f}/1%vol")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — OPTIONS CHAIN
# ══════════════════════════════════════════════════════════════════════════════
with tab_chain:
    st.markdown("### Live Options Chain")
    st.markdown("Compares market prices against Black-Scholes theoretical values.")

    ch_c1, ch_c2 = st.columns([2, 1])
    with ch_c1:
        chain_ticker = st.text_input(
            "Ticker", value=st.session_state.chain_ticker or st.session_state.ticker,
            placeholder="e.g. AAPL", key="chain_ticker_input",
        ).upper().strip()
    with ch_c2:
        fetch_chain = st.button("Load Chain", type="primary", use_container_width=True)

    if fetch_chain and chain_ticker:
        st.session_state.chain_ticker = chain_ticker
        with st.spinner("Fetching options expirations…"):
            expiries = _expiries(chain_ticker)
        if expiries:
            st.session_state["chain_expiries"] = expiries
        else:
            st.error("No options data found for this ticker.")

    expiries = st.session_state.get("chain_expiries", [])

    if expiries:
        ch_c3, ch_c4, ch_c5 = st.columns([2, 1, 1])
        with ch_c3:
            sel_expiry = st.selectbox("Expiration", expiries, key="chain_expiry_sel")
        with ch_c4:
            chain_otype = st.radio("Type", ["calls", "puts"], horizontal=True, key="chain_otype")
        with ch_c5:
            atm_band = st.slider("Strike Band ±%", 10, 50, 25, key="chain_atm_band") / 100.0

        load_chain_data = st.button("Load Selected Expiry", key="load_chain_expiry")

        if load_chain_data:
            with st.spinner("Fetching chain and computing implied vols…"):
                ch_info = _stock_info(chain_ticker)
                chain_S = ch_info.get("price", S) if ch_info.get("success") else S
                dte = days_to_expiry(sel_expiry)
                chain_T = max(dte / 365.0, 1/365)
                calls_raw, puts_raw = _chain(chain_ticker, sel_expiry)
                calls_en = enrich_chain(calls_raw, chain_S, chain_T, r, "call", atm_band)
                puts_en  = enrich_chain(puts_raw,  chain_S, chain_T, r, "put",  atm_band)
                st.session_state.chain_calls  = calls_en
                st.session_state.chain_puts   = puts_en
                st.session_state.chain_S      = chain_S
                st.session_state.chain_T      = chain_T

        calls_en = st.session_state.get("chain_calls")
        puts_en  = st.session_state.get("chain_puts")
        chain_S  = st.session_state.get("chain_S", S)
        chain_T  = st.session_state.get("chain_T", T)

        if calls_en is not None and not calls_en.empty:
            display_df = calls_en if chain_otype == "calls" else puts_en

            if display_df is not None and not display_df.empty:
                show_cols = [
                    "strike", "mid_price", "bs_price",
                    "mispricing", "mispricing_pct", "signal",
                    "yf_iv_pct", "our_iv_pct",
                    "volume", "openInterest",
                ]
                show_cols = [c for c in show_cols if c in display_df.columns]
                fmt_df = display_df[show_cols].copy()
                fmt_df = fmt_df.rename(columns={
                    "strike":         "Strike",
                    "mid_price":      "Mid Price",
                    "bs_price":       "BS Price",
                    "mispricing":     "Mispricing ($)",
                    "mispricing_pct": "Mispricing (%)",
                    "signal":         "Signal",
                    "yf_iv_pct":      "Market IV (%)",
                    "our_iv_pct":     "Our IV (%)",
                    "volume":         "Volume",
                    "openInterest":   "OI",
                })

                def _highlight(row):
                    sig = row.get("Signal", "")
                    if sig == "overpriced":
                        return ["background-color: rgba(255,107,107,0.25)"] * len(row)
                    if sig == "underpriced":
                        return ["background-color: rgba(0,212,170,0.20)"] * len(row)
                    return [""] * len(row)

                st.markdown(f"**{chain_ticker} {chain_otype.upper()} — Expiry {sel_expiry} "
                            f"({days_to_expiry(sel_expiry)} DTE) — Spot ${chain_S:.2f}**")
                st.dataframe(
                    fmt_df.style.apply(_highlight, axis=1).format({
                        "Strike": "${:.2f}", "Mid Price": "${:.4f}", "BS Price": "${:.4f}",
                        "Mispricing ($)": "${:.4f}", "Mispricing (%)": "{:.2f}%",
                        "Market IV (%)": "{:.1f}%", "Our IV (%)": "{:.1f}%",
                    }),
                    use_container_width=True, height=400,
                )
                st.caption("Green rows = underpriced vs BS model. Red = overpriced. "
                           "Threshold ±5% of BS price.")

                # IV Smile chart
                st.markdown("#### Implied Volatility Smile")
                smile_df = vol_smile(calls_en, puts_en, chain_S, chain_T, r, atm_band)

                if not smile_df.empty:
                    fig_smile = go.Figure()
                    for otype_s, col_s in [("call", "#00d4aa"), ("put", "#ff6b6b")]:
                        sub = smile_df[smile_df["option_type"] == otype_s]
                        if not sub.empty:
                            fig_smile.add_trace(go.Scatter(
                                x=sub["strike"], y=sub["iv_pct"],
                                name=f"{otype_s.capitalize()} IV",
                                mode="lines+markers",
                                line=dict(color=col_s, width=2),
                                marker=dict(size=6),
                            ))
                    fig_smile.add_vline(x=chain_S, line_dash="dash", line_color="white",
                                       annotation_text=f"Spot ${chain_S:.2f}")
                    fig_smile.update_layout(
                        **_layout,
                        title=f"IV Smile — {chain_ticker} {sel_expiry}",
                        xaxis_title="Strike ($)",
                        yaxis_title="Implied Volatility (%)",
                        height=380,
                    )
                    st.plotly_chart(fig_smile, use_container_width=True)
                else:
                    st.warning("Not enough data to plot the IV smile.")
    else:
        st.info("Enter a ticker and click 'Load Chain' to see live options data.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — VOLATILITY SURFACE
# ══════════════════════════════════════════════════════════════════════════════
with tab_surface:
    st.markdown("### Implied Volatility Surface")

    vs_c1, vs_c2 = st.columns([3, 1])
    with vs_c1:
        vs_ticker = st.text_input(
            "Ticker (leave blank for synthetic surface)",
            value=st.session_state.chain_ticker or st.session_state.ticker,
            key="vs_ticker_input",
        ).upper().strip()
    with vs_c2:
        fetch_surface = st.button("Build Surface", type="primary", use_container_width=True)

    use_synthetic = not vs_ticker

    if fetch_surface:
        if vs_ticker:
            with st.spinner("Fetching options chain across all expirations…"):
                vs_info = _stock_info(vs_ticker)
                vs_S = vs_info.get("price", S) if vs_info.get("success") else S
                expiries_all = _expiries(vs_ticker)

                chain_by_expiry = {}
                for exp in expiries_all[:10]:   # cap at 10 expiries for speed
                    dte = days_to_expiry(exp)
                    T_exp = max(dte / 365.0, 1/365)
                    c_df, p_df = _chain(vs_ticker, exp)
                    if not c_df.empty or not p_df.empty:
                        chain_by_expiry[exp] = (T_exp, c_df, p_df)

                surface_df = build_surface_from_chain(chain_by_expiry, vs_S, r)
                st.session_state.vol_surface_df = surface_df
                st.session_state.vs_S = vs_S
        else:
            # Synthetic
            T_vals = [1/52, 1/12, 2/12, 3/12, 6/12, 1.0, 1.5, 2.0]
            strikes_syn, T_syn, vol_mat_syn = build_synthetic_surface(
                S, K, T_vals, sigma
            )
            st.session_state.vol_surface_df  = None
            st.session_state.vol_mat_syn     = vol_mat_syn
            st.session_state.strikes_syn     = strikes_syn
            st.session_state.T_syn           = T_syn
            st.session_state.vs_S            = S

    vs_S = st.session_state.get("vs_S", S)
    surface_df = st.session_state.get("vol_surface_df")
    vol_mat_syn = st.session_state.get("vol_mat_syn")

    if surface_df is not None and not surface_df.empty:
        # 3D scatter surface from live data
        fig_3d = go.Figure(go.Scatter3d(
            x=surface_df["strike"],
            y=surface_df["T_days"],
            z=surface_df["iv_pct"],
            mode="markers",
            marker=dict(
                size=4,
                color=surface_df["iv_pct"],
                colorscale="Plasma",
                colorbar=dict(title="IV (%)"),
                opacity=0.85,
            ),
            hovertemplate=(
                "Strike: $%{x:.2f}<br>"
                "Days: %{y:.0f}<br>"
                "IV: %{z:.1f}%<extra></extra>"
            ),
        ))
        fig_3d.update_layout(
            scene=dict(
                xaxis_title="Strike ($)",
                yaxis_title="Days to Expiry",
                zaxis_title="IV (%)",
                bgcolor="rgba(0,0,0,0)",
            ),
            title=f"{vs_ticker} — Implied Volatility Surface",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=40, b=0),
            height=580,
        )
        st.plotly_chart(fig_3d, use_container_width=True)

        # Term structure
        ts_df = vol_term_structure(surface_df, vs_S)
        if not ts_df.empty:
            col_ts, col_smile = st.columns(2)
            with col_ts:
                fig_ts = go.Figure(go.Scatter(
                    x=ts_df["T_days"], y=ts_df["atm_iv_pct"],
                    mode="lines+markers",
                    line=dict(color="#00d4aa", width=2.5),
                    marker=dict(size=8),
                    hovertemplate="Days: %{x:.0f}<br>ATM IV: %{y:.2f}%<extra></extra>",
                ))
                fig_ts.update_layout(
                    **_layout,
                    title="ATM Volatility Term Structure",
                    xaxis_title="Days to Expiry",
                    yaxis_title="ATM IV (%)",
                    height=360,
                )
                st.plotly_chart(fig_ts, use_container_width=True)

            with col_smile:
                # Vol smile for shortest expiry with enough data
                expiry_counts = surface_df.groupby("expiry").size()
                best_expiry = expiry_counts[expiry_counts >= 5].index[0] if any(expiry_counts >= 5) else None
                if best_expiry:
                    slice_df = surface_df[surface_df["expiry"] == best_expiry].sort_values("strike")
                    fig_sl = go.Figure(go.Scatter(
                        x=slice_df["strike"], y=slice_df["iv_pct"],
                        mode="lines+markers",
                        line=dict(color="#ff6b6b", width=2.5),
                        marker=dict(size=7),
                        name=best_expiry,
                    ))
                    fig_sl.add_vline(x=vs_S, line_dash="dash", line_color="white",
                                     annotation_text=f"Spot ${vs_S:.2f}")
                    fig_sl.update_layout(
                        **_layout,
                        title=f"Vol Smile — {best_expiry}",
                        xaxis_title="Strike ($)",
                        yaxis_title="IV (%)",
                        height=360,
                    )
                    st.plotly_chart(fig_sl, use_container_width=True)

    elif vol_mat_syn is not None:
        # Synthetic surface
        strikes_syn = st.session_state.strikes_syn
        T_syn       = st.session_state.T_syn

        fig_syn = go.Figure(go.Surface(
            z=vol_mat_syn,
            x=strikes_syn,
            y=T_syn * 365,
            colorscale="Plasma",
            colorbar=dict(title="IV (%)"),
            hovertemplate="Strike: $%{x:.2f}<br>Days: %{y:.0f}<br>IV: %{z:.1f}%<extra></extra>",
        ))
        fig_syn.update_layout(
            scene=dict(
                xaxis_title="Strike ($)",
                yaxis_title="Days to Expiry",
                zaxis_title="IV (%)",
                bgcolor="rgba(0,0,0,0)",
            ),
            title="Synthetic Volatility Surface (Equity Skew/Smile Model)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=40, b=0),
            height=560,
        )
        st.plotly_chart(fig_syn, use_container_width=True)

        st.markdown("""
        <div class="info-box">
        <b>Synthetic surface model:</b> IV(K,T) = σ_base + skew·ln(K/S) + smile·ln(K/S)² + term_structure·√T.
        Negative skew mimics the typical equity vol skew — lower strikes carry higher IV (put demand,
        crash-risk premium). Enter a live ticker above to see real market data.
        </div>
        """, unsafe_allow_html=True)

    else:
        # Default: show synthetic immediately
        T_vals = [1/52, 1/12, 2/12, 3/12, 6/12, 1.0, 1.5, 2.0]
        strikes_d, T_d, vm_d = build_synthetic_surface(S, K, T_vals, sigma)
        fig_d = go.Figure(go.Surface(
            z=vm_d, x=strikes_d, y=T_d * 365,
            colorscale="Plasma",
            colorbar=dict(title="IV (%)"),
        ))
        fig_d.update_layout(
            scene=dict(
                xaxis_title="Strike ($)",
                yaxis_title="Days to Expiry",
                zaxis_title="IV (%)",
                bgcolor="rgba(0,0,0,0)",
            ),
            title="Synthetic IV Surface (current params)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=40, b=0),
            height=520,
        )
        st.plotly_chart(fig_d, use_container_width=True)
        st.info("Enter a ticker and click 'Build Surface' to load real market implied-volatility data.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Black-Scholes Option Pricing Calculator · "
    "Educational purposes only · Not financial advice · "
    "Live data via yfinance"
)
