"""
app.py — Fixed Income Bond Pricer & Yield Curve Dashboard

Four-tab Streamlit application:
  1. Bond Pricer     — Price, YTM, full analytics for any bond
  2. Yield Curve     — Live Treasury curve, Nelson-Siegel fit, forward rates
  3. Sensitivity     — Duration, convexity, price-yield relationship
  4. Scenario Analysis — Parallel shifts, steepening/flattening shocks
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from bond_pricer import (
    Bond, bond_price, ytm as calc_ytm,
    macaulay_duration, modified_duration, dollar_duration, dv01,
    convexity, dollar_convexity, bond_analytics,
    price_change_taylor, price_yield_curve,
)
from yield_curve import (
    YieldCurve, get_treasury_yields, fit_nelson_siegel,
    nelson_siegel_rate, bootstrap_spot_rates,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fixed Income Bond Pricer",
    page_icon="💵",
    layout="wide",
)

_dark = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=20, r=20, t=40, b=20),
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Fixed Income")
    st.markdown("---")
    st.markdown("**Bond Parameters**")

    face_value = st.number_input("Face Value ($)", value=1000.0, step=100.0)
    coupon_rate = st.slider("Coupon Rate (%)", 0.0, 20.0, 5.0, 0.25) / 100
    maturity = st.slider("Maturity (years)", 0.5, 30.0, 10.0, 0.5)
    frequency = st.radio("Frequency", [1, 2, 4], format_func=lambda x: {1: "Annual", 2: "Semi-annual", 4: "Quarterly"}[x], index=1)
    ytm_input = st.slider("YTM / Discount Rate (%)", 0.1, 20.0, 5.0, 0.05) / 100

    st.markdown("---")
    st.markdown("**Yield Curve**")
    curve_method = st.selectbox("Interpolation", ["cubic", "linear", "nelson-siegel"])
    load_treasury = st.button("Load Live Treasury Yields", use_container_width=True)

bond = Bond(face_value=face_value, coupon_rate=coupon_rate,
            maturity=maturity, frequency=frequency)
price = bond_price(bond, ytm_input)

# ── Treasury curve session state ──────────────────────────────────────────────
if "treasury_yields" not in st.session_state:
    st.session_state.treasury_yields = None

if load_treasury:
    with st.spinner("Fetching Treasury yields..."):
        st.session_state.treasury_yields = get_treasury_yields()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_price, tab_curve, tab_sens, tab_scenario = st.tabs(
    ["Bond Pricer", "Yield Curve", "Sensitivity", "Scenario Analysis"]
)

# ═══════════════════════════════════════════════════════════════════
# TAB 1 — BOND PRICER
# ═══════════════════════════════════════════════════════════════════
with tab_price:
    st.markdown("### Bond Analytics")
    analytics = bond_analytics(bond, ytm_=ytm_input)

    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    r1c1.metric("Price", f"${analytics['price']:.4f}")
    r1c2.metric("YTM", f"{analytics['ytm']:.4%}")
    r1c3.metric("Current Yield", f"{analytics['current_yield']:.4%}")
    r1c4.metric("Price / Par", f"{analytics['price_to_par']:.4f}")

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    r2c1.metric("Macaulay Duration", f"{analytics['macaulay_duration']:.4f} yrs")
    r2c2.metric("Modified Duration", f"{analytics['modified_duration']:.4f}")
    r2c3.metric("DV01", f"${analytics['dv01']:.4f}")
    r2c4.metric("Convexity", f"{analytics['convexity']:.4f}")

    st.markdown("---")

    # Cash flow schedule
    cf = bond.cash_flows()
    t = bond.time_to_cash_flows()
    y_per = ytm_input / frequency
    df_arr = (1 + y_per) ** (-np.arange(1, bond.n_periods + 1))
    pv = cf * df_arr

    cf_df = pd.DataFrame({
        "Period": np.arange(1, bond.n_periods + 1),
        "Time (yrs)": t.round(4),
        "Cash Flow ($)": cf.round(4),
        "Discount Factor": df_arr.round(6),
        "PV ($)": pv.round(4),
        "Weight": (pv / pv.sum()).round(6),
    })
    st.markdown("#### Cash Flow Schedule")
    st.dataframe(cf_df, use_container_width=True, hide_index=True)

    # Cash flow chart
    fig = go.Figure()
    fig.add_trace(go.Bar(x=t, y=cf, name="Cash Flow",
                         marker_color="#00d4aa", opacity=0.8))
    fig.add_trace(go.Bar(x=t, y=pv, name="PV of Cash Flow",
                         marker_color="#ffd700", opacity=0.7))
    fig.update_layout(**_dark, title="Cash Flow vs. Present Value",
                      xaxis_title="Time (years)", yaxis_title="Amount ($)",
                      barmode="group", height=380)
    st.plotly_chart(fig, use_container_width=True)

    # Implied YTM from custom price
    st.markdown("---")
    st.markdown("#### Implied YTM from Market Price")
    mkt_price = st.number_input("Market Price ($)", value=float(round(price, 2)),
                                min_value=1.0, step=1.0)
    if mkt_price:
        implied_ytm = calc_ytm(bond, mkt_price)
        iy1, iy2, iy3 = st.columns(3)
        iy1.metric("Implied YTM", f"{implied_ytm:.4%}")
        iy2.metric("Premium / Discount", f"${mkt_price - face_value:.2f}")
        iy3.metric("Modified Duration", f"{modified_duration(bond, implied_ytm):.4f}")

# ═══════════════════════════════════════════════════════════════════
# TAB 2 — YIELD CURVE
# ═══════════════════════════════════════════════════════════════════
with tab_curve:
    st.markdown("### Yield Curve")

    raw_yields = st.session_state.treasury_yields
    if raw_yields is None:
        st.info("Click **Load Live Treasury Yields** in the sidebar, or enter rates below.")
        # Manual input
        default_curve = {0.25: 5.2, 0.5: 5.1, 1.0: 4.9, 2.0: 4.7,
                         3.0: 4.6, 5.0: 4.5, 7.0: 4.5, 10.0: 4.4, 30.0: 4.7}
        cols_per_row = 4
        entries = list(default_curve.items())
        raw_yields = {}
        for i in range(0, len(entries), cols_per_row):
            row_entries = entries[i:i + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, (mat, default_rate) in zip(cols, row_entries):
                val = col.number_input(f"{mat}y (%)", value=default_rate, step=0.05,
                                       key=f"manual_yield_{mat}")
                raw_yields[mat] = val / 100
    else:
        st.success("Using live Treasury yields.")
        for mat, rate in raw_yields.items():
            st.write(f"{mat}y: {rate:.3%}")

    mats = np.array(sorted(raw_yields.keys()))
    ylds = np.array([raw_yields[m] for m in mats])

    try:
        curve = YieldCurve(mats, ylds, method=curve_method)
        t_fine = np.linspace(mats[0], mats[-1], 300)
        spot_fine = curve.spot_rate(t_fine)
        fwd_tenors = np.linspace(mats[0], mats[-1] - 0.5, 200)
        fwd_rates = curve.forward_curve(fwd_tenors, tenor_width=0.5)

        fig = make_subplots(rows=1, cols=2, subplot_titles=["Spot Rates", "6M Forward Rates"])
        fig.add_trace(go.Scatter(x=t_fine, y=spot_fine, name=f"Spot ({curve_method})",
                                 line=dict(color="#00d4aa", width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=mats, y=ylds, mode="markers",
                                 marker=dict(color="#ffd700", size=8), name="Observed"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=fwd_tenors, y=fwd_rates, name="6M Fwd Rate",
                                 line=dict(color="#ff6b6b", width=2)), row=1, col=2)
        fig.update_yaxes(tickformat=".2%", row=1, col=1)
        fig.update_yaxes(tickformat=".2%", row=1, col=2)
        fig.update_layout(**_dark, height=420)
        st.plotly_chart(fig, use_container_width=True)

        # Nelson-Siegel fit
        ns = fit_nelson_siegel(mats, ylds)
        ns_col1, ns_col2, ns_col3, ns_col4 = st.columns(4)
        ns_col1.metric("β₀ (long-run)", f"{ns['beta0']:.4%}")
        ns_col2.metric("β₁ (slope)", f"{ns['beta1']:.4%}")
        ns_col3.metric("β₂ (curvature)", f"{ns['beta2']:.4%}")
        ns_col4.metric("τ (shape)", f"{ns['tau']:.4f}")
        st.caption(f"Nelson-Siegel RMSE: {ns['rmse']:.6%}")

    except Exception as e:
        st.error(f"Could not build yield curve: {e}")

# ═══════════════════════════════════════════════════════════════════
# TAB 3 — SENSITIVITY
# ═══════════════════════════════════════════════════════════════════
with tab_sens:
    st.markdown("### Price-Yield Sensitivity")

    ytm_range = np.linspace(max(0.001, ytm_input - 0.05), ytm_input + 0.05, 300)
    prices = price_yield_curve(bond, ytm_range)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ytm_range, y=prices, name="Exact Price",
                             line=dict(color="#00d4aa", width=2)))

    # Taylor approximation
    dy_range = ytm_range - ytm_input
    p0 = bond_price(bond, ytm_input)
    md = modified_duration(bond, ytm_input)
    cx = convexity(bond, ytm_input)
    taylor_prices = p0 * (1 - md * dy_range + 0.5 * cx * dy_range ** 2)

    fig.add_trace(go.Scatter(x=ytm_range, y=taylor_prices, name="Taylor Approx",
                             line=dict(color="#ffd700", width=1.5, dash="dash")))
    fig.add_vline(x=ytm_input, line_dash="dash", line_color="gray",
                  annotation_text=f"Current YTM {ytm_input:.2%}")
    fig.add_hline(y=face_value, line_dash="dot", line_color="#a0a0a0",
                  annotation_text="Par")
    fig.update_layout(**_dark, title="Price vs YTM",
                      xaxis_title="YTM", yaxis_title="Price ($)",
                      xaxis_tickformat=".2%", height=420)
    st.plotly_chart(fig, use_container_width=True)

    # Sensitivity table
    st.markdown("#### Sensitivity to Yield Shocks")
    shocks = [-0.02, -0.01, -0.005, -0.001, 0, 0.001, 0.005, 0.01, 0.02]
    rows = []
    for dy in shocks:
        taylor = price_change_taylor(bond, ytm_input, dy)
        rows.append({
            "Yield Shock": f"{dy:+.2%}",
            "New YTM": f"{ytm_input + dy:.3%}",
            "Exact Price": f"${taylor['price_new_exact']:.4f}",
            "Taylor Approx": f"${taylor['price_new_approx']:.4f}",
            "Duration Effect": f"${taylor['dp_duration']:.4f}",
            "Convexity Effect": f"${taylor['dp_convexity']:.4f}",
            "Error": f"${taylor['approximation_error']:.6f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 4 — SCENARIO ANALYSIS
# ═══════════════════════════════════════════════════════════════════
with tab_scenario:
    st.markdown("### Scenario Analysis")

    scenarios = {
        "Parallel +200bp": 0.02,
        "Parallel +100bp": 0.01,
        "Parallel +50bp": 0.005,
        "Base Case": 0.0,
        "Parallel -50bp": -0.005,
        "Parallel -100bp": -0.01,
        "Parallel -200bp": -0.02,
    }

    p0 = bond_price(bond, ytm_input)
    rows = []
    for name, shift in scenarios.items():
        new_ytm = ytm_input + shift
        new_price = bond_price(bond, new_ytm)
        p_change = new_price - p0
        p_change_pct = (new_price / p0 - 1)
        rows.append({
            "Scenario": name,
            "New YTM": f"{new_ytm:.3%}",
            "Price": f"${new_price:.4f}",
            "P&L ($)": f"${p_change:+.4f}",
            "P&L (%)": f"{p_change_pct:+.4%}",
            "DV01": f"${dv01(bond, new_ytm, new_price):.4f}",
        })

    df_scenarios = pd.DataFrame(rows)
    st.dataframe(df_scenarios, use_container_width=True, hide_index=True)

    # P&L bar chart
    names = [r["Scenario"] for r in rows]
    pnls = [bond_price(bond, ytm_input + s) - p0 for s in scenarios.values()]
    colors = ["#00d4aa" if p >= 0 else "#ff6b6b" for p in pnls]

    fig = go.Figure(go.Bar(x=names, y=pnls, marker_color=colors))
    fig.add_hline(y=0, line_color="white", opacity=0.3)
    fig.update_layout(**_dark, title="Scenario P&L",
                      yaxis_title="P&L ($)", height=380)
    st.plotly_chart(fig, use_container_width=True)

    # Multi-bond comparison
    st.markdown("---")
    st.markdown("#### Compare Multiple Bonds")
    st.caption("How different bonds respond to a yield shift")

    shift_bp = st.slider("Yield Shift (bp)", -300, 300, 100, 25)
    shift = shift_bp / 10000
    maturities_to_compare = [1, 2, 5, 7, 10, 15, 20, 30]
    coupon_compare = st.slider("Coupon Rate for Comparison (%)", 0.0, 10.0, float(coupon_rate * 100), 0.25) / 100

    compare_rows = []
    for mat in maturities_to_compare:
        b = Bond(face_value=face_value, coupon_rate=coupon_compare, maturity=mat, frequency=frequency)
        p_base = bond_price(b, ytm_input)
        p_shift = bond_price(b, ytm_input + shift)
        md_ = modified_duration(b, ytm_input)
        cx_ = convexity(b, ytm_input)
        compare_rows.append({
            "Maturity": f"{mat}y",
            "Base Price": f"${p_base:.2f}",
            "Shifted Price": f"${p_shift:.2f}",
            "P&L ($)": f"${p_shift - p_base:+.2f}",
            "P&L (%)": f"{(p_shift / p_base - 1):+.3%}",
            "Mod Duration": f"{md_:.3f}",
            "Convexity": f"{cx_:.3f}",
        })

    st.dataframe(pd.DataFrame(compare_rows), use_container_width=True, hide_index=True)

    mats_arr = np.array(maturities_to_compare, dtype=float)
    pnl_arr = np.array([
        bond_price(Bond(face_value=face_value, coupon_rate=coupon_compare,
                        maturity=m, frequency=frequency), ytm_input + shift)
        - bond_price(Bond(face_value=face_value, coupon_rate=coupon_compare,
                          maturity=m, frequency=frequency), ytm_input)
        for m in maturities_to_compare
    ])

    fig2 = go.Figure(go.Bar(x=[f"{m}y" for m in maturities_to_compare], y=pnl_arr,
                             marker_color=["#00d4aa" if v >= 0 else "#ff6b6b" for v in pnl_arr]))
    fig2.update_layout(**_dark, title=f"P&L by Maturity ({shift_bp:+d}bp shift)",
                       yaxis_title="P&L ($)", height=350)
    st.plotly_chart(fig2, use_container_width=True)
