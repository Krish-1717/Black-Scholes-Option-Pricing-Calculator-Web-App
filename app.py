import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from black_scholes import call_price, put_price
from greeks import delta, gamma, theta, vega, rho

st.set_page_config(page_title="Black-Scholes Calculator", layout="wide", page_icon="📈")
st.title("📈 Black-Scholes Option Pricing Calculator")
st.markdown("---")

st.sidebar.header("Option Parameters")
S = st.sidebar.slider("Stock Price (S)", 10.0, 500.0, 100.0, 1.0)
K = st.sidebar.slider("Strike Price (K)", 10.0, 500.0, 100.0, 1.0)
T = st.sidebar.slider("Time to Expiry (Years)", 0.01, 3.0, 1.0, 0.01)
r = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 15.0, 5.0, 0.1) / 100
sigma = st.sidebar.slider("Volatility σ (%)", 1.0, 100.0, 20.0, 0.5) / 100
option_type = st.sidebar.radio("Option Type", ["Call", "Put", "Both"])

col1, col2 = st.columns(2)
call = call_price(S, K, T, r, sigma)
put = put_price(S, K, T, r, sigma)

with col1:
    st.metric("Call Price", f"${call:.4f}")
with col2:
    st.metric("Put Price", f"${put:.4f}")

st.markdown("---")

greeks_data = {
    "Greek": ["Delta", "Gamma", "Theta (per day)", "Vega (per 1% vol)", "Rho (per 1% rate)"],
    "Call": [
        f"{delta(S,K,T,r,sigma,'call'):.4f}",
        f"{gamma(S,K,T,r,sigma):.4f}",
        f"{theta(S,K,T,r,sigma,'call'):.4f}",
        f"{vega(S,K,T,r,sigma):.4f}",
        f"{rho(S,K,T,r,sigma,'call'):.4f}"
    ],
    "Put": [
        f"{delta(S,K,T,r,sigma,'put'):.4f}",
        f"{gamma(S,K,T,r,sigma):.4f}",
        f"{theta(S,K,T,r,sigma,'put'):.4f}",
        f"{vega(S,K,T,r,sigma):.4f}",
        f"{rho(S,K,T,r,sigma,'put'):.4f}"
    ]
}
st.subheader("Option Greeks")
st.table(pd.DataFrame(greeks_data))

st.markdown("---")
st.subheader("Payoff at Expiration")
spot_range = np.linspace(max(1, S * 0.5), S * 1.5, 200)
fig_pnl = go.Figure()

if option_type in ["Call", "Both"]:
    call_payoff = np.maximum(spot_range - K, 0) - call
    fig_pnl.add_trace(go.Scatter(x=spot_range, y=call_payoff, name="Call P&L", line=dict(color='green')))
if option_type in ["Put", "Both"]:
    put_payoff = np.maximum(K - spot_range, 0) - put
    fig_pnl.add_trace(go.Scatter(x=spot_range, y=put_payoff, name="Put P&L", line=dict(color='red')))

fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray")
fig_pnl.add_vline(x=K, line_dash="dot", line_color="blue", annotation_text="Strike")
fig_pnl.update_layout(title="P&L at Expiration", xaxis_title="Stock Price at Expiry", yaxis_title="Profit / Loss ($)")
st.plotly_chart(fig_pnl, use_container_width=True)

st.markdown("---")
st.subheader("Option Price Surface")
vol_range = np.linspace(0.05, 0.80, 30)
spot_range_3d = np.linspace(S * 0.6, S * 1.4, 30)
Z = np.array([[call_price(s, K, T, r, v) for v in vol_range] for s in spot_range_3d])

fig3d = go.Figure(data=[go.Surface(z=Z, x=vol_range * 100, y=spot_range_3d, colorscale='Viridis')])
fig3d.update_layout(
    title='Call Price Surface',
    scene=dict(xaxis_title='Volatility (%)', yaxis_title='Stock Price', zaxis_title='Call Price')
)
st.plotly_chart(fig3d, use_container_width=True)

st.markdown("---")
st.caption("Built with Black-Scholes model. For educational purposes only.")
