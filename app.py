import streamlit as st
import numpy as np
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

st.caption("Built with Black-Scholes model. For educational purposes only.")
