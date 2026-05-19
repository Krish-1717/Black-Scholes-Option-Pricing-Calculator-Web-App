# Black-Scholes Option Pricing Calculator — Premium Edition

A professional-grade options analytics web app built with Streamlit and Python.

## Features

### Core Pricing
- **Black-Scholes** closed-form pricing for European calls and puts
- **Put-Call Parity** check displayed in real time
- **Binomial Tree** (CRR model, 200 steps) for American and European options
  - Visual tree with colour-coded early-exercise nodes
  - American vs European price comparison

### Live Market Data
- **Ticker Lookup** — type any symbol (AAPL, SPY, etc.) to auto-fill:
  - Current stock price, 52-week high/low, market cap, sector, beta, P/E
  - 30-day historical volatility from log-returns
  - Current 3-month T-bill risk-free rate (via `^IRX`)
- Candlestick price chart and rolling historical-volatility chart

### Greeks
**First-order:** Delta, Gamma, Theta, Vega, Rho

**Second-order:**
- Charm (delta decay per calendar day)
- Vanna (delta sensitivity to vol)
- Vomma / Volga (vega sensitivity to vol)
- Speed (gamma sensitivity to price)
- Color (gamma decay per calendar day)
- Ultima (third-order vol sensitivity)

### Monte Carlo Simulation
- 100,000-path GBM simulation with antithetic-variate variance reduction
- Adjustable display paths and random seed
- ITM/OTM terminal price distribution histogram
- MC vs Black-Scholes price comparison with 95% confidence interval
- Convergence analysis chart (price vs. number of paths)

### Multi-Leg Strategy Builder
Predefined strategies: Long Call/Put, Covered Call, Protective Put, Bull/Bear Spreads,
Straddle, Strangle, Iron Condor, Iron Butterfly, Butterfly, Calendar Spread, Ratio Spread

For each strategy:
- At-expiry P&L diagram (with current-time BS curve overlay)
- **P&L Scenario Heatmap** across stock prices AND time to expiry
- Max profit, max loss, breakeven prices
- Net strategy Greeks (delta, gamma, theta, vega)

### Live Options Chain
- Real market call/put prices from yfinance
- Black-Scholes theoretical price and mispricing (%) for each strike
- Colour-coded table: green = underpriced, red = overpriced (±5% threshold)
- Implied volatility from market prices vs yfinance IV
- **IV Smile** chart for selected expiry

### Volatility Surface
- **3D IV surface** across all available strikes and expirations
- Live data from yfinance (up to 10 expirations)
- ATM volatility term structure chart
- Per-expiry vol smile
- Falls back to a parametric synthetic surface (equity skew + smile + term structure)

### P&L Scenario Heatmap (Single Option)
- 2D heatmap: stock price × time remaining → P&L
- Visualises theta decay and delta exposure simultaneously

## File Structure

```
├── app.py                  Main Streamlit application (6-tab layout)
├── black_scholes.py        Black-Scholes pricing formulas (d1, d2, call, put)
├── greeks.py               First-order Greeks (delta, gamma, theta, vega, rho)
├── advanced_greeks.py      Second/third-order Greeks (charm, vanna, vomma, speed, color, ultima)
├── binomial_tree.py        CRR binomial tree (American/European)
├── monte_carlo.py          GBM path simulation and MC pricer
├── options_chain.py        Live chain enrichment with BS comparison and IV calculation
├── strategy_builder.py     Multi-leg strategy P&L engine
├── market_data.py          yfinance wrapper (stock info, HV, risk-free rate)
├── volatility_surface.py   IV surface builder and vol smile extraction
├── requirements.txt
└── run.sh
```

## Installation

```bash
pip install -r requirements.txt
```

## Running

```bash
streamlit run app.py
```

Or use the helper script:
```bash
chmod +x run.sh && ./run.sh
```

## Technical Notes

- All charts are interactive Plotly figures (hover, zoom, pan)
- Market data is cached for 5 minutes via `@st.cache_data`
- IV solver uses Brent's method (`scipy.optimize.brentq`) with a [0.01%, 1000%] search range
- Monte Carlo uses antithetic variates (effectively 200k paths from 100k draws)
- Binomial tree uses 200 steps for pricing, fewer for visualisation
- Synthetic vol surface uses a parametric model: σ(K,T) = σ₀ + skew·ln(K/S) + smile·ln(K/S)² + term·√T

## Disclaimer

Educational purposes only. Not financial advice. Options involve significant risk.
