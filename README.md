# Black-Scholes Option Pricing Calculator

![Python](https://img.shields.io/badge/python-3.9%2B-blue) ![Streamlit](https://img.shields.io/badge/streamlit-1.28%2B-red) ![License](https://img.shields.io/badge/license-MIT-green)

A fully interactive web app for pricing European options, computing the Greeks, solving for implied volatility, and visualizing payoff diagrams and price surfaces. Built with Python and Streamlit.

---

## Features

- **Call and Put pricing** using the Black-Scholes closed-form solution
- **All five Greeks**: Delta, Gamma, Theta, Vega, Rho (call and put)
- **Implied Volatility solver** using Brent's root-finding method
- **P&L payoff diagram** at expiration (call, put, or both)
- **3D option price surface** over volatility and spot price
- **Interactive sidebar** sliders for all input parameters

---

## Tech Stack

| Layer | Library |
|-------|---------|
| Web UI | Streamlit |
| Math / numerics | NumPy, SciPy |
| Visualization | Plotly |
| Data | Pandas |

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/Krish-1717/Black-Scholes-Option-Pricing-Calculator-Web-App.git
cd Black-Scholes-Option-Pricing-Calculator-Web-App

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501` by default.

---

## Usage

Use the **sidebar sliders** to set:

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Stock Price | S | Current underlying price |
| Strike Price | K | Option exercise price |
| Time to Expiry | T | Years until expiration |
| Risk-Free Rate | r | Annualized continuously compounded rate |
| Volatility | σ | Annualized implied or historical vol |

The main panel updates instantly to show prices, Greeks, and charts.

### Implied Volatility

Enter a market price under **Implied Volatility Calculator** and select call or put. The solver uses Brent's method to find the volatility that matches the observed price.

---

## Project Structure

```
.
├── app.py            # Streamlit front end
├── black_scholes.py  # Core pricing functions (call, put, d1, d2)
├── greeks.py         # Delta, Gamma, Theta, Vega, Rho
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Black-Scholes Formula

For a European call option:

```
C = S * N(d1) - K * e^(-rT) * N(d2)

d1 = [ln(S/K) + (r + σ²/2) * T] / (σ * √T)
d2 = d1 - σ * √T
```

Put-call parity gives the put price:

```
P = K * e^(-rT) * N(-d2) - S * N(-d1)
```

Where `N(·)` is the standard normal CDF.

---

## Assumptions

- European-style options (no early exercise)
- No dividends
- Constant volatility and risk-free rate
- Log-normal distribution of returns
- No transaction costs

---

## License

MIT License. Free to use and modify for personal or educational purposes.

---

> Built for educational purposes. Not financial advice.
