"""
market_data.py — yfinance wrapper for live market data.

Provides:
  get_stock_info()           — current price, 52-week range, market cap, sector
  get_historical_volatility() — annualised vol from log-returns
  get_risk_free_rate()        — 3-month US T-bill rate via ^IRX
  get_options_expirations()   — list of available option expiries
  get_options_chain()         — calls / puts DataFrames for a given expiry
  get_price_history()         — OHLCV history

All functions return sensible fallback values on failure so the app
never crashes due to a network or data error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime


# ---------------------------------------------------------------------------
# Stock information
# ---------------------------------------------------------------------------

def get_stock_info(ticker: str) -> dict:
    """
    Fetch basic stock information from yfinance.

    Returns a dict with keys:
        ticker, name, price, 52_week_high, 52_week_low,
        market_cap, sector, industry, beta, pe_ratio,
        dividend_yield, currency, exchange, success
    On failure, success=False and error contains the message.
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info

        # Resolve current price from multiple possible fields
        price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )

        return {
            "ticker":         ticker.upper(),
            "name":           info.get("longName") or info.get("shortName", ticker.upper()),
            "price":          float(price) if price else None,
            "52_week_high":   info.get("fiftyTwoWeekHigh"),
            "52_week_low":    info.get("fiftyTwoWeekLow"),
            "market_cap":     info.get("marketCap"),
            "sector":         info.get("sector", "N/A"),
            "industry":       info.get("industry", "N/A"),
            "beta":           info.get("beta"),
            "pe_ratio":       info.get("trailingPE"),
            "dividend_yield": info.get("dividendYield"),
            "currency":       info.get("currency", "USD"),
            "exchange":       info.get("exchange", "N/A"),
            "avg_volume":     info.get("averageVolume"),
            "success":        True,
        }
    except Exception as exc:
        return {"success": False, "error": str(exc), "ticker": ticker.upper()}


# ---------------------------------------------------------------------------
# Historical volatility
# ---------------------------------------------------------------------------

def get_historical_volatility(ticker: str, window: int = 30) -> float | None:
    """
    Compute annualised historical volatility from log-returns over
    the most recent `window` trading days.

    Uses 252 trading days per year.
    Returns None if data is unavailable or insufficient.
    """
    try:
        stock = yf.Ticker(ticker.upper())
        hist = stock.history(period="180d")

        if len(hist) < window + 1:
            hist = stock.history(period="2y")

        closes = hist["Close"].dropna()
        if len(closes) < window + 1:
            return None

        recent = closes.iloc[-(window + 1):]
        log_returns = np.log(recent / recent.shift(1)).dropna()
        vol = float(log_returns.std() * np.sqrt(252))
        return vol
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Risk-free rate (3-month US T-bill via ^IRX)
# ---------------------------------------------------------------------------

def get_risk_free_rate() -> float:
    """
    Fetch the current 3-month US T-bill yield from yfinance (^IRX).

    ^IRX is quoted in percent, so we divide by 100.
    Falls back to 5.30 % (reasonable as of 2025) if unavailable.
    """
    try:
        tbill = yf.Ticker("^IRX")
        hist = tbill.history(period="5d")
        if not hist.empty:
            rate = float(hist["Close"].iloc[-1]) / 100.0
            if 0.0 < rate < 0.30:   # sanity-check: 0–30 %
                return rate
    except Exception:
        pass
    return 0.053   # 5.3 % default


# ---------------------------------------------------------------------------
# Options expirations
# ---------------------------------------------------------------------------

def get_options_expirations(ticker: str) -> list[str]:
    """
    Return list of available option expiration dates (YYYY-MM-DD strings)
    sorted chronologically.
    """
    try:
        stock = yf.Ticker(ticker.upper())
        return list(stock.options)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Options chain
# ---------------------------------------------------------------------------

def get_options_chain(
    ticker: str, expiration: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (calls_df, puts_df) for the given ticker and expiry.

    Both DataFrames contain at minimum:
        strike, lastPrice, bid, ask, impliedVolatility,
        openInterest, volume, inTheMoney
    """
    try:
        stock = yf.Ticker(ticker.upper())
        chain = stock.option_chain(expiration)
        return chain.calls.copy(), chain.puts.copy()
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


# ---------------------------------------------------------------------------
# Price history
# ---------------------------------------------------------------------------

def get_price_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Return OHLCV history for the given ticker and period.

    Valid period strings: 1d 5d 1mo 3mo 6mo 1y 2y 5y 10y ytd max
    """
    try:
        stock = yf.Ticker(ticker.upper())
        hist = stock.history(period=period)
        return hist
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def format_market_cap(mc: float | None) -> str:
    """Return a human-readable market-cap string (e.g. '$2.94T')."""
    if mc is None:
        return "N/A"
    if mc >= 1e12:
        return f"${mc/1e12:.2f}T"
    if mc >= 1e9:
        return f"${mc/1e9:.2f}B"
    if mc >= 1e6:
        return f"${mc/1e6:.2f}M"
    return f"${mc:,.0f}"


def days_to_expiry(expiration_str: str) -> int:
    """Calendar days from today to expiration date (YYYY-MM-DD)."""
    try:
        exp_date = datetime.strptime(expiration_str, "%Y-%m-%d")
        return max((exp_date - datetime.today()).days, 0)
    except Exception:
        return 30
