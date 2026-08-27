"""
Black-Scholes European option pricer and Greeks.

Parameter conventions used throughout this module:
    S     - spot price of the underlying
    K     - strike price
    T     - time to expiry, in years
    r     - risk-free rate, annualised, continuously compounded
    sigma - volatility of the underlying, annualised
    q     - continuous dividend yield, annualised (0.0 if none)

All options are European exercise. Prices are per unit of underlying.

UNIT CONVENTIONS:
    vega  - per 1 percentage point of vol (analytic derivative / 100)
    theta - per calendar day (analytic derivative / 365)
    rho   - per 1 percentage point of r (analytic derivative / 100)
"""

import numpy as np
from scipy.stats import norm


def d1(S, K, T, r, sigma, q=0.0):
    """Compute d1 in the Black-Scholes formula."""
    return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def d2(S, K, T, r, sigma, q=0.0):
    """Compute d2 in the Black-Scholes formula."""
    return d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)


def call_price(S, K, T, r, sigma, q=0.0):
    """Price of a European call option."""
    _d1 = d1(S, K, T, r, sigma, q)
    _d2 = d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.cdf(_d1) - K * np.exp(-r * T) * norm.cdf(_d2)


def put_price(S, K, T, r, sigma, q=0.0):
    """Price of a European put option."""
    _d1 = d1(S, K, T, r, sigma, q)
    _d2 = d2(S, K, T, r, sigma, q)
    return K * np.exp(-r * T) * norm.cdf(-_d2) - S * np.exp(-q * T) * norm.cdf(-_d1)


def delta(S, K, T, r, sigma, q=0.0, option_type="call"):
    """Sensitivity of option price to the underlying price."""
    _d1 = d1(S, K, T, r, sigma, q)
    if option_type == "call":
        return np.exp(-q * T) * norm.cdf(_d1)
    elif option_type == "put":
        return np.exp(-q * T) * (norm.cdf(_d1) - 1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def gamma(S, K, T, r, sigma, q=0.0):
    """Rate of change of delta with respect to the underlying price."""
    _d1 = d1(S, K, T, r, sigma, q)
    return (np.exp(-q * T) * norm.pdf(_d1)) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma, q=0.0):
    """Sensitivity of option price to volatility.

    Returned per 1 percentage point of vol (e.g. sigma 0.20 -> 0.21),
    matching market convention. This is the analytic derivative
    dC/dsigma divided by 100.
    """
    _d1 = d1(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(_d1) * np.sqrt(T) / 100.0


def theta(S, K, T, r, sigma, q=0.0, option_type="call"):
    """Sensitivity of option price to the passage of time.

    Returned per calendar day (annual dT/dT divided by 365).
    Sign convention: this is dV/dT with T = time to expiry, not the
    negated 'decay per day' convention some sources use — a shrinking
    T with typical inputs gives a negative value here for most calls.
    """
    _d1 = d1(S, K, T, r, sigma, q)
    _d2 = d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return (-0.5 * S * np.exp(-q * T) * norm.pdf(_d1) * sigma / np.sqrt(T) - r * K * np.exp(-r * T) * norm.cdf(_d2) + q * S * np.exp(-q * T) * norm.cdf(_d1)) / 365.0
    elif option_type == "put":
        return (-0.5 * S * np.exp(-q * T) * norm.pdf(_d1) * sigma / np.sqrt(T) + r * K * np.exp(-r * T) * norm.cdf(-_d2) - q * S * np.exp(-q * T) * norm.cdf(-_d1)) / 365.0
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def rho(S, K, T, r, sigma, q=0.0, option_type="call"):
    """Sensitivity of option price to the risk-free rate.

    Returned per 1 percentage point of r (analytic dV/dr divided by 100).
    """
    _d2 = d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(_d2) / 100.0
    elif option_type == "put":
        return -K * T * np.exp(-r * T) * norm.cdf(-_d2) / 100.0
    else:
        raise ValueError("option_type must be 'call' or 'put'")