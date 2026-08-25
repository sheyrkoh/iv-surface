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

UNIT CONVENTIONS - decide and document before implementing the Greeks:
    vega  - TODO: per 1.00 of vol, or per 1 percentage point (i.e. / 100)?
    theta - TODO: per year, or per calendar day (i.e. / 365)?
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
    raise NotImplementedError


def gamma(S, K, T, r, sigma, q=0.0):
    """Rate of change of delta with respect to the underlying price."""
    raise NotImplementedError


def vega(S, K, T, r, sigma, q=0.0):
    """Sensitivity of option price to volatility."""
    raise NotImplementedError


def theta(S, K, T, r, sigma, q=0.0, option_type="call"):
    """Sensitivity of option price to the passage of time."""
    raise NotImplementedError


def rho(S, K, T, r, sigma, q=0.0, option_type="call"):
    """Sensitivity of option price to the risk-free rate."""
    raise NotImplementedError