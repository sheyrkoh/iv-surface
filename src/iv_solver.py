"""
Implied volatility solver.

Inverts the Black-Scholes pricer in src/pricer.py: given an observed option
price, find the volatility that reproduces it.

CONVENTIONS:
    sigma is handled in DECIMAL form throughout this module (0.20 = 20% vol).
    Note that pricer.vega() returns the derivative per 1 percentage point,
    i.e. the analytic dV/dsigma divided by 100. Any use of it as a derivative
    with respect to decimal sigma must be scaled accordingly.

    Failures return float('nan'). This module does not raise on bad market
    data — a single bad quote must not kill a full chain run.
"""

import numpy as np
from scipy.optimize import brentq

import pricer

# Search bracket for sigma, in decimals. 0.1% to 500% annualised.
SIGMA_LOWER = 0.001
SIGMA_UPPER = 5.0


def _price(S, K, T, r, sigma, q, option_type):
    """Dispatch to the correct pricer. Internal helper."""
    if option_type == "call":
        return pricer.call_price(S, K, T, r, sigma, q)
    elif option_type == "put":
        return pricer.put_price(S, K, T, r, sigma, q)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def price_bounds(S, K, T, r, q=0.0, option_type="call"):
    """No-arbitrage price bounds for a European option.

    Returns (lower, upper): the limits of the Black-Scholes price as
    sigma -> 0 and sigma -> infinity respectively. A market price strictly
    inside this interval has a unique implied volatility. A price outside
    it has none.

    Returns:
        tuple of (float, float)
    """
    if option_type == "call":
        lower = max(0.0, S * np.exp(-q * T) - K * np.exp(-r * T))
        upper = S * np.exp(-q * T)
    elif option_type == "put":
        lower = max(0.0, K * np.exp(-r * T) - S * np.exp(-q * T))
        upper = K * np.exp(-r * T)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return lower, upper