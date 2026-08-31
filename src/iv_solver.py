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
import logging

logger = logging.getLogger(__name__)

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

def implied_vol(market_price, S, K, T, r, q=0.0, option_type="call",
                 max_iter=50, tol=1e-8):
    """
    Solve for the Black-Scholes implied volatility that reproduces
    market_price.

    Returns:
        float: implied volatility (decimal, e.g. 0.20 for 20%), or
        float('nan') if no volatility exists that reproduces the price,
        or if the result would not be numerically reliable.
    """
    lower, upper = price_bounds(S, K, T, r, q, option_type)

    if market_price <= lower + 1e-6:
        logger.warning("Market price is below the no-arbitrage bounds: S=%s K=%s T=%s price=%s", S, K, T, market_price)
        return float('nan')
    if market_price >= upper - 1e-6:
        logger.warning("Market price is above the no-arbitrage bounds: S=%s K=%s T=%s price=%s", S, K, T, market_price)
        return float('nan')

    # Newton-Raphson fast path.
    sigma = 0.20  # initial guess

    for i in range(max_iter):
        price = _price(S, K, T, r, sigma, q, option_type)
        diff = price - market_price

        if abs(diff) < tol:
            return sigma

        derivative_ = pricer.vega(S, K, T, r, sigma, q) * 100.0  # convert to decimal sigma derivative

        # No solver can find a reliable solution here — the option's price is
        # genuinely insensitive to volatility, so any sigma "recovered" would
        # be dressing up noise as a number.
 
        if abs(derivative_) < 1e-4:
            logger.warning("Price is insensitive to volatility at this point so no solution exists: S=%s K=%s T=%s sigma=%s", S, K, T, sigma)
            return float('nan')

        # Then also guard: did sigma_new leave [SIGMA_LOWER, SIGMA_UPPER]?
        # If so, that's a sign Newton is misbehaving in a flat region —
        # break rather than trusting the step.

        sigma_new = sigma - diff / derivative_
        if sigma_new < SIGMA_LOWER or sigma_new > SIGMA_UPPER:
            break

        sigma = sigma_new

    # Newton either didn't converge in time or hit the bracket-exit guard
    # Use Brent's method to find a root in [SIGMA_LOWER, SIGMA_UPPER].

    f = lambda sigma: _price(S, K, T, r, sigma, q, option_type) - market_price
    try:
        return brentq(f, SIGMA_LOWER, SIGMA_UPPER)
    except ValueError:
        logger.warning("brentq found no bracket for S=%s, K=%s, T=%s, price=%s", S, K, T, market_price)
        return float('nan')