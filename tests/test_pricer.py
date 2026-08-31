"""
Tests for the Black-Scholes pricer in src/pricer.py.

Covers the three checks required for Phase 1 sign-off:
    - put-call parity holds to machine precision
    - a deep in-the-money call converges to its discounted intrinsic value
    - vega is positive everywhere and peaks near the at-the-money point
"""

import numpy as np
import pytest

from pricer import call_price, put_price, vega


def test_put_call_parity():
    S, K, T, r, sigma, q = 100.0, 95.0, 0.5, 0.03, 0.2, 0.01

    c = call_price(S, K, T, r, sigma, q)
    p = put_price(S, K, T, r, sigma, q)

    parity_rhs = S * np.exp(-q * T) - K * np.exp(-r * T)

    assert c - p == pytest.approx(parity_rhs, abs=1e-10)


def test_deep_itm_call_converges_to_discounted_intrinsic():
    S, T, r, sigma, q = 100.0, 0.5, 0.03, 0.2, 0.01
    K = 1.0  # deep in the money

    c = call_price(S, K, T, r, sigma, q)
    discounted_intrinsic = S * np.exp(-q * T) - K * np.exp(-r * T)

    assert c == pytest.approx(discounted_intrinsic, abs=1e-6)


def test_vega_positive_and_peaks_near_atm():
    S, T, r, sigma, q = 100.0, 0.5, 0.03, 0.2, 0.01
    forward = S * np.exp((r - q) * T)

    strikes = np.linspace(50.0, 150.0, 101)
    vegas = np.array([vega(S, K, T, r, sigma, q) for K in strikes])

    assert np.all(vegas > 0)

    peak_strike = strikes[np.argmax(vegas)]
    assert abs(peak_strike - forward) < 5.0