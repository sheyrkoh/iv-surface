"""
Tests for the implied volatility solver (src/iv_solver.py).

Each test targets one part of the Phase 2 definition of done:
    - the solver recovers a known input vol from a synthetic price to 1e-8
    - the solver fails gracefully (NaN, never raises) on constructed bad inputs
"""

import math
import pytest

import pricer
import iv_solver as iv


def test_recovers_known_vol_newton_path():
    """Well-conditioned ATM case: Newton alone converges, far from the
    initial 0.20 guess, without ever touching Brent."""
    S, K, T, r = 5000, 5000, 1.0, 0.04
    true_sigma = 0.55
    price = pricer.call_price(S, K, T, r, true_sigma)

    recovered = iv.implied_vol(price, S, K, T, r, option_type="call")

    assert not math.isnan(recovered)
    assert abs(recovered - true_sigma) < 1e-8


def test_recovers_known_vol_brent_fallback_path():
    """Deep-ITM, 1-week-to-expiry case: Newton's linear step overshoots
    [SIGMA_LOWER, SIGMA_UPPER] on iteration 0, forcing a handoff to
    brentq. This is the only test that exercises the Brent branch."""
    S, K, T, r = 5000, 4500, 7 / 365, 0.04
    true_sigma = 1.5
    price = pricer.call_price(S, K, T, r, true_sigma)

    recovered = iv.implied_vol(price, S, K, T, r, option_type="call")

    assert not math.isnan(recovered)
    assert abs(recovered - true_sigma) < 1e-8


def test_put_side_also_recovers_known_vol():
    """option_type='put' has never been exercised through the full
    Newton/Brent path before this test -- only calls were checked."""
    S, K, T, r = 5000, 5200, 0.5, 0.04
    true_sigma = 0.35
    price = pricer.put_price(S, K, T, r, true_sigma)

    recovered = iv.implied_vol(price, S, K, T, r, option_type="put")

    assert not math.isnan(recovered)
    assert abs(recovered - true_sigma) < 1e-8


@pytest.mark.parametrize("offset", [0.0, -0.01])
def test_price_at_or_below_lower_bound_returns_nan(offset):
    """A price at, or below, the discounted-intrinsic lower bound has
    no corresponding sigma -- must return NaN, never raise."""
    S, K, T, r = 5000, 5000, 1.0, 0.04
    lower, _ = iv.price_bounds(S, K, T, r, 0.0, "call")

    result = iv.implied_vol(lower + offset, S, K, T, r, option_type="call")

    assert math.isnan(result)


@pytest.mark.parametrize("offset", [0.0, 0.01])
def test_price_at_or_above_upper_bound_returns_nan(offset):
    """A price at, or above, the discounted-spot upper bound has no
    corresponding sigma -- must return NaN."""
    S, K, T, r = 5000, 5000, 1.0, 0.04
    _, upper = iv.price_bounds(S, K, T, r, 0.0, "call")

    result = iv.implied_vol(upper + offset, S, K, T, r, option_type="call")

    assert math.isnan(result)


def test_degenerate_vega_returns_nan():
    """Far-OTM, 1-day-to-expiry call: a root technically exists but
    vega is numerically negligible there. The solver must refuse to
    answer rather than return a number that looks precise but is
    actually noise."""
    S, K, T, r = 5000, 6000, 1 / 365, 0.04
    lower, upper = iv.price_bounds(S, K, T, r, 0.0, "call")
    mid_price = (lower + upper) / 2

    result = iv.implied_vol(mid_price, S, K, T, r, option_type="call")

    assert math.isnan(result)