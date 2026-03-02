"""Basic regression tests for the exact Heston pricer."""

from __future__ import annotations

import pytest

from src.models.heston import HestonPricer


@pytest.mark.parametrize(
    ("forward", "strike", "maturity", "v0", "kappa", "theta", "rho", "sigma"),
    [
        (100.0, 80.0, 1.0, 0.04, 2.0, 0.04, -0.6, 0.3),
        (100.0, 100.0, 1.5, 0.05, 1.8, 0.05, -0.5, 0.35),
        (100.0, 120.0, 2.0, 0.03, 2.5, 0.04, -0.4, 0.25),
    ],
)
def test_heston_call_price_respects_basic_arbitrage_bounds(
    forward: float,
    strike: float,
    maturity: float,
    v0: float,
    kappa: float,
    theta: float,
    rho: float,
    sigma: float,
) -> None:
    """The exact call price should stay between trivial lower and upper bounds."""
    call_price = HestonPricer.call_price(
        forward=forward,
        strike=strike,
        maturity=maturity,
        v0=v0,
        kappa=kappa,
        theta=theta,
        rho=rho,
        sigma=sigma,
    )

    lower_bound = max(forward - strike, 0.0)
    upper_bound = forward

    assert call_price >= lower_bound
    assert call_price <= upper_bound
