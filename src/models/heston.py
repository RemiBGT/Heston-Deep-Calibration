"""Semi-analytic Heston pricing routines.

This module intentionally uses direct numerical quadrature. It is not the
fastest possible Fourier method, but it is easy to read and easy to explain.
That makes it a good reference implementation for a portfolio project.

In production, desks often move to faster transforms such as Carr-Madan FFT
when throughput matters. Here, we keep plain quadrature because its slowness is
precisely what creates the business case for a surrogate model.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from scipy.integrate import quad


class HestonPricer:
    """Price European calls under the Heston stochastic volatility model."""

    @staticmethod
    def _validate_inputs(
        forward: float,
        strike: float,
        maturity: float,
        v0: float,
        kappa: float,
        theta: float,
        rho: float,
        sigma: float,
    ) -> None:
        """Validate the Heston parameter domain.

        These checks are intentionally strict. A pricing engine should fail fast
        on nonsensical inputs rather than bury unstable numbers downstream.
        """
        if forward <= 0.0:
            raise ValueError("The forward must be strictly positive.")
        if strike <= 0.0:
            raise ValueError("The strike must be strictly positive.")
        if maturity <= 0.0:
            raise ValueError("The maturity must be strictly positive.")
        if v0 <= 0.0:
            raise ValueError("The initial variance must be strictly positive.")
        if kappa <= 0.0:
            raise ValueError("The mean-reversion speed must be strictly positive.")
        if theta <= 0.0:
            raise ValueError("The long-run variance must be strictly positive.")
        if sigma <= 0.0:
            raise ValueError("The vol-of-vol must be strictly positive.")
        if not -0.999 < rho < 0.999:
            raise ValueError("The correlation must lie in (-0.999, 0.999).")

    @staticmethod
    def _characteristic_function(
        phi: complex,
        forward: float,
        strike: float,
        maturity: float,
        v0: float,
        kappa: float,
        theta: float,
        rho: float,
        sigma: float,
    ) -> complex:
        """Return the Heston log-price characteristic function.

        The characteristic function is the core analytic object in Heston.
        Instead of simulating paths, we work in Fourier space and recover option
        prices by integrating a transformed payoff.
        """
        imaginary_unit = 1j
        log_moneyness = math.log(forward / strike)

        b = kappa - rho * sigma * imaginary_unit * phi
        d = np.sqrt(b**2 + sigma**2 * (phi**2 + imaginary_unit * phi))
        g = (b - d) / (b + d)

        exp_term = np.exp(-d * maturity)
        c_term = (
            kappa
            * (
                (b - d) * maturity
                - 2.0 * np.log((1.0 - g * exp_term) / (1.0 - g))
            )
            / sigma**2
        )
        d_term = (
            (b - d)
            * (1.0 - exp_term)
            / (sigma**2 * (1.0 - g * exp_term))
        )

        return np.exp(
            c_term * theta
            + d_term * v0
            + imaginary_unit * phi * log_moneyness
        )

    @classmethod
    def call_price(
        cls,
        forward: float,
        strike: float,
        maturity: float,
        v0: float,
        kappa: float,
        theta: float,
        rho: float,
        sigma: float,
        integration_upper_bound: float = 100.0,
        integration_limit: int = 200,
    ) -> float:
        """Price a European call via Fourier inversion.

        Args:
            forward: Forward level of the underlying.
            strike: Option strike.
            maturity: Time to maturity in years.
            v0: Initial variance.
            kappa: Mean-reversion speed of variance.
            theta: Long-run variance level.
            rho: Correlation between spot and variance shocks.
            sigma: Volatility of variance.
            integration_upper_bound: Truncation bound for the Fourier integral.
            integration_limit: Maximum number of adaptive quadrature sub-intervals.

        Returns:
            The undiscounted call price under the forward measure.
        """
        cls._validate_inputs(
            forward=forward,
            strike=strike,
            maturity=maturity,
            v0=v0,
            kappa=kappa,
            theta=theta,
            rho=rho,
            sigma=sigma,
        )

        def integrand(phi: float) -> float:
            # ``quad`` performs adaptive numerical quadrature. It is robust and
            # readable, but it repeatedly evaluates the characteristic function.
            # That repeated work is exactly why exact Heston calibration is slow.
            characteristic_value = cls._characteristic_function(
                phi=phi - 0.5j,
                forward=forward,
                strike=strike,
                maturity=maturity,
                v0=v0,
                kappa=kappa,
                theta=theta,
                rho=rho,
                sigma=sigma,
            )
            return float(characteristic_value.real / (phi**2 + 0.25))

        integral_value, _ = quad(
            integrand,
            0.0,
            integration_upper_bound,
            limit=integration_limit,
            epsabs=1.0e-8,
            epsrel=1.0e-8,
        )

        raw_price = forward - np.sqrt(forward * strike) * integral_value / np.pi
        intrinsic_value = max(forward - strike, 0.0)
        return float(max(raw_price, intrinsic_value))

    @classmethod
    def price_curve(
        cls,
        forward: float,
        strikes: npt.NDArray[np.float64],
        maturity: float,
        v0: float,
        kappa: float,
        theta: float,
        rho: float,
        sigma: float,
        integration_upper_bound: float = 100.0,
        integration_limit: int = 200,
    ) -> npt.NDArray[np.float64]:
        """Price a full strike curve with repeated Fourier integrations.

        Args:
            forward: Forward level of the underlying.
            strikes: Strike grid.
            maturity: Time to maturity in years.
            v0: Initial variance.
            kappa: Mean-reversion speed of variance.
            theta: Long-run variance level.
            rho: Correlation between spot and variance shocks.
            sigma: Volatility of variance.
            integration_upper_bound: Truncation bound for each integral.
            integration_limit: Maximum quadrature sub-intervals per strike.

        Returns:
            A vector of call prices aligned with the strike grid.

        Each strike triggers its own numerical integral. This is simple to
        implement, but computationally expensive, which makes it a clean target
        for surrogate modeling.
        """
        strike_array = np.asarray(strikes, dtype=float)
        prices = [
            cls.call_price(
                forward=forward,
                strike=float(strike),
                maturity=maturity,
                v0=v0,
                kappa=kappa,
                theta=theta,
                rho=rho,
                sigma=sigma,
                integration_upper_bound=integration_upper_bound,
                integration_limit=integration_limit,
            )
            for strike in strike_array
        ]
        return np.asarray(prices, dtype=float)
