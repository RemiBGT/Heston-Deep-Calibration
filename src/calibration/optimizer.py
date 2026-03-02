"""Calibration routines for Heston option price curves.

The optimizer does not care whether prices come from the exact Heston engine or
from a learned surrogate. That separation is deliberate: it lets us compare the
two calibration loops with the same objective function and the same optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from src.models.heston import HestonPricer
from src.surrogate.nn_model import NeuralNetSurrogate


@dataclass(frozen=True)
class CalibrationResult:
    """Structured output returned by a calibration run."""

    parameters: dict[str, float]
    objective_value: float
    success: bool
    message: str
    elapsed_time: float
    iterations: int
    convergence_history: list[float]
    calibrated_curve: npt.NDArray[np.float64]


class HestonCalibrator:
    """Calibrate Heston parameters against a target option price curve."""

    PARAMETER_NAMES: tuple[str, ...] = ("v0", "kappa", "theta", "rho", "sigma")

    def __init__(
        self,
        target_curve: npt.NDArray[np.float64],
        strikes: npt.NDArray[np.float64],
        forward: float,
        maturity: float,
        model: NeuralNetSurrogate | type[HestonPricer],
        initial_guess: npt.NDArray[np.float64] | None = None,
        bounds: list[tuple[float, float]] | None = None,
    ) -> None:
        """Initialize the calibrator.

        Args:
            target_curve: Market option price curve to fit.
            strikes: Strike grid associated with the target curve.
            forward: Market forward corresponding to the curve.
            maturity: Time to maturity in years.
            model: Either the exact ``HestonPricer`` class or a trained surrogate.
            initial_guess: Initial guess for ``(v0, kappa, theta, rho, sigma)``.
            bounds: Box constraints for the optimizer.
        """
        self.target_curve = np.asarray(target_curve, dtype=float)
        self.strikes = np.asarray(strikes, dtype=float)
        self.forward = float(forward)
        self.maturity = float(maturity)
        self.model = model
        self.initial_guess = np.asarray(
            initial_guess
            if initial_guess is not None
            else [0.04, 2.00, 0.05, -0.50, 0.30],
            dtype=float,
        )
        self.bounds = bounds or [
            (1.0e-4, 0.5),
            (0.1, 6.0),
            (1.0e-4, 0.5),
            (-0.999, 0.999),
            (1.0e-3, 2.0),
        ]

        if self.target_curve.shape != self.strikes.shape:
            raise ValueError("The strike grid and target curve must have the same shape.")

        self._strike_weights = self._build_strike_weights()

    def _build_strike_weights(self) -> npt.NDArray[np.float64]:
        """Build ATM-focused weights used inside the calibration loss.

        In practice, the at-the-money region is usually the most informative and
        the most liquid part of the surface. It often anchors quoted implied
        volatilities, hedging inputs, and the overall shape of the smile used by
        traders and structurers. For that reason, we deliberately penalize ATM
        mispricing more than deep in/out-of-the-money noise.

        Returns:
            A normalized Gaussian-shaped weight vector centered on the forward.
        """
        width = max(0.10 * self.forward, 1.0)
        scaled_distance = (self.strikes - self.forward) / width
        raw_weights = np.exp(-0.5 * scaled_distance**2)

        # Normalizing the weights to an average of 1 keeps the SSE comparable
        # across strike grids while still changing the optimizer's priorities.
        return (raw_weights / np.mean(raw_weights)).astype(float)

    @staticmethod
    def _is_admissible(parameters: npt.NDArray[np.float64]) -> bool:
        """Return whether the Heston vector is inside a stable parameter region.

        We enforce the Feller condition here as well, not because the optimizer
        would crash instantly without it, but because unconstrained searches can
        waste many iterations in regions where the variance dynamics become less
        stable and the calibration target becomes harder to interpret.
        """
        v0, kappa, theta, rho, sigma = np.asarray(parameters, dtype=float)
        return bool(
            v0 > 0.0
            and kappa > 0.0
            and theta > 0.0
            and sigma > 0.0
            and -0.999 < rho < 0.999
            and 2.0 * kappa * theta > sigma**2
        )

    def _predict_curve(self, parameters: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute the model-implied price curve for a given parameter vector."""
        v0, kappa, theta, rho, sigma = np.asarray(parameters, dtype=float)

        if isinstance(self.model, NeuralNetSurrogate):
            n_points = self.strikes.size
            features = np.column_stack(
                (
                    np.full(n_points, self.forward, dtype=float),
                    self.strikes,
                    np.full(n_points, self.maturity, dtype=float),
                    np.full(n_points, v0, dtype=float),
                    np.full(n_points, kappa, dtype=float),
                    np.full(n_points, theta, dtype=float),
                    np.full(n_points, rho, dtype=float),
                    np.full(n_points, sigma, dtype=float),
                )
            )
            predictions = self.model.predict(features)
            return np.asarray(predictions, dtype=float)

        exact_curve = HestonPricer.price_curve(
            forward=self.forward,
            strikes=self.strikes,
            maturity=self.maturity,
            v0=v0,
            kappa=kappa,
            theta=theta,
            rho=rho,
            sigma=sigma,
        )
        return np.asarray(exact_curve, dtype=float)

    def _objective(self, parameters: npt.NDArray[np.float64]) -> float:
        """Return the sum of squared errors between target and model curve.

        A large penalty is cheaper than a hard optimizer failure. It keeps the
        search inside a meaningful part of parameter space while preserving a
        smooth interface for ``scipy.optimize.minimize``.
        """
        if not self._is_admissible(parameters):
            return 1.0e12

        residuals = self._predict_curve(parameters) - self.target_curve
        weighted_squared_errors = self._strike_weights * residuals**2
        return float(np.sum(weighted_squared_errors))

    def calibrate(self, maxiter: int = 200) -> CalibrationResult:
        """Run Heston calibration with L-BFGS-B.

        Args:
            maxiter: Maximum number of optimizer iterations.

        Returns:
            A ``CalibrationResult`` with calibrated parameters and diagnostics.
        """
        # Tracking the loss path is useful for two reasons:
        # it helps diagnose poor initial guesses, and it gives us a clean chart
        # for communicating optimizer behavior in the README.
        convergence_history: list[float] = [self._objective(self.initial_guess)]

        def callback(current_parameters: npt.NDArray[np.float64]) -> None:
            convergence_history.append(self._objective(current_parameters))

        start_time = perf_counter()
        optimization_result = minimize(
            fun=self._objective,
            x0=self.initial_guess,
            method="L-BFGS-B",
            bounds=self.bounds,
            callback=callback,
            options={"maxiter": maxiter, "ftol": 1.0e-12},
        )
        elapsed_time = perf_counter() - start_time

        optimal_parameters = np.asarray(optimization_result.x, dtype=float)
        calibrated_curve = self._predict_curve(optimal_parameters)
        parameter_map = {
            name: float(value)
            for name, value in zip(self.PARAMETER_NAMES, optimal_parameters, strict=True)
        }

        return CalibrationResult(
            parameters=parameter_map,
            objective_value=float(optimization_result.fun),
            success=bool(optimization_result.success),
            message=str(optimization_result.message),
            elapsed_time=float(elapsed_time),
            iterations=int(optimization_result.nit),
            convergence_history=convergence_history,
            calibrated_curve=calibrated_curve.astype(float),
        )


SmileCalibrator = HestonCalibrator
