"""Synthetic data generation for the Heston surrogate.

The network is only as good as the synthetic surface we show it during
training. This module samples economically plausible Heston parameters, prices
curves with the exact engine, and reshapes the result into a supervised
learning dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import train_test_split

from src.models.heston import HestonPricer


@dataclass(frozen=True)
class HestonDatasetSplit:
    """Container for both curve-level and pointwise train/test datasets."""

    X_train: npt.NDArray[np.float64]
    X_test: npt.NDArray[np.float64]
    y_train: npt.NDArray[np.float64]
    y_test: npt.NDArray[np.float64]
    parameters_train: npt.NDArray[np.float64]
    parameters_test: npt.NDArray[np.float64]
    curves_train: npt.NDArray[np.float64]
    curves_test: npt.NDArray[np.float64]


class HestonDataGenerator:
    """Generate synthetic Heston call price curves for supervised learning."""

    FEATURE_NAMES: tuple[str, ...] = (
        "forward",
        "strike",
        "maturity",
        "v0",
        "kappa",
        "theta",
        "rho",
        "sigma",
    )
    PARAMETER_NAMES: tuple[str, ...] = (
        "forward",
        "maturity",
        "v0",
        "kappa",
        "theta",
        "rho",
        "sigma",
    )

    def __init__(
        self,
        strike_grid: npt.NDArray[np.float64],
        test_size: float = 0.2,
        random_state: int = 42,
        parameter_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Initialize the synthetic data generator.

        Args:
            strike_grid: Grid of strikes used to build each curve.
            test_size: Proportion of samples reserved for the test split.
            random_state: Random seed for reproducibility.
            parameter_bounds: Optional sampling bounds for Heston inputs.
        """
        if np.any(np.asarray(strike_grid) <= 0.0):
            raise ValueError("All strikes in the strike grid must be positive.")
        if not 0.0 < test_size < 1.0:
            raise ValueError("The test size must lie in (0, 1).")

        self.strike_grid = np.asarray(strike_grid, dtype=float)
        self.test_size = test_size
        self.random_state = random_state
        self.parameter_bounds = parameter_bounds or {
            "forward": (80.0, 120.0),
            "maturity": (0.25, 3.0),
            "v0": (0.01, 0.09),
            "kappa": (0.5, 4.0),
            "theta": (0.01, 0.09),
            "rho": (-0.9, -0.05),
            "sigma": (0.1, 0.8),
        }
        self._rng = np.random.default_rng(seed=random_state)

    @staticmethod
    def _is_admissible(parameter_row: npt.NDArray[np.float64]) -> bool:
        """Return whether a sampled Heston parameter vector is numerically safe.

        The Feller condition ``2 * kappa * theta > sigma^2`` is not a hard
        mathematical requirement for every use of Heston, but it is a very
        practical stability filter. Enforcing it during data generation avoids
        feeding the neural network with pathological regions where the variance
        process gets too close to zero and the pricing surface becomes less
        well-behaved numerically.
        """
        _, _, v0, kappa, theta, rho, sigma = parameter_row
        return bool(
            v0 > 0.0
            and kappa > 0.0
            and theta > 0.0
            and sigma > 0.0
            and -0.999 < rho < 0.999
            and 2.0 * kappa * theta > sigma**2
        )

    def sample_parameters(self, n_samples: int) -> npt.NDArray[np.float64]:
        """Sample random Heston parameter combinations.

        Args:
            n_samples: Number of independent parameter vectors to generate.

        Returns:
            A matrix of shape ``(n_samples, 7)`` with columns ordered as
            ``(forward, maturity, v0, kappa, theta, rho, sigma)``.
        """
        if n_samples <= 1:
            raise ValueError("At least two samples are required to form train/test splits.")

        accepted_rows: list[npt.NDArray[np.float64]] = []
        while len(accepted_rows) < n_samples:
            # We oversample candidate rows, then keep only the admissible ones.
            # This is simpler than trying to sample directly under a nonlinear
            # constraint such as the Feller condition.
            columns: list[npt.NDArray[np.float64]] = []
            for name in self.PARAMETER_NAMES:
                lower_bound, upper_bound = self.parameter_bounds[name]
                sampled_column = self._rng.uniform(
                    low=lower_bound,
                    high=upper_bound,
                    size=n_samples,
                )
                columns.append(sampled_column.astype(float))

            candidate_rows = np.column_stack(columns).astype(float)
            for row in candidate_rows:
                if self._is_admissible(row):
                    accepted_rows.append(row)
                    if len(accepted_rows) == n_samples:
                        break

        return np.vstack(accepted_rows).astype(float)

    def generate_curves(
        self,
        parameter_sets: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Generate an option price curve for each parameter vector.

        Args:
            parameter_sets: Matrix with columns
                ``(forward, maturity, v0, kappa, theta, rho, sigma)``.

        Returns:
            A matrix of shape ``(n_samples, n_strikes)`` containing call prices.
        """
        # This loop is intentionally expensive. It creates the exact labels used
        # to teach the neural network how the Heston engine behaves.
        price_rows: list[npt.NDArray[np.float64]] = []
        for forward, maturity, v0, kappa, theta, rho, sigma in parameter_sets:
            curve = HestonPricer.price_curve(
                forward=forward,
                strikes=self.strike_grid,
                maturity=maturity,
                v0=v0,
                kappa=kappa,
                theta=theta,
                rho=rho,
                sigma=sigma,
            )
            price_rows.append(np.asarray(curve, dtype=float))

        return np.vstack(price_rows).astype(float)

    def build_supervised_dataset(
        self,
        parameter_sets: npt.NDArray[np.float64],
        curves: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Transform price curves into pointwise supervised learning samples.

        Args:
            parameter_sets: Matrix with columns
                ``(forward, maturity, v0, kappa, theta, rho, sigma)``.
            curves: Matrix of call prices, one curve per row.

        Returns:
            A tuple ``(X, y)`` where ``X`` contains 8 features per row
            and ``y`` contains the corresponding call price.
        """
        n_samples = parameter_sets.shape[0]
        n_strikes = self.strike_grid.size

        # The MLP does not learn "a whole curve" in one shot. It learns a local
        # pricing rule for one strike at a time, which is why we flatten each
        # curve into independent supervised observations.
        repeated_parameters = np.repeat(parameter_sets, repeats=n_strikes, axis=0)
        tiled_strikes = np.tile(self.strike_grid, reps=n_samples)

        X = np.column_stack(
            (
                repeated_parameters[:, 0],
                tiled_strikes,
                repeated_parameters[:, 1],
                repeated_parameters[:, 2],
                repeated_parameters[:, 3],
                repeated_parameters[:, 4],
                repeated_parameters[:, 5],
                repeated_parameters[:, 6],
            )
        ).astype(float)
        y = curves.reshape(-1).astype(float)
        return X, y

    def generate_dataset(self, n_samples: int) -> HestonDatasetSplit:
        """Create a full train/test dataset for Heston surrogate modeling.

        Args:
            n_samples: Number of independent Heston parameter vectors to sample.

        Returns:
            A train/test split at both curve level and flattened sample level.
        """
        parameter_sets = self.sample_parameters(n_samples=n_samples)
        curves = self.generate_curves(parameter_sets=parameter_sets)

        # We split at the curve level first. This prevents the same parameter
        # vector from leaking into both train and test sets through different
        # strikes, which would make out-of-sample metrics artificially flattering.
        (
            parameters_train,
            parameters_test,
            curves_train,
            curves_test,
        ) = train_test_split(
            parameter_sets,
            curves,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True,
        )

        X_train, y_train = self.build_supervised_dataset(
            parameter_sets=parameters_train,
            curves=curves_train,
        )
        X_test, y_test = self.build_supervised_dataset(
            parameter_sets=parameters_test,
            curves=curves_test,
        )

        return HestonDatasetSplit(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            parameters_train=parameters_train.astype(float),
            parameters_test=parameters_test.astype(float),
            curves_train=curves_train.astype(float),
            curves_test=curves_test.astype(float),
        )
