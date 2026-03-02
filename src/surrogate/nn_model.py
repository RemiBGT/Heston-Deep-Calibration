"""Neural-network surrogate used to approximate Heston option prices.

The production bottleneck in this project is not the optimizer itself.
It is the repeated numerical integration required by the exact Heston pricer.
The role of this module is therefore simple: learn the mapping

    (F, K, T, v0, kappa, theta, rho, sigma) -> call price

once offline, then answer this pricing query in milliseconds during calibration.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import numpy.typing as npt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class NeuralNetSurrogate:
    """Thin wrapper around ``MLPRegressor`` for Heston price approximation.

    A multi-layer perceptron is a pragmatic choice here:
    it is lightweight, easy to train in scikit-learn, and expressive enough
    to learn smooth pricing surfaces when the training data is dense.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (128, 128, 64),
        activation: str = "relu",
        learning_rate_init: float = 1.0e-3,
        max_iter: int = 1500,
        early_stopping: bool = True,
        random_state: int = 42,
    ) -> None:
        """Initialize the neural-network surrogate.

        Args:
            hidden_layer_sizes: Width of each hidden layer.
            activation: Hidden-layer activation function.
            learning_rate_init: Initial Adam learning rate.
            max_iter: Maximum training epochs.
            early_stopping: Whether to stop once validation loss stalls.
            random_state: Seed for reproducibility.
        """
        self._feature_scaler = StandardScaler()
        self._model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver="adam",
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=0.1,
            n_iter_no_change=25,
            random_state=random_state,
        )
        self._is_fitted = False

    def train(
        self,
        X_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.float64],
    ) -> None:
        """Fit the neural network on synthetic Heston prices.

        Neural networks are sensitive to feature scale. In this project the raw
        inputs mix quantities with very different magnitudes:
        spot-like levels around 100, maturities around 1, and variances around 0.01.
        Without normalization, gradient-based training wastes capacity trying to
        compensate for those scale differences instead of learning the pricing map.

        Args:
            X_train: Training features.
            y_train: Training targets.
        """
        X_array = np.asarray(X_train, dtype=float)
        y_array = np.asarray(y_train, dtype=float)

        X_scaled = self._feature_scaler.fit_transform(X_array)
        self._model.fit(X_scaled, y_array)
        self._is_fitted = True

    def predict(
        self,
        X: npt.NDArray[np.float64],
        return_std: bool = False,
    ) -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Predict Heston call prices.

        Args:
            X: Feature matrix.
            return_std: Preserved for interface compatibility with earlier versions.

        Returns:
            The predicted prices. If ``return_std`` is requested, a zero vector is
            returned as a placeholder because a vanilla MLP does not provide
            predictive uncertainty.

        Raises:
            RuntimeError: If the network has not been trained yet.
        """
        if not self._is_fitted:
            raise RuntimeError("The neural-network surrogate must be trained before prediction.")

        X_scaled = self._feature_scaler.transform(np.asarray(X, dtype=float))
        predictions = np.asarray(self._model.predict(X_scaled), dtype=float)

        if return_std:
            return predictions, np.zeros_like(predictions)

        return predictions

    def save_model(self, filepath: str | Path) -> None:
        """Persist the trained scaler and neural network with joblib.

        Args:
            filepath: Destination path for the serialized model.

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if not self._is_fitted:
            raise RuntimeError("The neural-network surrogate must be trained before saving.")

        payload = {
            "feature_scaler": self._feature_scaler,
            "model": self._model,
            "is_fitted": self._is_fitted,
        }
        joblib.dump(payload, Path(filepath))

    @classmethod
    def load_model(cls, filepath: str | Path) -> "NeuralNetSurrogate":
        """Reload a previously trained surrogate from disk.

        Args:
            filepath: Path to the serialized joblib file.

        Returns:
            A fitted ``NeuralNetSurrogate`` instance.
        """
        payload = joblib.load(Path(filepath))
        instance = cls()
        instance._feature_scaler = payload["feature_scaler"]
        instance._model = payload["model"]
        instance._is_fitted = bool(payload["is_fitted"])
        return instance
