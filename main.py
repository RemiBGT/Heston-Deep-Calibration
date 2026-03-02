"""Entry point for the Heston deep-surrogate calibration demo.

The script intentionally runs both calibration paths:
1. the exact Heston engine, which is slow but trusted;
2. the neural surrogate, which is fast once trained.

This side-by-side comparison is the clearest way to show the business value of
the surrogate model in a portfolio project.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error

from src.calibration.optimizer import HestonCalibrator
from src.data.generator import HestonDataGenerator
from src.models.heston import HestonPricer
from src.surrogate.nn_model import NeuralNetSurrogate
from src.utils.plotter import Visualizer


def _format_parameters(parameters: dict[str, float]) -> str:
    """Return a compact string representation of calibrated parameters."""
    return ", ".join(f"{name}={value:.4f}" for name, value in parameters.items())


def main() -> None:
    """Run the end-to-end Heston deep-surrogate workflow."""
    np.set_printoptions(precision=6, suppress=True)

    forward = 100.0
    maturity = 1.5
    strike_grid = np.linspace(80.0, 120.0, 9, dtype=float)

    generator = HestonDataGenerator(
        strike_grid=strike_grid,
        test_size=0.2,
        random_state=42,
    )
    dataset = generator.generate_dataset(n_samples=1500)

    surrogate = NeuralNetSurrogate(
        hidden_layer_sizes=(128, 128, 64),
        learning_rate_init=1.0e-3,
        max_iter=1500,
        early_stopping=True,
    )
    surrogate.train(dataset.X_train, dataset.y_train)

    nn_test_predictions = surrogate.predict(dataset.X_test)
    nn_rmse = float(np.sqrt(mean_squared_error(dataset.y_test, nn_test_predictions)))

    true_parameters = {
        "v0": 0.045,
        "kappa": 2.25,
        "theta": 0.035,
        "rho": -0.70,
        "sigma": 0.35,
    }
    target_curve = np.asarray(
        HestonPricer.price_curve(
            forward=forward,
            strikes=strike_grid,
            maturity=maturity,
            v0=true_parameters["v0"],
            kappa=true_parameters["kappa"],
            theta=true_parameters["theta"],
            rho=true_parameters["rho"],
            sigma=true_parameters["sigma"],
        ),
        dtype=float,
    )

    exact_calibrator = HestonCalibrator(
        target_curve=target_curve,
        strikes=strike_grid,
        forward=forward,
        maturity=maturity,
        model=HestonPricer,
    )
    surrogate_calibrator = HestonCalibrator(
        target_curve=target_curve,
        strikes=strike_grid,
        forward=forward,
        maturity=maturity,
        model=surrogate,
    )

    exact_result = exact_calibrator.calibrate()
    surrogate_result = surrogate_calibrator.calibrate()

    speed_up = (
        exact_result.elapsed_time / surrogate_result.elapsed_time
        if surrogate_result.elapsed_time > 0.0
        else float("inf")
    )

    exact_rmse = float(
        np.sqrt(np.mean((exact_result.calibrated_curve - target_curve) ** 2))
    )
    surrogate_rmse_calibration = float(
        np.sqrt(np.mean((surrogate_result.calibrated_curve - target_curve) ** 2))
    )

    print("=" * 72)
    print("Heston Deep Surrogate Modeling with MLP Regression")
    print("=" * 72)
    print(f"Training observations: {dataset.X_train.shape[0]}")
    print(f"Test observations: {dataset.X_test.shape[0]}")
    print(f"Hold-out NN RMSE: {nn_rmse:.8f}")
    print()
    print(f"True parameters      : {_format_parameters(true_parameters)}")
    print(f"Exact calibration    : {_format_parameters(exact_result.parameters)}")
    print(f"NN calibration       : {_format_parameters(surrogate_result.parameters)}")
    print()
    print(f"Exact solver time    : {exact_result.elapsed_time:.6f} seconds")
    print(f"NN solver time       : {surrogate_result.elapsed_time:.6f} seconds")
    print(f"Acceleration         : x{speed_up:.2f}")
    print()
    print(f"Exact curve RMSE     : {exact_rmse:.8f}")
    print(f"NN curve RMSE        : {surrogate_rmse_calibration:.8f}")
    print(f"Exact SSE            : {exact_result.objective_value:.10f}")
    print(f"NN SSE               : {surrogate_result.objective_value:.10f}")

    visualizer = Visualizer()
    visualizer.plot_curve(
        strikes=strike_grid,
        target_curve=target_curve,
        calibrated_curves={
            "Exact": exact_result.calibrated_curve,
            "NN": surrogate_result.calibrated_curve,
        },
    )
    visualizer.plot_convergence(
        histories={
            "Exact": exact_result.convergence_history,
            "NN": surrogate_result.convergence_history,
        }
    )
    exported_paths = visualizer.save_figures("docs/images")
    print()
    print("Exported figures     :", ", ".join(str(path) for path in exported_paths))
    visualizer.show()


if __name__ == "__main__":
    main()
