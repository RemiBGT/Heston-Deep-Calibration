"""Visualization helpers for Heston surrogate calibration.

This module keeps plotting concerns separate from modeling code so the pricing
and calibration logic stays readable. That separation also makes it easier to
reuse the visual output directly in a GitHub README.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


class Visualizer:
    """Plot calibration curves and optimizer diagnostics."""

    def __init__(self) -> None:
        """Track created figures so they can be exported consistently."""
        self._figures: dict[str, plt.Figure] = {}

    def plot_curve(
        self,
        strikes: npt.NDArray[np.float64],
        target_curve: npt.NDArray[np.float64],
        calibrated_curves: Mapping[str, npt.NDArray[np.float64]],
    ) -> plt.Figure:
        """Plot the target price curve against calibrated curves.

        Args:
            strikes: Strike grid.
            target_curve: Market or synthetic target option price curve.
            calibrated_curves: Mapping from label to calibrated curve.

        Returns:
            The matplotlib figure object.
        """
        figure, axis = plt.subplots(figsize=(10, 6))
        axis.plot(
            strikes,
            target_curve,
            marker="o",
            linewidth=2.5,
            label="Target Curve",
            color="black",
        )

        for label, curve in calibrated_curves.items():
            axis.plot(
                strikes,
                np.asarray(curve, dtype=float),
                marker="x",
                linewidth=1.8,
                linestyle="--",
                label=f"{label} Calibration",
            )

        axis.set_title("Heston Option Price Curve Calibration")
        axis.set_xlabel("Strike")
        axis.set_ylabel("Call Price")
        axis.grid(True, linestyle=":", alpha=0.6)
        axis.legend()
        figure.tight_layout()
        self._figures["calibration_curve"] = figure
        return figure

    def plot_convergence(
        self,
        histories: Mapping[str, Sequence[float]],
    ) -> plt.Figure:
        """Plot optimizer convergence histories.

        Args:
            histories: Mapping from label to objective value history.

        Returns:
            The matplotlib figure object.
        """
        figure, axis = plt.subplots(figsize=(10, 6))

        for label, history in histories.items():
            iterations = np.arange(len(history), dtype=int)
            objective_path = np.maximum(np.asarray(history, dtype=float), 1.0e-16)
            axis.plot(
                iterations,
                objective_path,
                linewidth=2.0,
                label=label,
            )

        axis.set_title("Calibration Convergence History")
        axis.set_xlabel("Iteration")
        axis.set_ylabel("Sum of Squared Errors")
        axis.set_yscale("log")
        axis.grid(True, linestyle=":", alpha=0.6)
        axis.legend()
        figure.tight_layout()
        self._figures["convergence_history"] = figure
        return figure

    def save_figures(self, dir_path: str) -> list[Path]:
        """Persist all generated figures as PNG files.

        This is useful for a portfolio workflow: the same charts used during
        local development can be exported automatically and dropped into the
        README without any manual screenshotting.

        Args:
            dir_path: Target directory where PNG files will be written.

        Returns:
            The list of generated file paths.
        """
        output_dir = Path(dir_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: list[Path] = []
        for figure_name, figure in self._figures.items():
            output_path = output_dir / f"{figure_name}.png"
            figure.savefig(output_path, dpi=200, bbox_inches="tight")
            saved_paths.append(output_path)

        return saved_paths

    @staticmethod
    def show() -> None:
        """Display all prepared figures."""
        plt.show()
