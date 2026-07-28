"""Prediction metrics used by the experiments."""

from __future__ import annotations

import numpy as np


def regression_metrics(measured: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Return MAE, RMSE, and the coefficient of determination."""

    measured = np.asarray(measured, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()
    error = predicted - measured
    residual_sum = float(np.sum(error**2))
    total_sum = float(np.sum((measured - measured.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(1.0 - residual_sum / total_sum) if total_sum > 0.0 else float("nan"),
    }
