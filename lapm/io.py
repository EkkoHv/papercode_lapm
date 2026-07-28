"""Input helpers for public or user-supplied spatial data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_spatial_csv(
    path: str | Path,
    target: str,
    covariates: list[str],
    x_column: str = "x",
    y_column: str = "y",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load coordinates, covariates, and target values from a CSV file."""

    frame = pd.read_csv(path)
    required = [x_column, y_column, target, *covariates]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    values = frame[required].apply(pd.to_numeric, errors="raise")
    if values[[x_column, y_column, target]].isna().any().any():
        raise ValueError("Coordinates and target values cannot contain missing values.")
    coordinates = values[[x_column, y_column]].to_numpy(dtype=float)
    ancillary = values[covariates].to_numpy(dtype=float)
    response = values[target].to_numpy(dtype=float)
    return coordinates, ancillary, response
