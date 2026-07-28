"""Train LAPM on a CSV file and predict at supplied locations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from lapm.io import load_spatial_csv
from lapm.model import LAPM, LAPMConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--predict", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--covariates", nargs="+", required=True)
    parser.add_argument("--x-column", default="x")
    parser.add_argument("--y-column", default="y")
    parser.add_argument("--output", type=Path, default=Path("results/lapm_predictions.csv"))
    args = parser.parse_args()

    coordinates, covariates, target = load_spatial_csv(
        args.train,
        args.target,
        args.covariates,
        args.x_column,
        args.y_column,
    )
    prediction_frame = pd.read_csv(args.predict)
    prediction_coordinates = prediction_frame[
        [args.x_column, args.y_column]
    ].to_numpy(dtype=float)
    prediction_covariates = prediction_frame[args.covariates].to_numpy(dtype=float)
    model = LAPM(LAPMConfig())
    model.fit(coordinates, covariates, target)
    prediction, standard_deviation = model.predict_mc(
        prediction_coordinates,
        prediction_covariates,
    )
    output = prediction_frame.copy()
    output["prediction"] = prediction
    output["prediction_sd"] = standard_deviation
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Predictions written to {args.output.resolve()}")
    print(model.fitted_settings)


if __name__ == "__main__":
    main()
