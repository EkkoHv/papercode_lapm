"""Evaluate LAPM on the generated random fields."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from lapm.metrics import regression_metrics
from lapm.model import LAPM, LAPMConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/generated/random_fields"),
    )
    parser.add_argument("--fields", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/random_field_lapm.csv"),
    )
    args = parser.parse_args()

    records: list[dict[str, float | int]] = []
    for field_number in range(1, args.fields + 1):
        path = args.data / f"field_{field_number:03d}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} does not exist. Run workflows/generate_random_fields.py first."
            )

        data = np.load(path)
        training_indices = data["training_indices"]
        test_indices = data["test_indices"]
        model = LAPM(LAPMConfig())
        model.fit(
            data["coordinates"][training_indices],
            data["covariates"][training_indices],
            data["target"][training_indices],
        )
        prediction = model.predict(
            data["coordinates"][test_indices],
            data["covariates"][test_indices],
        )
        _, prediction_sd = model.predict_mc(
            data["coordinates"][test_indices],
            data["covariates"][test_indices],
        )
        metrics = regression_metrics(data["target"][test_indices], prediction)
        records.append(
            {
                "field": field_number,
                **metrics,
                "mean_prediction_sd": float(np.mean(prediction_sd)),
            }
        )
        print(
            f"field={field_number:02d} "
            f"MAE={metrics['mae']:.4f} "
            f"RMSE={metrics['rmse']:.4f} "
            f"R2={metrics['r2']:.4f}"
        )

    result = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    summary = result.drop(columns="field").agg(["mean", "std"])
    summary.to_csv(args.output.with_name(f"{args.output.stem}_summary.csv"))
    print(summary)


if __name__ == "__main__":
    main()
