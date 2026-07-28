"""Generate the twenty random fields and their buffered interior splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lapm.random_fields import RandomFieldConfig, generate_field, make_interior_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/generated/random_fields"),
    )
    parser.add_argument("--fields", type=int, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = RandomFieldConfig(number_of_fields=args.fields)
    manifest: list[dict[str, object]] = []
    for field_number in range(1, config.number_of_fields + 1):
        field = generate_field(field_number, config)
        split = make_interior_split(field["coordinates"], field_number, config)
        path = args.output / f"field_{field_number:03d}.npz"
        np.savez_compressed(
            path,
            coordinates=field["coordinates"],
            covariates=field["covariates"],
            target=field["target"],
            latent_target=field["latent_target"],
            coefficient_fields=field["coefficient_fields"],
            axis=field["axis"],
            training_indices=split["training_indices"],
            calibration_indices=split["calibration_indices"],
            test_indices=split["test_indices"],
        )
        manifest.append(
            {
                "field_number": field_number,
                "source_field_id": field["source_field_id"],
                "seed": field["seed"],
                "training_locations": len(split["training_indices"]),
                "calibration_locations": len(split["calibration_indices"]),
                "test_locations": len(split["test_indices"]),
                "minimum_test_to_training_distance": split[
                    "test_to_training_minimum"
                ],
                "file": path.name,
            }
        )
    pd.DataFrame(manifest).to_csv(args.output / "manifest.csv", index=False)
    (args.output / "generation_config.json").write_text(
        json.dumps(config.__dict__, indent=2),
        encoding="utf-8",
    )
    print(f"Generated {len(manifest)} fields in {args.output.resolve()}")


if __name__ == "__main__":
    main()
