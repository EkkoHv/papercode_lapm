"""Generate the twenty independent random fields used for validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


@dataclass(frozen=True)
class RandomFieldConfig:
    """Settings for the confirmatory random-field experiment."""

    master_seed: int = 20260715
    number_of_fields: int = 20
    source_field_id_offset: int = 64
    grid_size: int = 51
    extent: float = 100.0
    training_sample_size: int = 360
    calibration_fraction: float = 0.20
    test_fraction_of_extent: float = 0.20
    buffer_width: float = 6.0
    coefficient_range: float = 42.0
    coefficient_strength: float = 0.95
    residual_weight: float = 0.08
    observation_noise: float = 0.04


BLOCK_CENTERS = (
    (25.0, 25.0),
    (50.0, 25.0),
    (75.0, 25.0),
    (25.0, 50.0),
    (50.0, 50.0),
    (75.0, 50.0),
    (25.0, 75.0),
    (50.0, 75.0),
    (75.0, 75.0),
)


def standardize(values: np.ndarray) -> np.ndarray:
    """Center a field and scale it to unit population standard deviation."""

    values = np.asarray(values, dtype=float)
    return (values - values.mean()) / max(values.std(ddof=0), 1.0e-12)


def spectral_matern_field(
    grid_size: int,
    extent: float,
    length_scale: float,
    smoothness: float,
    anisotropy_ratio: float,
    angle_degrees: float,
    generator: np.random.Generator,
) -> np.ndarray:
    """Approximate a Matérn-like field by padded spectral filtering."""

    padded_size = int(2 ** np.ceil(np.log2(grid_size * 2)))
    spacing = extent / (grid_size - 1)
    frequencies = 2.0 * np.pi * np.fft.fftfreq(padded_size, d=spacing)
    kx, ky = np.meshgrid(frequencies, frequencies, indexing="xy")
    angle = np.deg2rad(angle_degrees)
    major = np.cos(angle) * kx + np.sin(angle) * ky
    minor = -np.sin(angle) * kx + np.cos(angle) * ky
    major_scale = max(length_scale, spacing)
    minor_scale = max(length_scale * anisotropy_ratio, spacing)
    radial_frequency = (major_scale * major) ** 2 + (minor_scale * minor) ** 2
    spectrum = (1.0 + radial_frequency) ** (-(smoothness + 1.0))
    spectrum[0, 0] = 0.0
    white_noise = generator.normal(size=(padded_size, padded_size))
    filtered = np.fft.ifft2(np.fft.fft2(white_noise) * np.sqrt(spectrum)).real
    start = (padded_size - grid_size) // 2
    return standardize(filtered[start : start + grid_size, start : start + grid_size])


def generate_field(
    field_number: int,
    config: RandomFieldConfig | None = None,
) -> dict[str, np.ndarray | float | int]:
    """Generate one frozen confirmatory field.

    The offset reproduces source field identifiers 65 through 84 from the
    manuscript experiments while exposing public field numbers 1 through 20.
    """

    settings = config or RandomFieldConfig()
    if not 1 <= field_number <= settings.number_of_fields:
        raise ValueError(f"field_number must be in 1..{settings.number_of_fields}")
    source_field_id = settings.source_field_id_offset + field_number
    seed = int(settings.master_seed + 32452843 * source_field_id)
    generator = np.random.default_rng(seed)
    axis = np.linspace(0.0, settings.extent, settings.grid_size)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    coordinates = np.column_stack([xx.ravel(), yy.ravel()])

    covariate_fields = [
        spectral_matern_field(
            settings.grid_size,
            settings.extent,
            length_scale,
            1.0 if index < 2 else 0.5,
            1.0 if index == 0 else 0.75,
            angle,
            generator,
        )
        for index, (length_scale, angle) in enumerate(
            ((17.0, 0.0), (25.0, 25.0), (13.0, -30.0))
        )
    ]
    coefficient_fields = []
    for baseline in (0.45, -0.30, 0.25):
        variation = spectral_matern_field(
            settings.grid_size,
            settings.extent,
            settings.coefficient_range,
            1.5,
            1.0,
            0.0,
            generator,
        )
        coefficient_fields.append(
            baseline + settings.coefficient_strength * variation
        )

    covariate_contribution = sum(
        coefficient * covariate
        for coefficient, covariate in zip(
            coefficient_fields,
            covariate_fields,
            strict=True,
        )
    )
    covariate_contribution = standardize(covariate_contribution)
    residual = spectral_matern_field(
        settings.grid_size,
        settings.extent,
        18.0,
        0.75,
        1.0,
        0.0,
        generator,
    )
    signal = standardize(
        np.sqrt(1.0 - settings.residual_weight) * covariate_contribution
        + np.sqrt(settings.residual_weight) * residual
    )
    latent_target = 50.0 + 10.0 * signal
    noise_standard_deviation = 10.0 * settings.observation_noise
    measured_target = latent_target + noise_standard_deviation * generator.normal(
        size=latent_target.shape
    )
    covariates = np.column_stack(
        [
            mean + scale * field.ravel()
            for mean, scale, field in zip(
                (20.0, 8.0, 4.0),
                (3.0, 1.5, 0.8),
                covariate_fields,
                strict=True,
            )
        ]
    )
    return {
        "field_number": field_number,
        "source_field_id": source_field_id,
        "seed": seed,
        "coordinates": coordinates.astype(np.float32),
        "covariates": covariates.astype(np.float32),
        "target": measured_target.ravel().astype(np.float32),
        "latent_target": latent_target.ravel().astype(np.float32),
        "coefficient_fields": np.column_stack(
            [field.ravel() for field in coefficient_fields]
        ).astype(np.float32),
        "axis": axis.astype(np.float32),
        "noise_standard_deviation": noise_standard_deviation,
    }


def farthest_point_sample(
    coordinates: np.ndarray,
    candidates: np.ndarray,
    sample_size: int,
    seed: int,
) -> np.ndarray:
    """Select spatially dispersed samples using greedy maximin sampling."""

    if sample_size > len(candidates):
        raise ValueError("The requested sample exceeds the candidate count.")
    generator = np.random.default_rng(seed)
    candidate_coordinates = coordinates[candidates]
    first = int(generator.integers(0, len(candidates)))
    selected = [first]
    minimum_distance = np.linalg.norm(
        candidate_coordinates - candidate_coordinates[first],
        axis=1,
    )
    minimum_distance[first] = -np.inf
    for _ in range(1, sample_size):
        next_index = int(np.argmax(minimum_distance))
        selected.append(next_index)
        distance = np.linalg.norm(
            candidate_coordinates - candidate_coordinates[next_index],
            axis=1,
        )
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[np.asarray(selected)] = -np.inf
    return candidates[np.asarray(selected, dtype=int)]


def make_interior_split(
    coordinates: np.ndarray,
    field_number: int,
    config: RandomFieldConfig | None = None,
) -> dict[str, np.ndarray | float]:
    """Create a buffered interior test block and a training/calibration split."""

    settings = config or RandomFieldConfig()
    source_field_id = settings.source_field_id_offset + field_number
    split_identifier = source_field_id + 1000
    center_x, center_y = BLOCK_CENTERS[
        (split_identifier - 1) % len(BLOCK_CENTERS)
    ]
    half_width = settings.extent * settings.test_fraction_of_extent / 2.0
    expanded = half_width + settings.buffer_width
    x_coordinate, y_coordinate = coordinates[:, 0], coordinates[:, 1]
    test_mask = (
        (np.abs(x_coordinate - center_x) <= half_width)
        & (np.abs(y_coordinate - center_y) <= half_width)
    )
    buffer_mask = (
        (np.abs(x_coordinate - center_x) <= expanded)
        & (np.abs(y_coordinate - center_y) <= expanded)
    )
    candidates = np.flatnonzero(~buffer_mask)
    sampled = farthest_point_sample(
        coordinates,
        candidates,
        settings.training_sample_size,
        settings.master_seed + 104729 * split_identifier,
    )
    test_indices = np.flatnonzero(test_mask)

    calibration_size = int(round(len(sampled) * settings.calibration_fraction))
    labels = KMeans(
        n_clusters=8,
        random_state=10_000 + split_identifier,
        n_init=20,
    ).fit_predict(coordinates[sampled])
    cluster_centers = np.vstack(
        [
            coordinates[sampled][labels == label].mean(axis=0)
            for label in range(8)
        ]
    )
    distance_from_block = np.linalg.norm(
        cluster_centers - np.asarray([center_x, center_y]),
        axis=1,
    )
    calibration_local: list[int] = []
    for label in np.argsort(distance_from_block):
        calibration_local.extend(np.flatnonzero(labels == label).tolist())
        if len(calibration_local) >= calibration_size:
            break
    calibration_mask = np.zeros(len(sampled), dtype=bool)
    calibration_mask[np.asarray(calibration_local[:calibration_size])] = True
    calibration_indices = sampled[calibration_mask]
    training_indices = sampled[~calibration_mask]
    separation = cdist(
        coordinates[test_indices],
        coordinates[training_indices],
    ).min(axis=1)
    return {
        "training_indices": training_indices.astype(np.int32),
        "calibration_indices": calibration_indices.astype(np.int32),
        "test_indices": test_indices.astype(np.int32),
        "test_to_training_minimum": float(separation.min()),
        "test_to_training_median": float(np.median(separation)),
    }
