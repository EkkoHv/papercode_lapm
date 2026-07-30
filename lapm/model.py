"""Implementation of the Local Association Prediction Method (LAPM)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def set_seed(seed: int) -> None:
    """Set deterministic seeds for NumPy, Python, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


@dataclass(frozen=True)
class LAPMConfig:
    """Core settings reported for LAPM in the manuscript."""

    centers: int = 6
    bandwidth_multipliers: tuple[float, ...] = (
        np.sqrt(5.0),
        1.0,
        1.0 / np.sqrt(5.0),
    )
    hidden_width: int = 64
    residual_blocks: int = 2
    dropout: float = 0.15
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    epochs: int = 180
    mc_passes: int = 50
    seed: int = 2026


class LocalAssociationFeatures:
    """Construct center-specific Gaussian responses at multiple spatial scales."""

    def __init__(
        self,
        centers: int,
        bandwidth_multipliers: Iterable[float],
        seed: int,
    ) -> None:
        self.number_of_centers = int(centers)
        self.bandwidth_multipliers = np.asarray(tuple(bandwidth_multipliers), dtype=float)
        self.seed = int(seed)
        self.centers: np.ndarray | None = None
        self.reference_bandwidth: float | None = None
        self.bandwidths: np.ndarray | None = None

    def fit(self, coordinates: np.ndarray) -> "LocalAssociationFeatures":
        """Fit spatial centers and bandwidths using training coordinates only."""

        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("Coordinates must have shape (n, 2).")
        number = min(self.number_of_centers, len(coordinates))
        self.centers = KMeans(
            n_clusters=number,
            n_init=20,
            random_state=self.seed,
        ).fit(coordinates).cluster_centers_
        if number == 1:
            spans = np.ptp(coordinates, axis=0)
            reference = float(max(np.max(spans) / 4.0, 1.0e-8))
        elif number == 2:
            reference = float(np.linalg.norm(self.centers[0] - self.centers[1]))
        else:
            nearest = NearestNeighbors(n_neighbors=2).fit(self.centers)
            neighboring_distances = nearest.kneighbors(return_distance=True)[0]
            reference = float(np.median(neighboring_distances[:, 1]))
        self.reference_bandwidth = max(reference, 1.0e-8)
        self.bandwidths = self.reference_bandwidth * self.bandwidth_multipliers
        return self

    def transform(self, coordinates: np.ndarray) -> np.ndarray:
        """Return the K by J center-scale responses as one feature vector."""

        if self.centers is None or self.bandwidths is None:
            raise RuntimeError("The spatial feature builder has not been fitted.")
        distances = cdist(np.asarray(coordinates, dtype=float), self.centers)
        responses = [
            np.exp(-0.5 * (distances / bandwidth) ** 2)
            for bandwidth in self.bandwidths
        ]
        return np.column_stack(responses)


class ResidualBlock(torch.nn.Module):
    """A fully connected residual block."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(width, width)
        self.normalization = torch.nn.LayerNorm(width)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        update = torch.relu(self.normalization(self.linear(values)))
        return values + self.dropout(update)


class ResidualRegressor(torch.nn.Module):
    """Residual network returning one scalar prediction per location."""

    def __init__(self, input_size: int, config: LAPMConfig) -> None:
        super().__init__()
        self.input_layer = torch.nn.Linear(input_size, config.hidden_width)
        self.input_normalization = torch.nn.LayerNorm(config.hidden_width)
        self.blocks = torch.nn.ModuleList(
            [
                ResidualBlock(config.hidden_width, config.dropout)
                for _ in range(config.residual_blocks)
            ]
        )
        self.dropout = torch.nn.Dropout(config.dropout)
        self.output_layer = torch.nn.Linear(config.hidden_width, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.input_normalization(self.input_layer(values)))
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_layer(self.dropout(hidden))


class LAPM:
    """Fit LAPM and predict soil properties at unsampled locations."""

    def __init__(
        self,
        config: LAPMConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config or LAPMConfig()
        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(selected_device)
        self.spatial_features = LocalAssociationFeatures(
            self.config.centers,
            self.config.bandwidth_multipliers,
            self.config.seed,
        )
        self.feature_scaler = StandardScaler()
        self.covariate_medians: np.ndarray | None = None
        self.target_mean: float | None = None
        self.target_scale: float | None = None
        self.network: ResidualRegressor | None = None

    def _impute_covariates(
        self,
        covariates: np.ndarray | None,
        number_of_rows: int,
        fit: bool,
    ) -> np.ndarray:
        if covariates is None:
            covariate_array = np.empty((number_of_rows, 0), dtype=float)
        else:
            covariate_array = np.asarray(covariates, dtype=float).copy()
            if covariate_array.ndim == 1:
                covariate_array = covariate_array[:, None]
            if covariate_array.ndim != 2:
                raise ValueError("Covariates must have shape (n, p).")
            if len(covariate_array) != number_of_rows:
                raise ValueError(
                    "Coordinates and covariates must contain the same number of rows."
                )
        if fit:
            if covariate_array.shape[1] == 0:
                self.covariate_medians = np.empty(0, dtype=float)
            else:
                self.covariate_medians = np.nanmedian(covariate_array, axis=0)
                self.covariate_medians = np.where(
                    np.isfinite(self.covariate_medians),
                    self.covariate_medians,
                    0.0,
                )
        if self.covariate_medians is None:
            raise RuntimeError("Covariate preprocessing has not been fitted.")
        if covariate_array.shape[1] != len(self.covariate_medians):
            raise ValueError(
                "Prediction covariates must match those supplied during model fitting."
            )
        row, column = np.where(~np.isfinite(covariate_array))
        covariate_array[row, column] = self.covariate_medians[column]
        return covariate_array

    def _raw_features(
        self,
        coordinates: np.ndarray,
        covariates: np.ndarray | None,
        fit: bool,
    ) -> np.ndarray:
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("Coordinates must have shape (n, 2).")
        if not np.isfinite(coordinates).all():
            raise ValueError("Coordinates cannot contain missing or nonfinite values.")
        ancillary = self._impute_covariates(
            covariates,
            number_of_rows=len(coordinates),
            fit=fit,
        )
        if fit:
            self.spatial_features.fit(coordinates)
        responses = self.spatial_features.transform(coordinates)
        return np.column_stack([coordinates, ancillary, responses])

    def fit(
        self,
        coordinates: np.ndarray,
        covariates: np.ndarray | None,
        target: np.ndarray,
    ) -> "LAPM":
        """Fit all preprocessing operations and network weights."""

        set_seed(self.config.seed)
        target = np.asarray(target, dtype=float).ravel()
        raw_features = self._raw_features(coordinates, covariates, fit=True)
        if len(target) != len(raw_features):
            raise ValueError(
                "Coordinates and target values must contain the same number of rows."
            )
        if not np.isfinite(target).all():
            raise ValueError("Target values cannot contain missing or nonfinite values.")
        features = self.feature_scaler.fit_transform(raw_features).astype(np.float32)
        self.target_mean = float(np.mean(target))
        self.target_scale = float(max(np.std(target, ddof=0), 1.0e-8))
        normalized_target = ((target - self.target_mean) / self.target_scale).astype(
            np.float32
        )
        self.network = ResidualRegressor(features.shape[1], self.config).to(self.device)
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.epochs,
        )
        x_tensor = torch.as_tensor(features, device=self.device)
        y_tensor = torch.as_tensor(normalized_target, device=self.device)
        self.network.train()
        for _ in range(self.config.epochs):
            optimizer.zero_grad(set_to_none=True)
            output = self.network(x_tensor)
            loss = torch.mean((output[:, 0] - y_tensor) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()
        return self

    def _scaled_features(
        self,
        coordinates: np.ndarray,
        covariates: np.ndarray | None,
    ) -> torch.Tensor:
        raw = self._raw_features(coordinates, covariates, fit=False)
        scaled = self.feature_scaler.transform(raw).astype(np.float32)
        return torch.as_tensor(scaled, device=self.device)

    def predict(
        self,
        coordinates: np.ndarray,
        covariates: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return deterministic predictions with dropout disabled."""

        if self.network is None or self.target_mean is None or self.target_scale is None:
            raise RuntimeError("The model has not been fitted.")
        self.network.eval()
        with torch.no_grad():
            output = self.network(self._scaled_features(coordinates, covariates))
        normalized = output[:, 0].detach().cpu().numpy()
        return normalized * self.target_scale + self.target_mean

    def predict_mc(
        self,
        coordinates: np.ndarray,
        covariates: np.ndarray | None = None,
        passes: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the mean and population SD of stochastic MC dropout predictions."""

        if self.network is None or self.target_mean is None or self.target_scale is None:
            raise RuntimeError("The model has not been fitted.")
        number_of_passes = self.config.mc_passes if passes is None else int(passes)
        if number_of_passes < 1:
            raise ValueError("The number of MC dropout passes must be positive.")
        set_seed(self.config.seed + 1000)
        features = self._scaled_features(coordinates, covariates)
        self.network.train()
        prediction_samples: list[np.ndarray] = []
        with torch.no_grad():
            for _ in range(number_of_passes):
                output = self.network(features)
                normalized = output[:, 0].detach().cpu().numpy()
                prediction_samples.append(
                    normalized * self.target_scale + self.target_mean
                )
        samples = np.stack(prediction_samples, axis=0)
        predictive_mean = samples.mean(axis=0)
        predictive_variance = samples.var(axis=0, ddof=0)
        self.network.eval()
        return predictive_mean, np.sqrt(np.maximum(predictive_variance, 0.0))

    @property
    def fitted_settings(self) -> dict[str, object]:
        """Return fitted spatial parameters for reporting and reproducibility."""

        return {
            "centers": None
            if self.spatial_features.centers is None
            else self.spatial_features.centers.tolist(),
            "reference_bandwidth": self.spatial_features.reference_bandwidth,
            "bandwidths": None
            if self.spatial_features.bandwidths is None
            else self.spatial_features.bandwidths.tolist(),
            "input_dimension": None
            if self.network is None
            else self.network.input_layer.in_features,
            "output_dimension": 1,
            "training_loss": "mean_squared_error",
            "normalization": "layer_normalization",
            "mc_variance": "population_variance_across_stochastic_predictions",
            "device": str(self.device),
        }
