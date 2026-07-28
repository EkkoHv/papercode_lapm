"""Implementation of the Local Association Prediction Method (LAPM)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
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
        1.0 / np.sqrt(5.0),
        1.0,
        np.sqrt(5.0),
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
        else:
            distances = cdist(self.centers, self.centers)
            np.fill_diagonal(distances, np.inf)
            reference = float(np.median(np.min(distances, axis=1)))
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
    """Residual network returning a mean and a log variance."""

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
        self.output_layer = torch.nn.Linear(config.hidden_width, 2)

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
        covariates: np.ndarray,
        fit: bool,
    ) -> np.ndarray:
        covariates = np.asarray(covariates, dtype=float).copy()
        if covariates.ndim == 1:
            covariates = covariates[:, None]
        if fit:
            self.covariate_medians = np.nanmedian(covariates, axis=0)
            self.covariate_medians = np.where(
                np.isfinite(self.covariate_medians),
                self.covariate_medians,
                0.0,
            )
        if self.covariate_medians is None:
            raise RuntimeError("Covariate preprocessing has not been fitted.")
        row, column = np.where(~np.isfinite(covariates))
        covariates[row, column] = self.covariate_medians[column]
        return covariates

    def _raw_features(
        self,
        coordinates: np.ndarray,
        covariates: np.ndarray,
        fit: bool,
    ) -> np.ndarray:
        coordinates = np.asarray(coordinates, dtype=float)
        ancillary = self._impute_covariates(covariates, fit=fit)
        if fit:
            self.spatial_features.fit(coordinates)
        responses = self.spatial_features.transform(coordinates)
        return np.column_stack([coordinates, ancillary, responses])

    def fit(
        self,
        coordinates: np.ndarray,
        covariates: np.ndarray,
        target: np.ndarray,
    ) -> "LAPM":
        """Fit all preprocessing operations and network weights."""

        set_seed(self.config.seed)
        target = np.asarray(target, dtype=float).ravel()
        raw_features = self._raw_features(coordinates, covariates, fit=True)
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
            mean = output[:, 0]
            log_variance = torch.clamp(output[:, 1], min=-8.0, max=6.0)
            loss = 0.5 * torch.mean(
                log_variance + (y_tensor - mean) ** 2 * torch.exp(-log_variance)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()
        return self

    def _scaled_features(
        self,
        coordinates: np.ndarray,
        covariates: np.ndarray,
    ) -> torch.Tensor:
        raw = self._raw_features(coordinates, covariates, fit=False)
        scaled = self.feature_scaler.transform(raw).astype(np.float32)
        return torch.as_tensor(scaled, device=self.device)

    def predict(
        self,
        coordinates: np.ndarray,
        covariates: np.ndarray,
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
        covariates: np.ndarray,
        passes: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the MC dropout predictive mean and standard deviation."""

        if self.network is None or self.target_mean is None or self.target_scale is None:
            raise RuntimeError("The model has not been fitted.")
        set_seed(self.config.seed + 1000)
        features = self._scaled_features(coordinates, covariates)
        self.network.train()
        mean_samples: list[np.ndarray] = []
        variance_samples: list[np.ndarray] = []
        with torch.no_grad():
            for _ in range(passes or self.config.mc_passes):
                output = self.network(features)
                mean_samples.append(output[:, 0].detach().cpu().numpy())
                variance_samples.append(
                    torch.exp(torch.clamp(output[:, 1], -8.0, 6.0))
                    .detach()
                    .cpu()
                    .numpy()
                )
        means = np.stack(mean_samples) * self.target_scale + self.target_mean
        conditional_variances = np.stack(variance_samples) * self.target_scale**2
        predictive_mean = means.mean(axis=0)
        total_variance = means.var(axis=0, ddof=1) + conditional_variances.mean(axis=0)
        return predictive_mean, np.sqrt(np.maximum(total_variance, 1.0e-12))

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
            "device": str(self.device),
        }
