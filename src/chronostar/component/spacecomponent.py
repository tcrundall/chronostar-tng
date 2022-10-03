import numpy as np
from numpy import float64
from numpy.typing import NDArray
from typing import Any

from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
from sklearn.mixture._gaussian_mixture import _estimate_log_gaussian_prob
from sklearn.mixture._gaussian_mixture import _estimate_gaussian_parameters

from src.chronostar.base import BaseComponent


class SpaceComponent(BaseComponent):
    reg_covar = 1e-6
    covariance_type = "full"

    def __init__(self, config_params: dict[Any, Any]) -> None:
        self.config_params = config_params

    def maximize(
        self,
        X: NDArray[float64],
        log_resp: NDArray[float64]
    ) -> None:
        """
        Utilize sklearn methods, but adjusting array dimensions for
        single component usage.
        """
        _, means_, covariances_ = _estimate_gaussian_parameters(
            X,
            np.exp(log_resp[:, np.newaxis]),
            self.reg_covar,
            self.covariance_type,
        )

        self.mean = means_.squeeze()
        self.covariance = covariances_.squeeze()
        self.precision_chol = _compute_precision_cholesky(
            self.covariance[np.newaxis], self.covariance_type
        ).squeeze()

    def estimate_log_prob(self, X: NDArray[float64]) -> NDArray[float64]:
        return np.array(_estimate_log_gaussian_prob(
            X,
            self.mean[np.newaxis],
            self.precision_chol[np.newaxis],
            self.covariance_type,
        ).squeeze(), dtype=float64)

    @property
    def n_params(self) -> int:
        n_features = len(self.mean)
        mean_params = n_features
        cov_params = n_features * (n_features + 1) / 2.0
        return int(mean_params + cov_params)

    def set_parameters(self, params: tuple) -> None:
        self.mean, self.covariance = params

    def get_parameters(self) -> tuple:
        return self.mean, self.covariance
