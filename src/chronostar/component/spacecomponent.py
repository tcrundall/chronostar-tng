import numpy as np
from numpy import float64
from numpy.typing import NDArray

from src.chronostar.component.base import BaseComponent
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
from sklearn.mixture._gaussian_mixture import _estimate_log_gaussian_prob


class SpaceComponent(BaseComponent):
    def __init__(self, config_params) -> None:
        self.config_params = config_params

    def maximize(self, X, log_resp) -> None:
        REG_COVAR = 1e-6
        resp = np.exp(log_resp)
        nk = resp.sum() + 10 * np.finfo(resp.dtype).eps
        mean = np.dot(resp.T, X) / nk

        n_features = len(mean)

        diff = X - mean
        covariance = np.dot(resp * diff.T, diff) / nk
        covariance.flat[:: n_features + 1] += REG_COVAR

        self.mean = mean
        self.covariance = covariance
        self.precision_chol = _compute_precision_cholesky(
            self.covariance[np.newaxis, :],
            covariance_type="full",
        )[0]

    def estimate_log_prob(self, X) -> NDArray[float64]:
        return _estimate_log_gaussian_prob(
            X, self.mean[np.newaxis], self.precision_chol[np.newaxis], "full",
        )

    @property
    def n_params(self) -> int:
        n_features = len(self.mean)
        mean_params = n_features
        cov_params = n_features * (n_features + 1) / 2.0
        return int(mean_params + cov_params)
