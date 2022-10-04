import numpy as np
from numpy import float64
from numpy.typing import NDArray

from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
from sklearn.mixture._gaussian_mixture import _estimate_log_gaussian_prob
from sklearn.mixture._gaussian_mixture import _estimate_gaussian_parameters

from src.chronostar.base import BaseComponent


class SpaceComponent(BaseComponent):
    COVARIANCE_TYPE = "full"

    @classmethod
    def configure(
        cls,
        reg_covar=1e-6,
        **kwargs
    ):

        cls.reg_covar = reg_covar

        if kwargs:
            print(f"{cls} config: Extra keyword arguments provided:\n{kwargs}")

    def __init__(self, params=None) -> None:
        super().__init__(params)

    def maximize(
        self,
        X: NDArray[float64],
        log_resp: NDArray[float64]
    ) -> None:
        """
        Utilize sklearn methods, but adjusting array dimensions for
        single component usage.
        """
        nsamples = X.shape[0]
        if len(log_resp.shape) == 1:
            log_resp = log_resp[:, np.newaxis]
        assert log_resp.shape == (nsamples, 1)
        _, means_, covariances_ = _estimate_gaussian_parameters(
            X,
            np.exp(log_resp),
            self.reg_covar,
            self.COVARIANCE_TYPE,
        )

        self.mean = means_.squeeze()
        self.covariance = covariances_.squeeze()
        self.precision_chol = _compute_precision_cholesky(
            self.covariance[np.newaxis], self.COVARIANCE_TYPE
        ).squeeze()

    def estimate_log_prob(self, X: NDArray[float64]) -> NDArray[float64]:
        return np.array(_estimate_log_gaussian_prob(
            X,
            self.mean[np.newaxis],
            self.precision_chol[np.newaxis],
            self.COVARIANCE_TYPE,
        ).squeeze(), dtype=float64)

    @property
    def n_params(self) -> int:
        n_features = len(self.mean)
        mean_params = n_features
        cov_params = n_features * (n_features + 1) / 2.0
        return int(mean_params + cov_params)

    def set_parameters(self, params: tuple) -> None:
        self.mean, self.covariance = params
        self.precision_chol = _compute_precision_cholesky(
            self.covariance[np.newaxis], self.COVARIANCE_TYPE
        ).squeeze()

    def get_parameters(self) -> tuple:
        return self.mean, self.covariance
