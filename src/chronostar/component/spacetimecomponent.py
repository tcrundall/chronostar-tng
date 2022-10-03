from typing import Optional, Callable
import numpy as np
from numpy import float64
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from sklearn.mixture._gaussian_mixture import _estimate_gaussian_parameters
from sklearn.mixture._gaussian_mixture import _estimate_log_gaussian_prob
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky

from src.chronostar.base import BaseComponent
from src.chronostar.utils.utils import trace_epicyclic_orbit
from src.chronostar.utils.transform import transform_covmatrix


def morph_covariance(covariance: NDArray[float64]) -> NDArray[float64]:
    covariance[3:, :3] = 0.
    covariance[:3, 3:] = 0.
    return covariance


def apply_age_constraints(
    mean: NDArray[float64],
    covariance: NDArray[float64],
    age: float,
    trace_func: Optional[Callable] = None,
) -> tuple[NDArray[float64], NDArray[float64]]:
    if trace_func is None:
        trace_func = trace_epicyclic_orbit

    # Get birth mean
    mean_birth = trace_func(mean, -age)

    # Get approximate birth covariance matrix
    cov_birth_approx = transform_covmatrix(
        cov=covariance,
        trans_func=trace_func,
        loc=mean,
        args=(-age,),
    )

    # Morph birth covariance matrix so it matches assumptions
    cov_birth = morph_covariance(cov_birth_approx)

    # Project birth covariance matrix to current day
    mean_aged = mean
    cov_aged = transform_covmatrix(
        cov=cov_birth,
        trans_func=trace_epicyclic_orbit,
        loc=mean_birth,
        args=(age,),
    )

    return mean_aged, cov_aged


class SpaceTimeComponent(BaseComponent):
    reg_covar = 1e-6
    covariance_type = "full"
    # project = trace_epicyclic_orbit

    def __init__(self, config_params):
        self.config_params = config_params

    def _estimate_aged_gaussian_parameters(
        self,
        X,
        resp,
        age,
        reg_covar,
    ) -> tuple[NDArray[float64], NDArray[float64]]:
        _, means_, covariances_ = _estimate_gaussian_parameters(
            X,
            resp[:, np.newaxis],
            reg_covar,
            covariance_type="full",
        )
        fitted_mean_now = means_.squeeze()
        fitted_cov_now = covariances_.squeeze()

        return apply_age_constraints(fitted_mean_now, fitted_cov_now, age)

    def _loss(self, age, X, log_resp) -> float:
        mean_now, cov_now = self._estimate_aged_gaussian_parameters(
            X,
            np.exp(log_resp),
            age,
            self.reg_covar,
        )

        prec_now_chol = _compute_precision_cholesky(
            cov_now[np.newaxis],
            self.covariance_type,
        ).squeeze()

        log_prob = _estimate_log_gaussian_prob(
            X,
            mean_now[np.newaxis],
            prec_now_chol[np.newaxis],
            self.covariance_type,
        ).squeeze()

        return float(-np.sum(np.exp(log_resp) * log_prob))

    def maximize(self, X, log_resp) -> None:
        res = minimize_scalar(
            self._loss,
            args=(X, log_resp),
            method="brent",
        )
        self.age = res.x

        self.mean, self.covariance = self._estimate_aged_gaussian_parameters(
            X,
            np.exp(log_resp),
            self.age,
            self.reg_covar,
        )

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

    def morph_covariance(
        self,
        covariance: NDArray[float64]
    ) -> NDArray[float64]:
        """
        Retain pos-pos correlation, vel-vel correlation,
        total position volume and total velocity volume.

        i.e. remove all pos-vel correlations.

        so.... I just set all pos-vel corrs to zero?
        """
        covariance[3:, :3] = 0.
        covariance[:3, 3:] = 0.
        return covariance

    @property
    def n_params(self) -> int:
        mean_params = 6
        cov_params = 6 + 3 + 3
        age_param = 1
        return mean_params + cov_params + age_param

    def set_parameters(
        self,
        params,
        # mean: NDArray[float64],
        # covariance: NDArray[float64],
        # age: float,
    ) -> None:
        (
            self.mean,
            self.covariance,
            self.age,
        ) = params
        # self.age = age
        # self.mean = mean
        # self.covariance = covariance
        self.precision_chol = _compute_precision_cholesky(
            self.covariance[np.newaxis], self.covariance_type,
        ).squeeze()

    def get_parameters(self):
        return (
            self.mean,
            self.covariance,
            self.age,
        )
