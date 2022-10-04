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


def remove_posvel_correlations(covariance: NDArray[float64]) -> NDArray[float64]:
    covariance[3:, :3] = 0.
    covariance[:3, 3:] = 0.
    return covariance


def apply_age_constraints(
    mean: NDArray[float64],
    covariance: NDArray[float64],
    age: float,
    trace_orbit_func: Callable = trace_epicyclic_orbit,
    morph_cov_func: Callable = remove_posvel_correlations,
) -> tuple[NDArray[float64], NDArray[float64]]:

    # Get birth mean
    mean_birth = trace_orbit_func(mean, -age)

    # Get approximate birth covariance matrix
    cov_birth_approx = transform_covmatrix(
        cov=covariance,
        trans_func=trace_orbit_func,
        loc=mean,
        args=(-age,),
    )

    # Morph birth covariance matrix so it matches assumptions
    cov_birth = morph_cov_func(cov_birth_approx)

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
    COVARIANCE_TYPE = "full"     # an sklearn specific parameter. DON'T CHANGE!

    @classmethod
    def configure(
        cls,
        *,
        minimize_method='brent',
        reg_covar=1e-6,
        trace_orbit_func=trace_epicyclic_orbit,
        morph_cov_func=remove_posvel_correlations,
    **kwargs) -> None:

        cls.minimize_method = minimize_method
        cls.reg_covar = reg_covar

        if isinstance(trace_orbit_func, Callable):
            cls.trace_orbit_func = trace_orbit_func
        elif trace_orbit_func == 'epiyclic':
            cls.trace_orbit_func = trace_epicyclic_orbit
        else:
            raise UserWarning(f"{cls} config: Unknown {trace_orbit_func=}")

        if isinstance(morph_cov_func, Callable):
            cls.morph_cov_func = morph_cov_func
        elif morph_cov_func == "elliptical":
            cls.morph_cov_func = remove_posvel_correlations
        else:
            raise UserWarning(f"{cls} config: Unknown {morph_cov_func=}")

        if (kwargs):
            print(f"Extra keyword arguments provided:\n{kwargs}")

    def __init__(self, params=None):
        if params:
            self.set_parameters(params)

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
            covariance_type=self.COVARIANCE_TYPE,
        )
        fitted_mean_now = means_.squeeze()
        fitted_cov_now = covariances_.squeeze()

        # Todo: CHECK THIS IS RIGHT
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
            "full",
        ).squeeze()

        log_prob = _estimate_log_gaussian_prob(
            X,
            mean_now[np.newaxis],
            prec_now_chol[np.newaxis],
            self.COVARIANCE_TYPE,
        ).squeeze()

        return float(-np.sum(np.exp(log_resp) * log_prob))

    def maximize(self, X, log_resp) -> None:
        res = minimize_scalar(
            self._loss,
            args=(X, log_resp),
            method=self.minimize_method,
        )
        self.age = res.x

        self.mean, self.covariance = self._estimate_aged_gaussian_parameters(
            X,
            np.exp(log_resp),
            self.age,
            self.reg_covar,
        )

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
        mean_params = 6
        cov_params = 6 + 3 + 3
        age_param = 1
        return mean_params + cov_params + age_param

    def set_parameters(
        self,
        params,
    ) -> None:
        (self.mean, self.covariance, self.age) = params

        self.precision_chol = _compute_precision_cholesky(
            self.covariance[np.newaxis], self.COVARIANCE_TYPE,
        ).squeeze()

    def get_parameters(self):
        return (self.mean, self.covariance, self.age)
