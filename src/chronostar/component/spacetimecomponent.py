from typing import Callable
import numpy as np
from numpy import float64
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

from sklearn.mixture._gaussian_mixture import _estimate_gaussian_parameters
from sklearn.mixture._gaussian_mixture import _estimate_log_gaussian_prob
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky

from ..base import BaseComponent
from ..traceorbit import trace_epicyclic_orbit
from ..utils.transform import transform_covmatrix


def remove_posvel_correlations(
    covariance: NDArray[float64]
) -> NDArray[float64]:
    """
    Impose birth-site assumption of no pos-vel correlations

    Parameters
    ----------
    covariance : NDArray[float64] of shape (n_features, n_features)
        Current approximation of birth-site's covariance

    Returns
    -------
    NDArray[float64] of shape (n_features, n_features)
        An association birth-site's covariance with no pos-vel
        correlations.
    """
    new_cov = np.array(covariance)
    new_cov[3:, :3] = 0.
    new_cov[:3, 3:] = 0.
    return new_cov


# def estimate_log_prob(self, X: NDArray[float64]) -> NDArray[float64]:
def apply_age_constraints(
    mean: NDArray[float64],
    covariance: NDArray[float64],
    age: float,
    trace_orbit_func: Callable = trace_epicyclic_orbit,
    morph_cov_func: Callable = remove_posvel_correlations,
) -> tuple[NDArray[float64], NDArray[float64]]:
    """Adjust a given multi-dimensional Gaussian such that
    it's covariance reflects any expected correlations
    implied by it's age.

    Parameters
    ----------
    mean : NDArray[float64] of shape (n_features)
        The current estimate of component's mean
    covariance : NDArray[float64] of shape (n_features, n_features)
        The current estimate of component's covariance
    age : float
        The current estimate of component's age
    trace_orbit_func : Callable, optional
        A function that traces an orbit through feature space by
        A certain amoutn of time, by default
        :func:`~src.chronostar.utils.utils.trace_epicyclic_orbit`.

        Signature must be of the form:
            f(start: array-like of shape (n_features), age: float)
            -> end: array-like of shape (n_features)
    morph_cov_func : Callable, optional
        A function that modifies the birth-site's covariance
        matrix, imposing assumptions, by default
        :func:`remove_posvel_correlations`.

        Signature must be of the form:
            f(covariance: array-like of shape (n_features, n_features))
            -> res: array-like of shape (n_features, n_features)

    Returns
    -------
    tuple[NDArray[float64], NDArray[float64]]
        Return the mean (nfeatures,) and covariance matrix
        (n_features, n_features) of the component (current day)
        with time evolution baked into the covariance matrix.
    """

    # Get birth mean
    mean_birth = trace_orbit_func(mean[np.newaxis, :], -age)

    """ Attempting to fix broken model...
    aged_identity = transform_covmatrix(
        cov=np.eye(6),
        trans_func=trace_epicyclic_orbit,
        loc=mean_birth,
        args=(age,),
    )
    aged_identity_stds = np.sqrt(aged_identity[np.diag_indices(6)])

    pos_vel_corrs = np.copy(aged_identity[:3, 3:])
    pos_vel_corrs[0] /= aged_identity_stds[0]
    pos_vel_corrs[1] /= aged_identity_stds[1]
    pos_vel_corrs[2] /= aged_identity_stds[2]
    pos_vel_corrs[:, 0] /= aged_identity_stds[3]
    pos_vel_corrs[:, 1] /= aged_identity_stds[4]
    pos_vel_corrs[:, 2] /= aged_identity_stds[5]

    stdevs = np.sqrt(covariance[np.diag_indices(6)])

    pos_vel_covs = np.copy(pos_vel_corrs)
    pos_vel_covs[0] *= stdevs[0]
    pos_vel_covs[1] *= stdevs[1]
    pos_vel_covs[2] *= stdevs[2]
    pos_vel_covs[:, 0] *= stdevs[3]
    pos_vel_covs[:, 1] *= stdevs[4]
    pos_vel_covs[:, 2] *= stdevs[5]

    new_cov_aged = np.copy(covariance)
    new_cov_aged[:3, 3:] = pos_vel_covs
    new_cov_aged[3:, :3] = pos_vel_covs.T

    new_cov_birth = transform_covmatrix(
        cov=new_cov_aged,
        trans_func=trace_orbit_func,
        loc=mean,
        args=(-age,),
    )
    """

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
    """A 6D phase-space Gaussian component with age.

    Capable of fitting itself to a set of samples and
    responsibilities (membership probabilities)
    """

    COVARIANCE_TYPE = "full"     # an sklearn specific parameter. DON'T CHANGE!

    # Use this to convert function name strings from config files into funcs
    function_parser = {
        'trace_epicyclic_orbit': trace_epicyclic_orbit,
        'remove_pos_vel_correlations': remove_posvel_correlations,
    }

    # Configurable attributes
    minimize_method: str = 'brent'
    reg_covar: float = 1e-6
    trace_orbit_func = trace_epicyclic_orbit
    morph_cov_func = remove_posvel_correlations

    def __init__(self, params=None):
        super().__init__(params)

    def _estimate_aged_gaussian_parameters(
        self,
        X: NDArray,
        resp: NDArray,
        age: float,
        reg_covar: float,
    ) -> tuple[NDArray, NDArray]:
        """Estimate gaussian parameters, taking into account
        assumed age.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data
        resp : ndarray of shape (n_samples)
            Component responsibilities (membership probabilities)
        age : float
            Assumed age of component
        reg_covar : float
            Regularization constant added to covariance diagonal
            elements.

        Returns
        -------
        tuple[NDArray[float64], NDArray[float64]]
            The best fitting mean (n_features) and
            covariance matrix (n_features, n_features)
            consistent with `age`
        """
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

    def loss(
        self,
        age: float,
        X: NDArray[float64],
        resp: NDArray[float64]
    ) -> float:
        """Calculate the loss (i.e. -log likelihood) of the
        data.

        Parameters
        ----------
        age : float
            The assumed age. This is the sole independent parameter.
        X : ndarray of shape (n_samples, n_features)
            Input data
        resp : ndarray of shape (n_samples)
            component responsibilities (membership probabilities)

        Returns
        -------
        float
            Negative log likelihood of data given `age` and derived
            model parameters.
        """
        mean_now, cov_now = self._estimate_aged_gaussian_parameters(
            X,
            resp,
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

        return float(-np.sum(resp * log_prob))

    def maximize(
        self,
        X: NDArray[float64],
        resp: NDArray[float64]
    ) -> None:
        """Find the best model parameters for the data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data
        resp : ndarray of shape (n_samples)
            component responsibilities (membership probabilities)

        Notes
        -----
        This method performs a parameter exploration on the age,
        and infers an appropriate mean and covariance based on the
        best age and the data.
        """

        res = minimize_scalar(
            self.loss,
            args=(X, resp),
            method=self.minimize_method,
        )
        self.age = res.x

        self.mean, self.covariance = self._estimate_aged_gaussian_parameters(
            X,
            resp,
            self.age,
            self.reg_covar,
        )

        self.precision_chol = _compute_precision_cholesky(
            self.covariance[np.newaxis], self.COVARIANCE_TYPE
        ).squeeze()

    def estimate_log_prob(self, X: NDArray[float64]) -> NDArray[float64]:
        """Calculate the log probability of each sample given
        this component

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data

        Returns
        -------
        NDArray[float64] of shape (n_samples)
            The multivariate normal defined by `self.mean` and
            `self.covariance` evaluated at each point in X
        """
        return np.array(_estimate_log_gaussian_prob(
            X,
            self.mean[np.newaxis],
            self.precision_chol[np.newaxis],
            self.COVARIANCE_TYPE,
        ).squeeze(), dtype=float64)

    @property
    def n_params(self) -> int:
        """Return the number of parameters required to
        define this model

        Returns
        -------
        int
            The number of parameters required to define
            this model
        """
        mean_params = 6
        cov_params = 6 + 3 + 3
        age_param = 1
        return mean_params + cov_params + age_param

    def set_parameters(
        self,
        params,
    ) -> None:
        """Set the internal parameters of the model.

        Parameters
        ----------
        params : (n_features), (n_features, n_features), float
            mean, covariance, age
        """
        (self.mean, self.covariance, self.age) = params

        self.precision_chol = _compute_precision_cholesky(
            self.covariance[np.newaxis], self.COVARIANCE_TYPE,
        ).squeeze()

    def get_parameters(self):
        """Get the internal parameters of the model

        Returns
        -------
        (n_features), (n_features, n_features), float
            mean, covariance, age
        """
        return (self.mean, self.covariance, self.age)
