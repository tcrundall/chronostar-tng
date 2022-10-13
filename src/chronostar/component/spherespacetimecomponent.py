import numpy as np
from numpy import float64
from numpy.typing import NDArray
# from scipy.optimize import minimize_scalar
from scipy import optimize

from sklearn.mixture._gaussian_mixture import _estimate_gaussian_parameters
from sklearn.mixture._gaussian_mixture import _estimate_log_gaussian_prob
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky

from ..base import BaseComponent
from ..traceorbit import trace_epicyclic_orbit
from ..utils.transform import transform_covmatrix


def construct_cov_from_params(cov_params: list[float]) -> NDArray[float64]:
    dxyz, duvw = 1./np.array(cov_params)
    cov = np.eye(6)
    cov[:3] *= dxyz**2
    cov[3:] *= duvw**2
    return cov


def construct_params_from_cov(
    covariance: NDArray[float64]
) -> NDArray[float64]:
    dxyz = np.power(np.prod(np.linalg.eigvals(covariance[:3, :3])), 1./6.)
    duvw = np.power(np.prod(np.linalg.eigvals(covariance[3:, 3:])), 1./6.)
    return np.hstack(
        [1./dxyz, 1./duvw],
    )


def estimate_aged_gaussian_parameters(
    X: NDArray,
    resp: NDArray,
    model_params,
    construct_cov_func=construct_cov_from_params,
    trace_orbit_func=trace_epicyclic_orbit,
    reg_covar: float = 1e-5,
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
    age, *cov_params = model_params

    nk = resp.sum() + 10 * np.finfo(resp.dtype).eps
    mean_now = np.dot(resp.T, X) / nk

    mean_birth = trace_orbit_func(
        mean_now[np.newaxis, :],   # type: ignore
        -age,
    )

    cov_birth = construct_cov_func(cov_params)

    cov_now = transform_covmatrix(
        cov_birth,
        trace_orbit_func,
        loc=mean_birth,
        args=(age,),
    )

    cov_now[np.diag_indices(6)] += reg_covar

    return mean_now, cov_now


class SphereSpaceTimeComponent(BaseComponent):
    """A 6D phase-space Gaussian component with age.

    Capable of fitting itself to a set of samples and
    responsibilities (membership probabilities)
    """

    COVARIANCE_TYPE = "full"     # an sklearn specific parameter. DON'T CHANGE!

    # Use this to convert function name strings from config files into funcs
    function_parser = {
        'trace_epicyclic_orbit': trace_epicyclic_orbit,
        'construct_cov_func': construct_cov_from_params,
    }

    # Configurable attributes
    minimize_method: str = 'Powell'
    reg_covar: float = 1e-6
    trace_orbit_func = staticmethod(trace_epicyclic_orbit)
    construct_cov_func = staticmethod(construct_cov_from_params)

    def __init__(self, params=None):
        self.params_set = False
        super().__init__(params)

    def cov_lnpriors(self, cov_params):
        dxyz, duvw = 1./np.array(cov_params)

        if any([delta <= 0 for delta in [dxyz, duvw]]):
            return -np.inf

        # TODO: Apply some prior based on standard deviations
        return 0.

    def loss(
        self,
        model_params: list[float64],
        X: NDArray[float64],
        resp: NDArray[float64]
    ) -> float:
        """Calculate the loss (i.e. -log likelihood) of the
        data.

        Parameters
        ----------
        model_params :
            Values that parameterise the covariance matrix and age
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
        age, *cov_params = model_params

        lnprior = self.cov_lnpriors(cov_params)

        if lnprior == -np.inf:
            return -lnprior

        mean_now, cov_now = estimate_aged_gaussian_parameters(
            X,
            resp,
            model_params,
            self.construct_cov_func,
            self.trace_orbit_func,
            reg_covar=self.reg_covar,
        )

        # Check we received a valid covariance matrix
        if not np.all(np.linalg.eigvals(cov_now) > 0):
            return np.inf

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

        return -float(lnprior + np.sum(resp * log_prob))

    def get_parameter_bounds(self):
        bound_map = {
            'INV_STDEV': (0., np.inf),
            'CORR': (-1, 1),
            'AGE': (-200, 200),
        }

        par_types = ['AGE'] + ['INV_STDEV'] * 2

        bounds = np.array([bound_map[par] for par in par_types])
        opt_bounds = optimize.Bounds(lb=bounds.T[0], ub=bounds.T[1])
        return opt_bounds

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
        # Choosing initial guess for optimize routine

        # If we have already fit this component, then use it's
        # previous parameters
        if self.params_set:
            fitted_birth_cov = transform_covmatrix(
                self.covariance,
                self.trace_orbit_func,
                self.mean,
                args=(self.age,)
            )
            cov_params_init_guess = construct_params_from_cov(fitted_birth_cov)
            base_init_guess = np.hstack([self.age, cov_params_init_guess])

        else:
            _, _, est_covariance = _estimate_gaussian_parameters(
                X,
                resp[:, np.newaxis],
                self.reg_covar,
                self.COVARIANCE_TYPE,
            )
            cov_params_init_guess = construct_params_from_cov(
                est_covariance[0]
            )
            base_init_guess = np.hstack([0, cov_params_init_guess])

        bounds = self.get_parameter_bounds()
        all_results = []
        for age_offset in [-20., -10., 0., 10., 20.]:
            init_guess = np.copy(base_init_guess)
            init_guess[0] += age_offset
            res = optimize.minimize(
                self.loss,
                x0=init_guess,
                args=(X, resp),
                method='Powell',
                # method=self.minimize_method,
                bounds=bounds,
            )
            all_results.append(res)
        best_res = min(all_results, key=lambda x: x.fun)
        # self.age, *self.cov_params = res.x

        mean, covariance = estimate_aged_gaussian_parameters(
            X,
            resp,
            best_res.x,
            self.construct_cov_func,
            self.trace_orbit_func,
            self.reg_covar,
        )
        age = best_res.x[0]
        self.set_parameters((mean, covariance, age))

        if not np.all(np.linalg.eigvals(self.covariance) > 0):
            raise UserWarning("Cov not pos-def")

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
        cov_params = 2
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

        if not np.all(np.linalg.eigvals(self.covariance) > 0):
            raise UserWarning("Didn't provide a positive definite cov")

        self.precision_chol = _compute_precision_cholesky(
            self.covariance[np.newaxis], self.COVARIANCE_TYPE,
        ).squeeze()

        self.params_set = True

    def get_parameters(self):
        """Get the internal parameters of the model

        Returns
        -------
        (n_features), (n_features, n_features), float
            mean, covariance, age
        """
        return (self.mean, self.covariance, self.age)
