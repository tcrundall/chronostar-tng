from __future__ import annotations
from typing import Optional
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


def construct_cov_from_params(
    cov_params: NDArray[float64]
) -> NDArray[float64]:
    dxyz, duvw = cov_params
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
        [dxyz, duvw],
    )


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
    minimize_method: str = 'Nelder-Mead'
    reg_covar: float = 1e-6
    trace_orbit_func = staticmethod(trace_epicyclic_orbit)
    construct_cov_func = staticmethod(construct_cov_from_params)

    def __init__(self, params: Optional[NDArray[float64]] = None):
        super().__init__(params)

    def cov_lnpriors(self, cov_params):
        dxyz, duvw = cov_params

        if any([delta <= 0 for delta in [dxyz, duvw]]):
            return -np.inf

        # TODO: Apply some prior based on standard deviations
        return 0.

    def loss(
        self,
        model_params: NDArray[float64],
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
        mean_birth = model_params[:6]
        cov_birth_params = model_params[6:-1]
        age = model_params[-1]

        lnprior = self.cov_lnpriors(cov_birth_params)

        if lnprior == -np.inf:
            return -lnprior

        mean_now = self.trace_orbit_func(mean_birth[np.newaxis], age).squeeze()
        cov_birth = construct_cov_from_params(cov_birth_params)

        cov_now = transform_covmatrix(
            cov_birth,
            self.trace_orbit_func,
            mean_birth,
            args=(age,)
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
            'MEAN': (-2000., 2000.),
            'STDEV': (0., 1000.,),
            'INV_STDEV': (0., np.inf),
            'CORR': (-1, 1),
            'AGE': (-200, 200),
        }

        par_types = 6*['MEAN'] + 2*['STDEV'] + ['AGE']

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

        # If we have already fit this component, or this component
        # was initialized with parameters previously, then use them
        if self.parameters_set:
            base_init_guess = self.parameters

        # Otherwise, initialise based on the data, with age 0.
        else:
            _, est_means, est_covariances = _estimate_gaussian_parameters(
                X,
                resp[:, np.newaxis],
                self.reg_covar,
                self.COVARIANCE_TYPE,
            )
            mean_params_init_guess = est_means.squeeze()
            cov_params_init_guess = construct_params_from_cov(
                est_covariances.squeeze()
            )
            base_init_guess = np.hstack(
                [mean_params_init_guess, cov_params_init_guess, 0.]
            )

        bounds = self.get_parameter_bounds()
        all_results = []

        # TODO: consider only offsetting on first maximization?
        # We should only need to force age offsets if this is the first
        # time this component is being maximized. Otherwise it's previous
        # parameter set should be close enough.
        for age_offset in [0., 40., 120.]:
            # Offset initial guess age by a certain amount
            ig_age = base_init_guess[-1] + age_offset

            # Adjust initial guess mean by tracing back an extra amount
            ig_mean = self.trace_orbit_func(
                base_init_guess[:6][np.newaxis],
                -age_offset
            )
            init_guess = np.copy(base_init_guess)
            init_guess[:6] = ig_mean
            init_guess[-1] = ig_age
            res = optimize.minimize(
                self.loss,
                x0=init_guess,
                args=(X, resp),
                method=self.minimize_method,
                bounds=bounds,
            )
            all_results.append(res)
        best_res = min(all_results, key=lambda x: x.fun)
        # self.age, *self.cov_params = res.x

        self.set_parameters(best_res.x)
        print(f"age: {self.age}")
        # if not np.all(np.linalg.eigvals(self.covariance) > 0):
        #     raise UserWarning("Cov not pos-def")

        # self.precision_chol = _compute_precision_cholesky(
        #     self.covariance[np.newaxis], self.COVARIANCE_TYPE
        # ).squeeze()

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
        params: NDArray[float64],
    ) -> None:
        """Set the internal parameters of the model.

        Parameters
        ----------
        params : (n_features), (n_features, n_features), float
            mean, covariance, age
        """
        self.parameters_set = True
        self.parameters = params
        # (self.mean, self.covariance, self.age) = params

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
        return self.parameters

    @property
    def mean(self):
        mean_birth = self.parameters[0:6]
        mean_now = self.trace_orbit_func(
            mean_birth[np.newaxis], self.age
        ).squeeze()
        return mean_now

    @property
    def covariance(self):
        cov_params = self.parameters[6:-1]
        cov_birth = construct_cov_from_params(cov_params)
        mean_birth = self.parameters[0:6]
        cov_now = transform_covmatrix(
            cov_birth,
            self.trace_orbit_func,
            mean_birth,
            args=(self.age,)
        )
        return cov_now

    @property
    def age(self) -> float:
        return self.parameters[-1]

    def split(
        self
    ) -> tuple[SphereSpaceTimeComponent, SphereSpaceTimeComponent]:

        # Get primary axis (longest eigen vector)
        eigvals, eigvecs = np.linalg.eigh(self.covariance)
        prim_axis_length = np.sqrt(np.max(eigvals))
        prim_axis = eigvecs[:, np.argmax(eigvals)]

        # Offset the two new means along the primary axis
        new_mean_1 = self.mean + prim_axis_length * prim_axis / 2.0
        new_mean_2 = self.mean - prim_axis_length * prim_axis / 2.0

        # Transform to birth
        birth_mean_1 = self.trace_orbit_func(
            new_mean_1[np.newaxis], -self.age
        ).squeeze()
        birth_mean_2 = self.trace_orbit_func(
            new_mean_2[np.newaxis], -self.age
        ).squeeze()

        # Don't bother reshaping covariance matrix, it's really tricky in 7D

        comp1 = self.__class__(np.hstack((
            birth_mean_1,
            self.parameters[6:],
        )))
        comp2 = self.__class__(np.hstack((
            birth_mean_2,
            self.parameters[6:],
        )))

        return comp1, comp2
