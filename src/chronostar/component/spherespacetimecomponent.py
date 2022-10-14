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
    """Construct a covariance matrix from given parameters

    This implementation is a spherical covariance, with a uniform
    standard deviation in position, and a uniform standard
    devaition in velocity

    Parameters
    ----------
    cov_params : NDArray[float64] of shape(2)
        The standard deviations in position and velocity: (dxyz, duvw)

    Returns
    -------
    NDArray[float64] of shape(6, 6)
        A spherical covariance matrix
    """
    dxyz, duvw = cov_params
    cov = np.eye(6)
    cov[:3] *= dxyz**2
    cov[3:] *= duvw**2
    return cov


def construct_params_from_cov(
    covariance: NDArray[float64]
) -> NDArray[float64]:
    """Derives spherical parameters from an arbitrary covariance
    matrix

    This implementation takes the volume of the covariance matrix
    in position (an ellipsoid whose axes' half-lengths are the
    square root of the matrix's eigen values) and finds the equivalent
    radius of a sphere with equal volume.

    An identical approach is taken for velocity.

    Parameters
    ----------
    covariance : NDArray[float64] of shape(6, 6)
        A covariance matrix (can be full)

    Returns
    -------
    NDArray[float64] of shape(2)
        The standard deviations in space and velocity: (dxyz, duvw)
    """
    dxyz = np.power(np.prod(np.linalg.eigvals(covariance[:3, :3])), 1./6.)
    duvw = np.power(np.prod(np.linalg.eigvals(covariance[3:, 3:])), 1./6.)
    return np.hstack(
        [dxyz, duvw],
    )


class SphereSpaceTimeComponent(BaseComponent):
    """A 6D phase-space Gaussian component with age.

    Capable of fitting itself to a set of samples and
    responsibilities (membership probabilities)

    Parameters
    ----------
    params : ndarray of shape (9), optional
        Model parameters parameterising the modeled spherical birth site
        of the component and its age flattened into a 1D array like so:
        ``np.hstack((mean, dxyz, duvw, age))``

    Attributes
    ----------
    reg_covar : float, default 1.e-6
        A regularisation constant added to the diagonals
        of the covariance matrix, configurable
    minimize_method: str, default 'Nelder-Mead'
        The method used by ``scipy.optimize.minimize``. Must be one of

        - 'Nelder-Mead' (recommended)
        - 'Powell' (not receommended)

    trace_orbit_func: Callable: f(start_loc, time), default :func:`trace_epicyclic_orbit`
        A function that traces a position by `time` Myr. Positive `time`
        traces forward, negative `time` backwards
    parameters : ndarray of shape (42)
        The model parameters, either as set by initialization, or as
        determined by :meth:`maximize`
    """
    # an sklearn specific parameter. DON'T CHANGE!
    # Used when evaluating log probs of stars
    COVARIANCE_TYPE = "full"

    # Use this to convert function name strings from config files into funcs
    function_parser = {
        'trace_epicyclic_orbit': trace_epicyclic_orbit,
    }

    # Configurable attributes
    minimize_method: str = 'Nelder-Mead'
    reg_covar: float = 1e-6

    # We declare this as a staticmethod so that we can call
    # `self.trace_orbit_func` without passing an instance of `self` as
    # an argument
    #: :noindex:
    trace_orbit_func = staticmethod(trace_epicyclic_orbit)

    def __init__(self, params: Optional[NDArray[float64]] = None) -> None:
        super().__init__(params)

    def cov_lnpriors(self, cov_params) -> float:
        """Apply priors to covariance parameters

        For now, just a simple hard edge for negative standard
        deviations.

        Currently this should rarely be reached since the optimize
        routine should respect the bounds as set by
        :meth:`get_parameter_bounds`

        Parameters
        ----------
        cov_params : NDArray of shape (2)
            The spherical standard deviations of position and velocity

        Returns
        -------
        float
            If standard deviations are negative, returns -np.inf,
            otherwise 0.
        """
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
        model_params : ndarray of shape (9)
            Values that parameterise the birth mean, birth covariance
            matrix and age
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

        # Extract and label sets of params
        mean_birth = model_params[:6]
        cov_birth_params = model_params[6:-1]
        age = model_params[-1]

        # Check covariance params are valid
        lnprior = self.cov_lnpriors(cov_birth_params)
        if lnprior == -np.inf:
            return -lnprior

        # Get current day mean
        mean_now = self.trace_orbit_func(mean_birth[np.newaxis], age).squeeze()

        # Get current day covariance
        cov_birth = construct_cov_from_params(cov_birth_params)
        cov_now = transform_covmatrix(
            cov_birth,
            self.trace_orbit_func,
            mean_birth,
            args=(age,)
        )

        # Confirm current day covariance is valid
        if not np.all(np.linalg.eigvals(cov_now) > 0):
            return np.inf

        # Decompose covariance's precision with cholesky method
        # (optimisation used by sklearn)
        prec_now_chol = _compute_precision_cholesky(
            cov_now[np.newaxis],
            "full",
        ).squeeze()

        # Evaluate each star's log prob given the current day model
        log_prob = _estimate_log_gaussian_prob(
            X,
            mean_now[np.newaxis],
            prec_now_chol[np.newaxis],
            self.COVARIANCE_TYPE,
        ).squeeze()

        # Weight each log_prob by membership probability (resp)
        # and take the sum (which would be the product if we weren't in
        # log space), include prior.
        # https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm#E_step
        # ^^ see final equaion in this section, ignore the inner sum since we
        # only have one component here.
        # We negate since we want to *minimize* this function.
        return -float(lnprior + np.sum(resp * log_prob))

    def get_parameter_bounds(self) -> optimize.Bounds:
        """Generate a set of parameter bounds for scipy.optimize

        Returns
        -------
        optimize.Bounds
            A set of bounds to be used in :meth:`scipy.optimize.minimize`
        """
        bound_map = {
            'MEAN': (-10_000., 10_000.),
            'STDEV': (0., 10_000.,),
            'INV_STDEV': (0., np.inf),
            'CORR': (-1, 1),
            'AGE': (-50, 500),
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

        Performs a minimization on :meth:`loss` to find the
        best parameters. To account for local minima caused by the
        Z-W phase / age degeneracy, we perform 3 minimizations with age
        offset by +0., +40. and +120. These offsets appear to guarantee
        successful fits up to 200 Myr.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data
        resp : ndarray of shape (n_samples)
            component responsibilities (membership probabilities)

        Notes
        -----
        The age offsets might be overkill for subsequent fittings of
        this component. Perhaps they are only needed for the initial fit,
        since each subsequent fit is initialised on previous parameters.
        Since the 0. age offset is twice as fast as the other two, it
        represents only 20% of execution time, and thus removing the need
        for the other minimizations would yield a 5x speed increase.

        Though... individual minimization runs don't all converge to the
        same solution despite ending up in the vicinity of each other. So
        perhaps multiple runs can improve the fit?
        """
        # --------------------------------------------------
        # Choosing initial guess for optimize routine
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Perform minimization with 3 separate age offsets
        # --------------------------------------------------

        # TODO: consider only offsetting on first maximization?
        # We should only need to force age offsets if this is the first
        # time this component is being maximized. Otherwise it's previous
        # parameter set should be close enough.
        bounds = self.get_parameter_bounds()
        all_results = []
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

        # Take the best minimization result
        best_res = min(all_results, key=lambda x: x.fun)

        # We use the setter method because it handles any derived
        # attributes, e.g. :attr:`precision_chol`
        self.set_parameters(best_res.x)
        print(f"age: {self.age}")

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
    def mean(self) -> NDArray[float64]:
        """Get the current day mean of the component

        Returns
        -------
        ndarray of shape(6)
            The central mean of the component
        """
        mean_birth = self.parameters[0:6]
        mean_now = self.trace_orbit_func(
            mean_birth[np.newaxis], self.age
        ).squeeze()
        return mean_now

    @property
    def covariance(self) -> NDArray[float64]:
        """Get the current day covariance of the component

        Returns
        -------
        ndarray of shape(6, 6)
            The covariance of the component
        """
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
        """Get the age of the component

        Returns
        -------
        float
            The age of the component
        """
        return self.parameters[-1]

    def split(
        self
    ) -> tuple[SphereSpaceTimeComponent, SphereSpaceTimeComponent]:
        """Split the component along its primary phase-space axis

        Similar to :meth:`SpaceComponent.split`, however it is
        non-trivial to adjust covariance matrices which were
        transformed from birth-sites, so we don't bother.

        Returns
        -------
        tuple[SphereSpaceTimeComponent, SphereSpaceTimeComponent]
            Two components with identical parameters as `self` but
            with current-day means offset in either direction of
            the primary axis
        """

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
