from __future__ import annotations
from typing import Optional
import numpy as np
from numpy import float64
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
from sklearn.mixture._gaussian_mixture import _estimate_log_gaussian_prob
from sklearn.mixture._gaussian_mixture import _estimate_gaussian_parameters

from ..base import BaseComponent


class SpaceComponent(BaseComponent):
    """A 6D phase-space Gaussian component

    Capable of fitting itself to a set of samples and
    responsibilities (membership probabilities)

    Parameters
    ----------
    params : ndarray of shape (42), optional
        Model parameters, the mean and covariance of a 6D
        Gaussian, flattened into a 1D array like so:
        ``np.hstack((mean, covariance.flatten()))``

    Attributes
    ----------
    reg_covar : float, default 1.e-6
        A regularisation constant added to the diagonals
        of the covariance matrix, configurable

    nthreads : int, default None
        Number of OMP threads used by numpy matrix operations
    """

    COVARIANCE_TYPE = "full"

    # Configurable attributes
    reg_covar: float = 1e-6
    nthreads: Optional[int] = None

    def __init__(self, params=None) -> None:
        super().__init__(params)

    def maximize(
        self,
        X: NDArray[float64],
        resp: NDArray[float64]
    ) -> None:
        """Find the best model parameters for the data and stores them
        in ``self.parameters``.

        You may access the parameters via the properties ``mean`` and
        ``covariance``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data

        resp : ndarray of shape (n_samples)
            component responsibilities (membership probabilities)

        Notes
        -----
        This method relies heavily on functions written by
        scikit-learn.

        References
        ----------
        TODO: Cite the authors of code which was used as inspiration
        """
        with threadpool_limits(self.nthreads, user_api='openmp'):
            nsamples = X.shape[0]
            if len(resp.shape) == 1:
                resp = resp[:, np.newaxis]
            assert resp.shape == (nsamples, 1)
            _, means_, covariances_ = _estimate_gaussian_parameters(
                X,
                resp,
                self.reg_covar,
                self.COVARIANCE_TYPE,
            )

            self.set_parameters(
                np.hstack((means_.flatten(), covariances_.flatten()))
            )

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

        n_features = len(self.mean)
        mean_params = n_features
        cov_params = n_features * (n_features + 1) / 2.0
        return int(mean_params + cov_params)

    def set_parameters(self, params: NDArray[float64]) -> None:
        """Set the internal parameters of the model.

        Parameters
        ----------
        params : (n_features), (n_features, n_features)
            mean, covariance
        """

        # Set this flag so we know whether parameters are available
        # This is used in :class:`sklmixture` to determine whether
        # or not components need to be automatically initialized
        self.parameters_set = True

        self.parameters = params
        self.precision_chol = _compute_precision_cholesky(
            self.covariance[np.newaxis], self.COVARIANCE_TYPE
        ).squeeze()

    def get_parameters(self) -> NDArray[float64]:
        """Get the internal parameters of the model

        Returns
        -------
        (n_features), (n_features, n_features)
            mean, covariance
        """

        return self.parameters

    @property
    def mean(self) -> NDArray[float64]:
        """Get the mean, as encoded in internal parameters

        Returns
        -------
        NDArray[float64] of shape(6)
            The central mean of the parameterised 6D Gaussian
        """
        return self.parameters[:6]

    @property
    def covariance(self) -> NDArray[float64]:
        """Get the covariance, as encoded in internal parameters

        Returns
        -------
        NDArray[float64] of shape(6, 6)
            The central covariance of the parameterised 6D Gaussian
        """
        return self.parameters[6:].reshape(6, 6)

    def split(self) -> tuple[SpaceComponent, SpaceComponent]:
        """Split this component into two new comps along the largest
        phase-space dimension

        This method finds the primary axis (i.e. largest eigen vector)
        and generates two components with means offset in either
        direction of the primary axis.

        The two new covariances are shrunk in the direction of the
        primary axis such that the two new ellipsoids formed by one
        standard deviation span the original ellipsoid formed by one
        standard deviation.

        Returns
        -------
        tuple[SpaceComponent, SpaceComponent]
            Two new components with offset mean and shrunken
            covariance
        """
        # Get primary axis (longest eigen vector)
        eigvals, eigvecs = np.linalg.eigh(self.covariance)
        prim_axis_length = np.sqrt(np.max(eigvals))
        prim_axis = eigvecs[:, np.argmax(eigvals)]

        new_mean_1 = self.mean + prim_axis_length * prim_axis / 2.0
        new_mean_2 = self.mean - prim_axis_length * prim_axis / 2.0

        ###########################################################
        # Reconstruct covariance matrix but with halved prim axis #
        ###########################################################
        # Follows M . V = V . D
        #   where V := [v1, v2, ... vn]  (Matrix of eigvecs)
        #     and D is diagonal matrix where diagonals are eigvals
        new_eigvals = np.copy(eigvals)
        new_eigvals[np.argmax(eigvals)] /= 4.0      # eigvals are std**2

        D = np.eye(6) * new_eigvals
        new_covariance = np.dot(eigvecs, np.dot(D, eigvecs.T))

        comp1 = self.__class__(np.hstack((
            new_mean_1,
            new_covariance.flatten(),
        )))
        assert hasattr(comp1, 'precision_chol')
        comp2 = self.__class__(np.hstack((
            new_mean_2,
            new_covariance.flatten(),
        )))

        return comp1, comp2
