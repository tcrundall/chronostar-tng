from __future__ import annotations
from typing import Optional
import numpy as np
from numpy import float64
from numpy.typing import NDArray

from ..base import BaseComponent


class UniformComponent(BaseComponent):
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
    """

    COVARIANCE_TYPE = "full"

    # Configurable attributes
    reg_covar: float = 1e-6
    nthreads: Optional[int] = None

    def __init__(self, params=None) -> None:
        super().__init__(params)
        if params is None:
            raise UserWarning("UniformComponent must be initialised with a density")

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
        pass

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
        return np.tile(np.log(self.density), len(X))

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
        return 1

    @property
    def density(self) -> float:
        return self.parameters[0]

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

    def get_parameters(self) -> NDArray[float64]:
        """Get the internal parameters of the model

        Returns
        -------
        (n_features), (n_features, n_features)
            mean, covariance
        """

        return self.parameters

    def split(self):
        raise UserWarning("Not splittable")
