import numpy as np
from numpy import float64
from numpy.typing import NDArray

from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
from sklearn.mixture._gaussian_mixture import _estimate_log_gaussian_prob
from sklearn.mixture._gaussian_mixture import _estimate_gaussian_parameters

from src.chronostar.base import BaseComponent


class SpaceComponent(BaseComponent):
    """A 6D phase-space Gaussian component

    Capable of fitting itself to a set of samples and
    responsibilities (membership probabilities)
    """

    COVARIANCE_TYPE = "full"

    @classmethod
    def configure(
        cls,
        reg_covar=1e-6,
        **kwargs
    ):
        r"""Set class level configuration parameters that will be
        carried through to all instances.

        Parameters
        ----------
        reg_covar : float, optional
            A regularization constant added to the diagonal elements
            of covariance matrices to ensure invertability, by default
            1e-6
        """

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
        """Find the best model parameters for the data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data
        log_resp : ndarray of shape (n_samples)
            log of component responsibilities (membership probabilities)

        Notes
        -----
        This method relies heavily on functions written by
        scikit-learn.

        References
        ----------
        TODO: Cite the authors of code which was used as inspiration
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

    def set_parameters(self, params: tuple) -> None:
        """Set the internal parameters of the model.

        Parameters
        ----------
        params : (n_features), (n_features, n_features)
            mean, covariance
        """

        self.mean, self.covariance = params
        self.precision_chol = _compute_precision_cholesky(
            self.covariance[np.newaxis], self.COVARIANCE_TYPE
        ).squeeze()

    def get_parameters(self) -> tuple:
        """Get the internal parameters of the model

        Returns
        -------
        (n_features), (n_features, n_features)
            mean, covariance
        """

        return self.mean, self.covariance
