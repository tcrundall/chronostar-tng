from typing import Any, Tuple
import numpy as np
from numpy import float64
from numpy.typing import NDArray

from ..base import BaseComponent, BaseMixture
from .sklmixture import SKLComponentMixture


class ComponentMixture(BaseMixture):
    """A mixture model of arbitrary components

    Parameters
    ----------
    init_weights : NDArray[float64] of shape (n_components)
        Initial weights of the components
    init_components : list[BaseComponent]
        Component objects which will be maximised to the data,
        optionally with pre-initialised parameters
    """

    def __init__(
        self,
        init_weights: NDArray[float64],
        init_components: list[BaseComponent],
    ) -> None:
        # Can handle extra parameters if I want...
        self.sklmixture = SKLComponentMixture(
            init_weights,
            init_components,
            tol=self.tol,
            reg_covar=self.reg_covar,
            max_iter=self.max_iter,
            n_init=self.n_init,
            init_params=self.init_params,
            random_state=self.random_state,
            warm_start=self.warm_start,
            verbose=self.verbose,
            verbose_interval=self.verbose_interval,
        )

    @classmethod
    def configure(
        cls,
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: str = 'random',
        random_state: Any = None,
        warm_start: bool = True,
        verbose: int = 0,
        verbose_interval: int = 10,
        **kwargs
    ):
        """Configure class level parameters

        Most of these are passed on to the SKLMixture object

        Parameters
        ----------
        tol : float, optional
            Some tolerance used by sklearn... , by default 1e-3
        reg_covar : float, optional
            Regularisation factor added to diagonal elements of
            covariance matrices, by default 1e-6
        max_iter : int, optional
            Maximum iterations of EM algorithm, by default 100
        n_init : int, optional
            sklearn parameter we don't use, by default 1
        init_params : str, optional
            How to initialise components if not already set,
            'random' assigns memberships randomly then maximises,
            by default 'random'
        random_state : Any, optional
            sklearn parameter... the random seed?, by default None
        warm_start : bool, optional
            sklearn parameter that we don't use, by default True
        verbose : int, optional
            sklearn parameter..., by default 0
        verbose_interval : int, optional
            sklearn parameter, by default 10
        """

        cls.tol = tol
        cls.reg_covar = reg_covar
        cls.max_iter = max_iter
        cls.n_init = n_init
        cls.init_params = init_params
        cls.random_state = random_state
        cls.warm_start = warm_start
        cls.verbose = verbose
        cls.verbose_interval = verbose_interval

        if kwargs:
            print(f"{cls} config: Extra keyword arguments provided:\n{kwargs}")

    def fit(self, X: NDArray[float64]) -> None:
        """Fit the mixture model to the input data

        Parameters
        ----------
        X : NDArray[float64] of shape (n_samples, n_features)
            Input data
        """

        self.sklmixture.fit(X)

    def bic(self, X: NDArray[float64]) -> float:
        """Calculate the Bayesian Information Criterion

        Parameters
        ----------
        X : NDArray[float64] of shape (n_samples, n_features)
            Input data

        Returns
        -------
        float
            The calculated BIC
        """

        return float(self.sklmixture.bic(X))

    def set_parameters(
        self,
        params: Tuple[NDArray[float64], list[BaseComponent]],
    ) -> None:
        """Set the parameters that characterise the mixture

        Parameters
        ----------
        params: tuple[NDArray[float64], list[BaseComponent]]
            The weights of the components and the component objects
        """

        self.sklmixture._set_parameters(params)

    def get_parameters(self) -> tuple[NDArray[float64], list[BaseComponent]]:
        """Get the parameters that characterise the mixture

        Returns
        -------
        tuple[NDArray[float64], list[BaseComponent]]
            The weights of the components and the component objects
        """

        return self.sklmixture._get_parameters()

    def get_components(self) -> list[BaseComponent]:
        """Get the list of components fitted to the data

        Returns
        -------
        list[BaseComponent]
            The list of components
        """
        _, components = self.get_parameters()
        return components

    def estimate_membership_prob(self, X: NDArray[float64]):
        """Estimate the membership probabilities of each sample to
        each component

        This method assumes the mixture has already been fit with
        :meth:`fit`

        Parameters
        ----------
        X : NDArray[float64] of shape (n_samples, n_features)
            Input data

        Returns
        -------
        NDArray[float64] of shape (n_samples, n_components)
            The membership probabilities of each sample to each
            component.
        """
        weighted_log_prob = self.sklmixture._estimate_weighted_log_prob(X)

        # Take exponent
        weighted_prob = np.exp(weighted_log_prob)

        # Normalize such that each row sums to 1
        return (weighted_prob.T / weighted_prob.sum(axis=1)).T
