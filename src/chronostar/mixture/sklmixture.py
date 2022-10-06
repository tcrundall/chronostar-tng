from typing import Any
from numpy.typing import NDArray
from numpy import float64
import numpy as np
from sklearn.mixture._base import BaseMixture as SKLBaseMixture

from src.chronostar.base import BaseComponent


class SKLComponentMixture(SKLBaseMixture):
    """A derived class utilising much from scikit-learn
    to fit a Gaussian Mixture Model
    """
    def __init__(
        self,
        weights_init: NDArray[float64],
        components_init: list[BaseComponent],
        *,
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: str = 'random',
        random_state: Any = None,
        warm_start: bool = True,
        verbose: int = 0,
        verbose_interval: int = 10,
        **kwargs: dict[str, Any],
    ) -> None:
        """Constructor method

        Parameters
        ----------
        weights_init : NDArray[float64] of shape (n_components)
            Initial weights of each component, ideally normalized
            to sum to 1
        components_init : list[BaseComponent]
            Component objects which will be maximised to the data,
            optionally with pre-initialised parameters
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
        super().__init__(
            n_components=len(components_init),
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            random_state=random_state,
            warm_start=warm_start,
            verbose=verbose,
            verbose_interval=verbose_interval,
        )

        self.weights_: NDArray[float64] = weights_init
        self.components_: list[BaseComponent] = components_init

        if kwargs:
            print("Extra arguments provided...")

    ##################################################
    #     Methods implemented by GaussianMixture     #
    ##################################################
    def _check_parameters(self, X: NDArray[float64]) -> None:
        """Override to do nothing

        Parameters
        ----------
        X : NDArray[float64]
            data
        """

        pass

    def _initialize(
        self,
        X: NDArray[float64],
        resp: NDArray[float64]
    ) -> None:
        """Initialize component parameters if not already set

        Parameters
        ----------
        X : NDArray[float64] of shape (n_samples, n_features)
            Input data
        resp : NDArray[float64] of shape (n_samples, n_components)
            Responsibilities (membership probabilities) of each sample
            to each component

        TODO: Actually only initialise randomly when "random" is set
        """
        # if self.init_params == "random":

        if any([not hasattr(comp, 'mean') for comp in self.components_]):
            nsamples = X.shape[0]
            resp = np.random.rand(nsamples, self.n_components)
            resp = (resp.T / resp.sum(axis=1)).T
            for i, component in enumerate(self.components_):
                component.maximize(X, np.log(resp[:, i]))

    def _m_step(
        self,
        X: NDArray[float64],
        log_resp: NDArray[float64],
    ) -> None:
        """EM's M-step, maximise model parameters based on
        data and estimated responsibilities

        Parameters
        ----------
        X : NDArray[float64] of shape (n_samples, n_features)
            Input data
        log_resp : NDArray[float64] of shape (n_samples, n_components)
            Log responsibilities (membership probabilities) of each sample
            to each component
        """

        resp = np.exp(log_resp)
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        self.weights_ = nk
        self.weights_ /= self.weights_.sum()

        for i, component in enumerate(self.components_):
            component.maximize(X=X, log_resp=log_resp[:, i])

    def _estimate_log_prob(self, X: NDArray[float64]) -> NDArray[float64]:
        """Estimate the log probability of each sample for each
        component

        Parameters
        ----------
        X : NDArray[float64] of shape (n_samples, n_features)
            Input data

        Returns
        -------
        NDArray[float64] of shape (n_samples, n_components)
            Log probabilities
        """
        n_samples = X.shape[0]
        n_components = len(self.components_)

        log_probs = np.zeros((n_samples, n_components))
        for k, component in enumerate(self.components_):
            log_probs[:, k] = component.estimate_log_prob(X)

        return log_probs

    def _estimate_log_weights(self) -> NDArray[float64]:
        """Estimate the log of the component weights

        Returns
        -------
        NDArray[float64] of shape (ncomponents)
            The log of the component weights
        """
        return np.log(self.weights_)

    def _compute_lower_bound(self, _: Any, log_prob_norm: Any) -> Any:
        """Used internally by sklearn

        Parameters
        ----------
        _ : Any
            Argument only here to match API
        log_prob_norm : Any
            Probably a float

        Returns
        -------
        Any
            Same as input
        """
        return log_prob_norm

    def _get_parameters(
        self,
    ) -> tuple[NDArray[float64], list[BaseComponent]]:
        """Get the parameters that characterise the mixture

        Returns
        -------
        tuple[NDArray[float64], list[BaseComponent]]
            The weights of the components and the component objects
        """

        return (self.weights_, self.components_)

    def _set_parameters(
        self,
        params: tuple[NDArray[float64], list[BaseComponent]],
    ) -> None:
        """Set the parameters that characterise the mixture

        Parameters
        ----------
        params: tuple[NDArray[float64], list[BaseComponent]]
            The weights of the components and the component objects
        """
        (self.weights_, self.components_) = params
        self.n_components = len(self.components_)

    def _n_parameters(self) -> int:
        """Get the number of parameters used to characterise the
        mixture

        Used for BIC and AIC calculations

        Returns
        -------
        int
            The number of parameters used to characterise the
            mixture
        """
        n_params = 0
        for component in self.components_:
            n_params += component.n_params
        n_params += len(self.weights_) - 1
        return int(n_params)

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
        return float(
            -2 * self.score(X) * X.shape[0]
            + self._n_parameters() * np.log(X.shape[0])
        )

    def aic(self, X: NDArray[float64]) -> float:
        """Calculate the Akaike Information Criterion

        Parameters
        ----------
        X : NDArray[float64] of shape (n_samples, n_features)
            Input data

        Returns
        -------
        float
            The calculated AIC
        """
        return float(
            -2 * self.score(X) * X.shape[0]
            + 2 * self._n_parameters()
        )
