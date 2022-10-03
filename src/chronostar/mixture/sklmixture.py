from typing import Any
from numpy.typing import NDArray
from numpy import float64
import numpy as np
from sklearn.mixture._base import BaseMixture as SKLBaseMixture

from src.chronostar.base import BaseComponent


class SKLComponentMixture(SKLBaseMixture):
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
        print(f"{init_params=}")
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

        self.weights_: NDArray[float64] = np.array(weights_init)
        self.components_: list[BaseComponent] = list(components_init)

        if kwargs:
            print("Extra arguments provided...")

    ##################################################
    #     Methods implemented by GaussianMixture     #
    ##################################################
    def _check_parameters(self, X: NDArray[float64]) -> None:
        pass

    def _initialize(
        self,
        X: NDArray[float64],
        resp: NDArray[float64]
    ) -> None:
        """
        Not implemented.

        Lets try to avoid automatic initialization
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
        resp = np.exp(log_resp)
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        self.weights_ = nk
        self.weights_ /= self.weights_.sum()
        for i, component in enumerate(self.components_):
            component.maximize(X=X, log_resp=log_resp[:, i])

    def _estimate_log_prob(self, X: NDArray[float64]) -> NDArray[float64]:
        n_samples = X.shape[0]
        n_components = len(self.components_)

        log_probs = np.zeros((n_samples, n_components))
        for k, component in enumerate(self.components_):
            log_probs[:, k] = component.estimate_log_prob(X)

        return log_probs

    def _estimate_log_weights(self) -> NDArray[float64]:
        return np.log(self.weights_)

    def _compute_lower_bound(self, _: Any, log_prob_norm: Any) -> Any:
        return log_prob_norm

    def _get_parameters(
        self,
    ) -> tuple[NDArray[float64], list[BaseComponent]]:

        return (self.weights_, self.components_)

    def _set_parameters(
        self,
        params: tuple[NDArray[float64], list[BaseComponent]],
    ) -> None:
        (self.weights_, self.components_) = params
        self.n_components = len(self.components_)

    def _n_parameters(self) -> int:
        n_params = 0
        for component in self.components_:
            n_params += component.n_params
        n_params += len(self.weights_) - 1
        return int(n_params)

    def bic(self, X: NDArray[float64]) -> float:
        return float(
            -2 * self.score(X) * X.shape[0]
            + self._n_parameters() * np.log(X.shape[0])
        )

    def aic(self, X: NDArray[float64]) -> float:
        return float(
            -2 * self.score(X) * X.shape[0]
            + 2 * self._n_parameters()
        )
