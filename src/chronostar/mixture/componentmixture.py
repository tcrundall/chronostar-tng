from typing import Any, Tuple
from numpy import float64
from numpy.typing import NDArray

from src.chronostar.base import BaseComponent, BaseMixture
from src.chronostar.mixture.sklmixture import SKLComponentMixture


class ComponentMixture(BaseMixture):
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
        self.sklmixture.fit(X)

    def bic(self, X: NDArray[float64]) -> float:
        return float(self.sklmixture.bic(X))

    def set_params(
        self,
        params: Tuple[NDArray[float64], list[BaseComponent]],
    ) -> None:
        self.sklmixture._set_parameters(params)

    def get_params(self) -> tuple[NDArray[float64], list[BaseComponent]]:
        return self.sklmixture._get_parameters()

    def get_components(self) -> list[BaseComponent]:
        _, components = self.get_params()
        return components
