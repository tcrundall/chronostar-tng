from typing import Any, Tuple
from numpy import float64
from numpy.typing import NDArray

from src.chronostar.base import BaseComponent, BaseMixture
from src.chronostar.mixture.sklmixture import SKLComponentMixture


# SKL_DEFAULT_PARAMS = {
#     'n_components': 1,
#     'tol': 1e-3,
#     'reg_covar': 1e-6,
#     'max_iter': 100,
#     'n_init': 1,
#     'init_params': 'kmeans',
#     'random_state': None,
#     'warm_start': False,
#     'verbose': 0,
#     'verbose_interval': 10,
# }


class ComponentMixture(BaseMixture):
    def __init__(
        self,
        config_params: dict[Any, Any],
        init_weights: NDArray[float64],
        init_components: list[BaseComponent],
    ) -> None:
        # Can handle extra parameters if I want...
        self.sklmixture = SKLComponentMixture(init_weights, init_components)
        # self.sklmixture = SKLComponentMixture()
        self.config_params = config_params

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
