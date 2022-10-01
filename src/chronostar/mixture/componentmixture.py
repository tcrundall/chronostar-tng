from numpy import float64

from src.chronostar.component.base import BaseComponent
from src.chronostar.mixture.sklmixture import SKLComponentMixture
# from tests.fooclasses import CONFIG_PARAMS, DATA
from .base import BaseMixture

SKL_DEFAULT_PARAMS = {
    'n_components': 1,
    'tol': 1e-3,
    'reg_covar': 1e-6,
    'max_iter': 100,
    'n_init': 1,
    'init_params': 'kmeans',
    'random_state': None,
    'warm_start': False,
    'verbose': 0,
    'verbose_interval': 10,
}


class ComponentMixture(BaseMixture):
    def __init__(
        self,
        config_params: dict,
        init_weights: list[float],
        init_components: list[BaseComponent],
    ) -> None:
        # Can handle extra parameters if I want...
        self.sklmixture = SKLComponentMixture(init_weights, init_components)
        super().__init__(config_params)

    def fit(self, X) -> None:
        self.sklmixture.fit(X)

    def bic(self, X) -> float64:
        return self.sklmixture.bic(X)

    def set_params(self, params):
        self.sklmixture._set_parameters(params)

    def get_params(self):
        return self.sklmixture._get_parameters()

    def get_components(self) -> list[BaseComponent]:
        _, components = self.get_params()
        return components

# if __name__ == '__main__':
#     cm = ComponentMixture(CONFIG_PARAMS)
#     cm.fit(DATA)
