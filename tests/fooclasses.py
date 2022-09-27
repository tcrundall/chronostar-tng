from numpy.typing import ArrayLike
import numpy as np

from src.chronostar.component.base import BaseComponent
from src.chronostar.mixture.base import BaseMixture

try:
    from .context import chronostar as c
except ImportError:
    from context import chronostar as c


class FooComponent(BaseComponent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def estimate_log_prob(self, X: ArrayLike) -> ArrayLike:
        return np.ones(X.shape[0])      # type: ignore

    def maximize(self, X: ArrayLike, log_resp: ArrayLike) -> None:
        return


class FooMixture(BaseMixture):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def set_params(self, initial_conditions: list) -> None:
        self._params = initial_conditions

    def fit(self, data: ArrayLike) -> None:
        self.memberships = np.ones(data.shape) / data.shape[1]   # type: ignore

    def bic(self, data) -> float:
        return 10.


class FooIntroducer(c.introducer.base.BaseIntroducer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def next_gen(self, prev_mixtures) -> list[BaseMixture]:
        if prev_mixtures is None:
            m = FooMixture(self.config_params)
            m.set_params([self.component_class(self.config_params)])
            return [m]
        elif isinstance(prev_mixtures, BaseMixture):
            return [prev_mixtures]
        else:
            return prev_mixtures


class FooICPool(c.icpool.base.BaseICPool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def pool(self) -> list[tuple[int, FooComponent]]:
        return [
            (i, fc)
            for i, fc in enumerate([FooComponent(self.config_params)])
        ]

    def register_result(self, unique_id, mixture, score) -> None:
        self.registry[unique_id] = (mixture, score)

    def best_mixture(self) -> FooMixture:
        return max(self.registry.values(), key=lambda x: x[1])[0]
