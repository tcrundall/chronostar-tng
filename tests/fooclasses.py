from __future__ import annotations

from numpy.typing import ArrayLike
import numpy as np

from src.chronostar.component.base import BaseComponent, Splittable
from src.chronostar.icpool.base import BaseICPool
from src.chronostar.introducer.base import BaseIntroducer
from src.chronostar.mixture.base import BaseMixture


CONFIG_PARAMS = {
    'icpool': {'a': 1, 'b': 2, 'c': 3},
    'introducer': {'a': 1, 'b': 2, 'c': 3},
    'mixture': {'a': 1, 'b': 2, 'c': 3},
    'component': {'a': 1, 'b': 2, 'c': 3},
}

NSAMPLES, NFEATURES = 100, 6
DATA = np.ones((NSAMPLES, NFEATURES))


class FooComponent(BaseComponent, Splittable):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def n_params(self) -> int:
        return 1

    def estimate_log_prob(self, X: ArrayLike) -> ArrayLike:
        return np.ones(X.shape[0])      # type: ignore

    def maximize(self, X: ArrayLike, log_resp: ArrayLike) -> None:
        return

    def split(self) -> tuple[FooComponent, FooComponent]:
        """Split this component into two, returning the result"""
        c1 = FooComponent(self.config_params)
        c2 = FooComponent(self.config_params)
        return c1, c2


class FooMixture(BaseMixture):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_params(self) -> list[BaseComponent]:
        return self._params

    def set_params(self, initial_conditions: list) -> None:
        self._params: list[BaseComponent] = initial_conditions

    def fit(self, data: ArrayLike) -> None:
        self.memberships = np.ones(data.shape) / data.shape[1]   # type: ignore

    def bic(self, data) -> float:
        """
        Calculates a quadratic based on number of components.
        Quadratic peaks at n=5
        """
        return -((len(self._params) - 5)**2)


class FooIntroducer(BaseIntroducer):
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


class FooICPool(BaseICPool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def pool(self) -> list[tuple[int, list[FooComponent]]]:
        return [
            (i, [fc])
            for i, fc in enumerate([FooComponent(self.config_params)])
        ]

    def register_result(self, unique_id, mixture, score) -> None:
        self.registry[unique_id] = (mixture, score)

    def best_mixture(self) -> FooMixture:
        return max(self.registry.values(), key=lambda x: x[1])[0]
