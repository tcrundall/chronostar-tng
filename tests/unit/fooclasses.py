from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from numpy import float64
from typing import Any, Optional, Union

from ..context import chronostar     # noqa

from chronostar.base import (
    BaseComponent,
    BaseMixture,
    BaseICPool,
    BaseIntroducer,
    ScoredMixture,
    InitialCondition,
)


CONFIG_PARAMS: dict = {
    'icpool': {'a': 1, 'b': 2, 'c': 3},
    'introducer': {'a': 1, 'b': 2, 'c': 3},
    'mixture': {'a': 1, 'b': 2, 'c': 3},
    'component': {
        'reg_covar': 1e-3,
        'nthreads': 1,
    },
    'run': {
        'nthreads': 1,
    }
}

NSAMPLES, NFEATURES = 100, 6
DATA = np.random.rand(NSAMPLES, NFEATURES) * 10.


class FooComponent(BaseComponent):
    dim = 6

    def __init__(
        self,
        params: Optional[NDArray[float64]],
    ) -> None:
        super().__init__(params)

    @property
    def n_params(self) -> int:
        return 1

    def estimate_log_prob(self, X: NDArray[float64]) -> NDArray[float64]:
        return np.ones(X.shape[0])

    def maximize(
        self,
        X: NDArray[float64],
        resp: NDArray[float64]
    ) -> None:
        mean = np.ones(self.dim)
        covariance = np.eye(self.dim)
        self.set_parameters(
            np.hstack((
                mean, covariance.flatten(),
            ))
        )

    def get_parameters(self) -> NDArray[float64]:
        return np.hstack((self.mean, self.covariance.flatten()))

    def set_parameters(self, params: NDArray[float64]) -> None:
        self.parameters = params

    def split(self) -> tuple[FooComponent, FooComponent]:
        """Split this component into two, returning the result"""
        c1 = FooComponent(self.get_parameters())
        c2 = FooComponent(self.get_parameters())
        return c1, c2

    @property
    def mean(self) -> NDArray[float64]:
        return self.parameters[:6]

    @property
    def covariance(self) -> NDArray[float64]:
        return self.parameters[6:].reshape(6, 6)


class FooMixture(BaseMixture):
    def __init__(self, *args, **kwargs) -> None:        # type: ignore
        super().__init__(*args, **kwargs)
        self.comps = self.init_comps
        self.weights = self.init_weights

    @classmethod
    def configure(cls, **kwargs):
        if kwargs:
            print(f"{cls} config: Extra keyword arguments provided:\n{kwargs}")

    def get_parameters(self) -> tuple[NDArray[float64], tuple[BaseComponent, ...]]:
        return (self.weights, self.comps)

    def set_parameters(
        self,
        params: tuple[NDArray[float64], tuple[BaseComponent, ...]],
    ) -> None:
        self.weights, self.comps = params

    def fit(self, X: NDArray[float64]) -> None:
        self.memberships = np.ones((len(X), len(self.comps))) / len(self.comps)
        for c in self.comps:
            c.maximize(X, self.memberships)

    def bic(self, X: NDArray[float64]) -> float:
        """
        Calculates a quadratic based on number of components.
        Quadratic peaks at n=5
        """
        return -((len(self.comps) - 5)**2)

    def get_components(self) -> tuple[BaseComponent, ...]:
        return self.get_parameters()[1]

    def estimate_membership_prob(
        self,
        X: NDArray[float64]
    ) -> NDArray[float64]:
        return self.memberships


class FooIntroducer(BaseIntroducer):
    def __init__(
        self,
        *args: tuple[Any],
        **kwargs: dict[Any, Any],
    ) -> None:
        super().__init__(*args, **kwargs)       # type: ignore

    @classmethod
    def configure(cls, **kwargs) -> None:
        if kwargs:
            print(f"{cls} config: Extra keyword arguments provided:\n{kwargs}")

    def next_gen(
        self,
        prev_components: Union[None, list[InitialCondition], InitialCondition],
    ) -> list[InitialCondition]:
        init_comps = tuple([FooComponent(params=None) for _ in range(5)])
        label = 'fooic_5'
        return [InitialCondition(label, init_comps)]


class FooICPool(BaseICPool):
    def __init__(self, *args, **kwargs) -> None:        # type: ignore
        super().__init__(*args, **kwargs)
        self.registry: dict[Union[str, int], ScoredMixture] = {}

        self.queue = [InitialCondition('0', tuple([FooComponent(params=None)]))]

    @classmethod
    def configure(cls, **kwargs):
        if kwargs:
            print(f"{cls} config: Extra keyword arguments provided:\n{kwargs}")

    def has_next(self):
        return len(self.queue) > 0

    def get_next(self):
        return self.queue.pop()

    def provide_start(self, init_conds):
        self.queue.append(init_conds)

    def register_result(
        self,
        label: str,
        mixture: BaseMixture,
        score: float,
    ) -> None:
        self.registry[label] = ScoredMixture(mixture, score, label)

    @property
    def best_mixture(self) -> BaseMixture:
        mixture, _, _ = max(
            self.registry.values(), key=lambda x: x.score
        )
        return mixture
