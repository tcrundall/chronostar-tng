from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from numpy import float64
from typing import Any, Generator, Union

from ..context import chronostar     # noqa

from chronostar.base import (
    BaseComponent,
    BaseMixture,
    BaseICPool,
    BaseIntroducer,
    ScoredMixture,
)


CONFIG_PARAMS = {
    'icpool': {'a': 1, 'b': 2, 'c': 3},
    'introducer': {'a': 1, 'b': 2, 'c': 3},
    'mixture': {'a': 1, 'b': 2, 'c': 3},
    'component': {'a': 1, 'b': 2, 'c': 3},
}

NSAMPLES, NFEATURES = 100, 6
DATA = np.random.rand(NSAMPLES, NFEATURES)


class FooComponent(BaseComponent):
    dim = 6

    def __init__(self, params) -> None:        # type: ignore
        super().__init__(params)

    @classmethod
    def configure(cls, a=0, b=0, c=0, **kwargs):
        cls.a = a
        cls.b = b
        cls.c = c

        if kwargs:
            print(f"{cls} config: Extra keyword arguments provided:\n{kwargs}")

    @property
    def n_params(self) -> int:
        return 1

    def estimate_log_prob(self, X: NDArray[float64]) -> NDArray[float64]:
        return np.ones(X.shape[0])

    def maximize(
        self,
        X: NDArray[float64],
        log_resp: NDArray[float64]
    ) -> None:
        self.mean = np.ones(self.dim)
        self.covariance = np.eye(self.dim)

    def get_parameters(self) -> tuple:
        return (self.mean, self.covariance)

    def set_parameters(self, params: tuple) -> None:
        self.mean, self.covariance = params

    def split(self) -> tuple[FooComponent, FooComponent]:
        """Split this component into two, returning the result"""
        c1 = FooComponent(self.get_parameters())
        c2 = FooComponent(self.get_parameters())
        return c1, c2


class FooMixture(BaseMixture):
    def __init__(self, *args, **kwargs) -> None:        # type: ignore
        super().__init__(*args, **kwargs)

    @classmethod
    def configure(cls, **kwargs):
        if kwargs:
            print(f"{cls} config: Extra keyword arguments provided:\n{kwargs}")

    def get_parameters(self) -> tuple[NDArray[float64], list[BaseComponent]]:
        return (self.weights, self.comps)

    def set_parameters(
        self,
        params: tuple[NDArray[float64], list[BaseComponent]],
    ) -> None:
        self.weights, self.comps = params

    def fit(self, X: NDArray[float64]) -> None:
        self.memberships = np.ones((len(X), len(self.comps))) / len(self.comps)

    def bic(self, X: NDArray[float64]) -> float:
        """
        Calculates a quadratic based on number of components.
        Quadratic peaks at n=5
        """
        return -((len(self.comps) - 5)**2)

    def get_components(self) -> list[BaseComponent]:
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
        prev_components: Union[None, list[list[BaseComponent]], list[BaseComponent]],  # noqa E501
    ) -> list[list[BaseComponent]]:

        return [[FooComponent(params=None) for _ in range(5)]]


class FooICPool(BaseICPool):
    def __init__(self, *args, **kwargs) -> None:        # type: ignore
        super().__init__(*args, **kwargs)
        self.registry: dict[Union[str, int], ScoredMixture] = {}

    @classmethod
    def configure(cls, **kwargs):
        if kwargs:
            print(f"{cls} config: Extra keyword arguments provided:\n{kwargs}")

    def pool(self) -> Generator[tuple[int, list[BaseComponent]], None, None]:
        for i, fc in enumerate([FooComponent(params=None)]):
            yield (i, [fc])

    def register_result(
        self,
        unique_id: Union[str, int],
        mixture: BaseMixture,
        score: float,
    ) -> None:
        self.registry[unique_id] = ScoredMixture(mixture, score)

    @property
    def best_mixture(self) -> BaseMixture:
        mixture, _ = max(
            self.registry.values(), key=lambda x: x.score
        )
        return mixture
