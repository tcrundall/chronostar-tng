from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Generator, NamedTuple, Union, Type, Any
import numpy as np
from numpy.typing import NDArray
from numpy import float64


class ScoredMixture(NamedTuple):
    mixture: BaseMixture
    score: float


class BaseICPool(metaclass=ABCMeta):

    def __init__(
        self,
        config_params: dict[Any, Any],
        introducer_class: Type[BaseIntroducer],
        component_class: Type[BaseComponent],
    ) -> None:
        self.config_params = config_params
        self.introducer_class = introducer_class
        self.component_class = component_class

        self.registry: dict[Union[str, int], ScoredMixture] = {}

    @abstractmethod
    def pool(self) -> Generator[tuple[int, list[BaseComponent]], None, None]:
        pass

    @abstractmethod
    def register_result(
        self,
        unique_id: Union[str, int],
        mixture: BaseMixture,
        score: float
    ) -> None:
        pass

    @property
    @abstractmethod
    def best_mixture(self) -> BaseMixture:
        pass


class BaseIntroducer(metaclass=ABCMeta):
    def __init__(
        self,
        config_params: dict[Any, Any],
        component_class: Type[BaseComponent],
    ) -> None:
        self.config_params = config_params
        self.component_class = component_class

    @abstractmethod
    def next_gen(
        self,
        prev_comp_sets: Union[
            list[list[BaseComponent]],
            list[BaseComponent],
            None
        ],
    ) -> list[list[BaseComponent]]:
        pass


class BaseComponent(metaclass=ABCMeta):
    def __init__(self, config_params: dict[Any, Any]) -> None:
        self.config_params = config_params

    @abstractmethod
    def estimate_log_prob(self, X: NDArray[float64]) -> NDArray[float64]:
        pass

    @abstractmethod
    def maximize(
        self,
        X: NDArray[float64],
        log_resp: NDArray[float64],
    ) -> None:
        pass

    @property
    @abstractmethod
    def n_params(self) -> int:
        pass

    def split(self) -> tuple[BaseComponent, BaseComponent]:
        params = self.get_parameters()
        mean = params[0]
        covariance = params[1]
        args = params[2:]

        # Get primary axis (longest eigen vector)
        eigvals, eigvecs = np.linalg.eigh(covariance)
        prim_axis_length = np.sqrt(np.max(eigvals))
        prim_axis = eigvecs[:, np.argmax(eigvals)]

        new_mean_1 = mean + prim_axis_length * prim_axis / 2.0
        new_mean_2 = mean - prim_axis_length * prim_axis / 2.0

        ###########################################################
        # Reconstruct covariance matrix but with halved prim axis #
        ###########################################################
        # Follows M . V = V . D
        #   where V := [v1, v2, ... vn]  (Matrix of eigvecs)
        #     and D is diagonal matrix where diagonals are eigvals
        new_eigvals = eigvals[:]
        new_eigvals[np.argmax(eigvals)] /= 4.0      # eigvals are std**2
        D = np.eye(6) * eigvals
        new_covariance = np.dot(eigvecs, np.dot(D, eigvecs.T))

        comp1 = self.__class__(self.config_params)
        comp1.set_parameters((new_mean_1, new_covariance, *args))
        comp2 = self.__class__(self.config_params)
        comp2.set_parameters((new_mean_2, new_covariance, *args))

        return comp1, comp2

    @abstractmethod
    def get_parameters(self) -> tuple:
        pass

    @abstractmethod
    def set_parameters(self, params: tuple) -> None:
        pass


class Splittable(metaclass=ABCMeta):
    @abstractmethod
    def split(self) -> tuple[Splittable, Splittable]:
        pass


class BaseMixture(metaclass=ABCMeta):
    def __init__(self, config_params: dict[Any, Any]) -> None:
        self.config_params = config_params

    @abstractmethod
    def set_params(
        self,
        params: tuple[NDArray[float64], list[BaseComponent]],
    ) -> None:
        pass

    @abstractmethod
    def get_params(self) -> Any:
        pass

    @abstractmethod
    def fit(self, data: NDArray[float64]) -> None:
        pass

    @abstractmethod
    def bic(self, X: NDArray[float64]) -> float:
        pass

    @abstractmethod
    def get_components(self) -> list[BaseComponent]:
        pass
