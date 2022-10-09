from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import NamedTuple, Optional, Union, Type, Any
import numpy as np
from numpy.typing import NDArray
from numpy import float64


class ScoredMixture(NamedTuple):
    mixture: BaseMixture
    score: float


class BaseICPool(metaclass=ABCMeta):
    function_parser = {}

    def __init__(
        self,
        introducer_class: Type[BaseIntroducer],
        component_class: Type[BaseComponent],
    ) -> None:
        """_summary_

        Parameters
        ----------
        introducer_class : Type[BaseIntroducer]
            A class derived from BaseIntroducer
        component_class : Type[BaseComponent]
            A class derived from BaseComponent
        """
        self.introducer_class = introducer_class
        self.component_class = component_class

        self.registry: dict[Union[str, int], ScoredMixture] = {}

    @classmethod
    def configure(cls, **kwargs) -> None:
        for param, val in kwargs.items():
            if hasattr(cls, param):
                if val in cls.function_parser:
                    setattr(cls, param, cls.function_parser[val])
                else:
                    setattr(cls, param, val)
            else:
                print(f"[CONFIG]:{cls} unexpected config param: {param}={val}")

    @abstractmethod
    def has_next(self) -> bool:
        pass

    @abstractmethod
    def get_next(self) -> tuple[Union[str, int], list[BaseComponent]]:
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
    function_parser = {}

    def __init__(
        self,
        component_class: Type[BaseComponent],
    ) -> None:
        """Abstract base class for Introducer objects

        Parameters
        ----------
        component_class : Type[BaseComponent]
            A derived class from BaseComponent
        """
        self.component_class = component_class

    @classmethod
    def configure(cls, **kwargs) -> None:
        for param, val in kwargs.items():
            if hasattr(cls, param):
                if val in cls.function_parser:
                    setattr(cls, param, cls.function_parser[val])
                else:
                    setattr(cls, param, val)
            else:
                print(f"[CONFIG]:{cls} unexpected config param: {param}={val}")

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
    """Abstract class for a (assumed-to-be Gaussian)
    component to be used in a mixture model

    Capable of fitting itself to a set of samples and
    responsibilities (membership probabilities)
    """
    function_parser = {}

    def __init__(self, params: Optional[tuple] = None) -> None:
        if params:
            self.set_parameters(params)

    @classmethod
    def configure(cls, **kwargs) -> None:
        for param, val in kwargs.items():
            if hasattr(cls, param):
                if val in cls.function_parser:
                    setattr(cls, param, cls.function_parser[val])
                else:
                    setattr(cls, param, val)
            else:
                print(f"[CONFIG]:{cls} unexpected config param: {param}={val}")

    @abstractmethod
    def estimate_log_prob(self, X: NDArray[float64]) -> NDArray[float64]:
        pass

    @abstractmethod
    def maximize(
        self,
        X: NDArray[float64],
        log_resp: NDArray[float64],
    ) -> None:
        print("What...?")
        pass

    @property
    @abstractmethod
    def n_params(self) -> int:
        pass

    def split(self) -> tuple[BaseComponent, BaseComponent]:
        """Split this component in half along primary axis
        in feature space.

        Notes
        -----
        This method generates two components, identical to `self`
        but with half the width along the primary axis and means
        offset in direction of primary axis such that
        new_mean = old_mean +/- prim_axis_length/2

        Returns
        -------
        tuple[BaseComponent, BaseComponent]
            Two components with identical parameters except
            half as wide and offset from mean
        """
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
        new_eigvals = np.copy(eigvals)
        new_eigvals[np.argmax(eigvals)] /= 4.0      # eigvals are std**2

        D = np.eye(6) * new_eigvals
        new_covariance = np.dot(eigvecs, np.dot(D, eigvecs.T))

        comp1 = self.__class__((new_mean_1, new_covariance, *args))
        comp2 = self.__class__((new_mean_2, new_covariance, *args))

        return comp1, comp2

    @abstractmethod
    def get_parameters(self) -> tuple:
        pass

    @abstractmethod
    def set_parameters(self, params: tuple) -> None:
        pass


class BaseMixture(metaclass=ABCMeta):

    function_parser = {}

    def __init__(
        self,
        init_weights: NDArray[float64],
        init_comps: list[BaseComponent],
    ) -> None:
        self.init_comps = init_comps
        self.init_weights = init_weights

    @classmethod
    def configure(cls, **kwargs) -> None:
        for param, val in kwargs.items():
            if hasattr(cls, param):
                if val in cls.function_parser:
                    setattr(cls, param, cls.function_parser[val])
                else:
                    setattr(cls, param, val)
            else:
                print(f"[CONFIG]:{cls} unexpected config param: {param}={val}")

    @abstractmethod
    def set_parameters(
        self,
        params: tuple[NDArray[float64], list[BaseComponent]],
    ) -> None:
        pass

    @abstractmethod
    def get_parameters(self) -> Any:
        pass

    @abstractmethod
    def fit(self, X: NDArray[float64]) -> None:
        pass

    @abstractmethod
    def bic(self, X: NDArray[float64]) -> float:
        pass

    @abstractmethod
    def get_components(self) -> list[BaseComponent]:
        pass

    @abstractmethod
    def estimate_membership_prob(
        self, X: NDArray[float64]
    ) -> NDArray[float64]:
        pass
