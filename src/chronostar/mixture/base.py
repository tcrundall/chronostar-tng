from abc import ABCMeta, abstractmethod
from typing import Any

from src.chronostar.component.base import BaseComponent
from numpy.typing import ArrayLike


class BaseMixture(metaclass=ABCMeta):
    def __init__(self, config_params: dict) -> None:
        self.config_params = config_params

    @abstractmethod
    def set_params(self, params: list[BaseComponent]) -> None:
        pass

    @abstractmethod
    def get_params(self) -> Any:
        pass

    @abstractmethod
    def fit(self, data: ArrayLike) -> None:
        pass

    @abstractmethod
    def bic(self, X) -> float:
        pass

    @abstractmethod
    def get_components(self) -> list[BaseComponent]:
        pass
