from abc import ABCMeta, abstractmethod

from src.chronostar.component.base import BaseComponent
from numpy.typing import ArrayLike


class BaseMixture(metaclass=ABCMeta):
    def __init__(self, config_params: dict) -> None:
        self.config_params = config_params

    @abstractmethod
    def set_params(self, params: list[BaseComponent]):
        pass

    @abstractmethod
    def fit(self, data: ArrayLike):
        pass
