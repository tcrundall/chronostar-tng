from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Tuple
from numpy.typing import ArrayLike


class BaseComponent(metaclass=ABCMeta):
    def __init__(self, config_params) -> None:
        self.config_params = config_params

    @abstractmethod
    def estimate_log_prob(self, X: ArrayLike) -> ArrayLike:
        pass

    @abstractmethod
    def maximize(self, X: ArrayLike, log_resp: ArrayLike) -> None:
        pass


class Splittable(metaclass=ABCMeta):
    @abstractmethod
    def split(self) -> Tuple[Splittable, Splittable]:
        pass
