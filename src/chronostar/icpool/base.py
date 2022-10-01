from abc import ABCMeta, abstractmethod
from typing import Generator, Union, Type
from ..mixture.base import BaseMixture
from ..introducer.base import BaseIntroducer
from ..component.base import BaseComponent


class BaseICPool(metaclass=ABCMeta):

    def __init__(
        self,
        config_params: dict,
        introducer_class: Type[BaseIntroducer],
        component_class: Type[BaseComponent],
    ) -> None:
        self.config_params = config_params
        self.introducer_class = introducer_class
        self.component_class = component_class

        self.registry = {}

    @abstractmethod
    def pool() -> Generator[tuple[int, list[BaseComponent]], None, None]:
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
