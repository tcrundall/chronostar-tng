from abc import ABCMeta, abstractmethod
from typing import Type, Union

from ..component.base import BaseComponent
from ..mixture.base import BaseMixture


class BaseIntroducer(metaclass=ABCMeta):
    def __init__(
        self,
        config_params: dict,
        component_class: Type[BaseComponent],
    ) -> None:
        self.config_params = config_params
        self.component_class = component_class

    @abstractmethod
    def next_gen(
        self,
        prev_mixtures: Union[list[BaseMixture], BaseMixture, None],
    ) -> list[list[BaseComponent]]:
        pass
