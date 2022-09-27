from abc import ABCMeta, abstractmethod
from typing import Type, Union

from ..component.base import BaseComponent


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
        prev_comp_sets: Union[
            list[list[BaseComponent]],
            list[BaseComponent],
            None
        ],
    ) -> list[list[BaseComponent]]:
        pass
