from abc import ABCMeta, abstractmethod
from typing import Union
from ..mixture.base import BaseMixture
from ..introducer.base import BaseIntroducer
from ..component.base import BaseComponent


class BaseICPool(metaclass=ABCMeta):

    def __init__(
        self,
        config_params: dict,
        introducer_class: BaseIntroducer,
        component_class: BaseComponent,
    ) -> None:
        self.config_params = config_params
        self.introducer_class = introducer_class
        self.component_class = component_class

        self.registry = {}

    @abstractmethod
    def pool():
        pass

    @abstractmethod
    def register_result(
        self,
        unique_id: Union[str, int],
        mixture: BaseMixture,
        score: float
    ) -> None:
        pass
