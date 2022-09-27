from dataclasses import dataclass
import numpy as np


@dataclass
class FooComponent:
    config_params: dict


@dataclass
class FooMixture:
    config_params: dict

    def set_params(self, initial_conditions: list) -> None:
        self._params = initial_conditions

    def fit(self, data) -> None:
        self.memberships = np.ones(data.shape) / data.shape[1]
        pass

    def bic(self, data) -> float:
        return 10.


@dataclass
class FooIntroducer:
    config_params: dict


class FooICPool:
    def __init__(
        self,
        config_params: dict,
        introducer_class,
        component_class,
    ) -> None:
        self.config_params: dict = config_params
        self.introducer_class = introducer_class
        self.component_class = component_class
        self.registry = {}

    def pool(self) -> list[tuple[int, FooComponent]]:
        return [
            (i, fc)
            for i, fc in enumerate([FooComponent(self.config_params)])
        ]

    def register_result(self, unique_id, mixture, score) -> None:
        self.registry[unique_id] = (mixture, score)

    def best_mixture(self) -> FooMixture:
        return max(self.registry.values(), key=lambda x: x[1])[0]
