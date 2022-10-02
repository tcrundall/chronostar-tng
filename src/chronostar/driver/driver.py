from pathlib import Path
from typing import Any, Type, Union
import yaml

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from src.chronostar.base import (
    BaseComponent,
    BaseMixture,
    BaseICPool,
    BaseIntroducer,
)


class Driver:
    def __init__(
        self,
        config_file: Union[str, Path],
        mixture_class: Type[BaseMixture],
        icpool_class: Type[BaseICPool],
        introducer_class: Type[BaseIntroducer],
        component_class: Type[BaseComponent],
    ) -> None:
        """Constructor method"""

        self.config_params = self.read_config_file(config_file)

        self.component_class = component_class
        self.mixture_class = mixture_class
        self.icpool_class = icpool_class
        self.introducer_class = introducer_class

    def run(self, data: NDArray[float64]) -> BaseMixture:

        icpool = self.icpool_class(
            config_params=self.config_params['icpool'],
            introducer_class=self.introducer_class,
            component_class=self.component_class,
        )

        for unique_id, init_conds in icpool.pool():
            m = self.mixture_class(self.config_params)
            ncomps = len(init_conds)
            m.set_params((np.ones(ncomps) / ncomps, init_conds))
            m.fit(data)
            icpool.register_result(unique_id, m, m.bic(data))

        # loop will end when icg stops generating reasonable initial conditions
        return icpool.best_mixture

    def read_config_file(
        self,
        config_file: Union[str, Path]
    ) -> dict[Any, Any]:
        with open(config_file, "r") as stream:
            try:
                config_params = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise exc

        assert isinstance(config_params, dict)
        return config_params
