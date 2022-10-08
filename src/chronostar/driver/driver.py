from collections import defaultdict
from pathlib import Path
from typing import Any, Type, Union
import yaml

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from ..base import (
    BaseComponent,
    BaseMixture,
    BaseICPool,
    BaseIntroducer,
)

from ..component.spacetimecomponent import SpaceTimeComponent
from ..mixture.componentmixture import ComponentMixture
from ..introducer.simpleintroducer import SimpleIntroducer
from ..icpool.simpleicpool import SimpleICPool


class Driver:
    def __init__(
        self,
        config_file: Union[str, Path],
        mixture_class: Type[BaseMixture] = ComponentMixture,
        icpool_class: Type[BaseICPool] = SimpleICPool,
        introducer_class: Type[BaseIntroducer] = SimpleIntroducer,
        component_class: Type[BaseComponent] = SpaceTimeComponent,
    ) -> None:
        """Top level class of Chronostar which drives the
        entire fitting process

        Parameters
        ----------
        config_file : Union[str, Path]
            A yaml configuration file with sections mixture, icpool,
            introducer and component.
        mixture_class : Type[BaseMixture]
            A class derived from BaseMixture
        icpool_class : Type[BaseICPool]
            A class derived from BaseICPool
        introducer_class : Type[BaseIntroducer]
            A class derived from BaseIntroducer
        component_class : Type[BaseComponent]
            A class derived from BaseComponent
        """

        self.config_params = defaultdict(dict)

        self.config_params.update(self.read_config_file(config_file))

        self.component_class = component_class
        self.mixture_class = mixture_class
        self.icpool_class = icpool_class
        self.introducer_class = introducer_class

        self.component_class.configure(**self.config_params["component"])
        self.mixture_class.configure(**self.config_params["mixture"])
        self.icpool_class.configure(**self.config_params["icpool"])
        self.introducer_class.configure(**self.config_params["introducer"])

    def run(self, data: NDArray[float64]) -> BaseMixture:
        """Run a fit on the input data

        Parameters
        ----------
        data : NDArray[float64] of shape (n_samples, n_features)
            The input data

        Returns
        -------
        BaseMixture
            A mixture object containing the best fitting parameters,
            retrievable by mixture.get_parameters()
        """

        icpool = self.icpool_class(
            introducer_class=self.introducer_class,
            component_class=self.component_class,
        )

        for unique_id, init_conds in icpool.pool():
            ncomps = len(init_conds)
            init_weights = np.ones(ncomps)/ncomps
            m = self.mixture_class(
                init_weights,
                init_conds,
            )
            m.fit(data)
            icpool.register_result(unique_id, m, -m.bic(data))

        # loop will end when icg stops generating reasonable initial conditions
        return icpool.best_mixture

    def read_config_file(
        self,
        config_file: Union[str, Path]
    ) -> dict[Any, Any]:
        """Read the contents of the config file into a
        dictionary

        Parameters
        ----------
        config_file : Union[str, Path]
            A yaml configuration file with sections mixture, icpool,
            introducer and component.

        Returns
        -------
        dict[Any, Any]
            A dictionary of all configuration parameters

        Raises
        ------
        exc
            _description_
        """
        with open(config_file, "r") as stream:
            try:
                config_params = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise exc

        assert isinstance(config_params, dict)
        return config_params
