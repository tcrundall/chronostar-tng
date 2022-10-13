from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Type, Union
import yaml

import numpy as np
from numpy import float64
from numpy.typing import NDArray


from .base import (
    BaseComponent,
    BaseMixture,
    BaseICPool,
    BaseIntroducer,
)

from .component.spherespacetimecomponent import SphereSpaceTimeComponent
# from .component.ellipspacetimecomponent import EllipSpaceTimeComponent
from .mixture.componentmixture import ComponentMixture
from .introducer.simpleintroducer import SimpleIntroducer
from .icpool.simpleicpool import SimpleICPool
from .datatools import replace_cov_with_sampling


class Driver:

    # Configurable attrbutes
    approx_covariances = False
    n_draws = 25

    def __init__(
        self,
        config_file: Union[dict, str, Path],
        mixture_class: Type[BaseMixture] = ComponentMixture,
        icpool_class: Type[BaseICPool] = SimpleICPool,
        introducer_class: Type[BaseIntroducer] = SimpleIntroducer,
        component_class: Type[BaseComponent] = SphereSpaceTimeComponent,
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

        if isinstance(config_file, dict):
            config_params = config_file
        else:
            config_params = self.read_config_file(config_file)

        self.component_class = component_class
        self.mixture_class = mixture_class
        self.icpool_class = icpool_class
        self.introducer_class = introducer_class

        self.configure(**config_params["driver"])

        self.component_class.configure(**config_params["component"])
        self.mixture_class.configure(**config_params["mixture"])
        self.icpool_class.configure(**config_params["icpool"])
        self.introducer_class.configure(**config_params["introducer"])

    def configure(self, **kwargs) -> None:
        function_parser: dict[str, Callable] = {}

        for param, val in kwargs.items():
            if hasattr(self, param):
                if val in function_parser:
                    setattr(self, param, function_parser[val])
                else:
                    setattr(self, param, val)
            else:
                print(
                    f"[CONFIG]:{self} unexpected config param: {param}={val}"
                )

    def run(self, data: NDArray[float64], covariances=None) -> BaseMixture:
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

        if self.approx_covariances:
            data = replace_cov_with_sampling(
                data,
                covariances,
                n_draws=self.n_draws,
            )
            # If membership used to initialise:
            # memb_probs = memb_probs.repeat(
            #       self.config_params["dirver"]["n_draws"], axis=0
            #       )

        icpool = self.icpool_class(
            introducer_class=self.introducer_class,
            component_class=self.component_class,
        )

        while icpool.has_next():
            unique_id, init_conds = icpool.get_next()

            ncomps = len(init_conds)
            init_weights = np.ones(ncomps)/ncomps
            m = self.mixture_class(
                init_weights,
                init_conds,
            )
            m.fit(data)
            icpool.register_result(unique_id, m, -m.bic(data))

        # If covariances were approximated, make sure to average over
        # any returned membership probabilities

        # loop will end when icpool stops generating initial conditions
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
        # We use a default dict to gracefully handle empty config file
        config_params: dict[str, dict] = defaultdict(dict)
        with open(config_file, "r") as stream:
            try:
                config_params.update(yaml.safe_load(stream))
            except yaml.YAMLError as exc:
                raise exc

        # Fail loudly if config file improperly structured
        acceptable_keys = [
            'driver', 'icpool', 'component', 'introducer', 'mixture',
        ]
        for key in config_params:
            if key not in acceptable_keys:
                raise UserWarning(f"[CONFIG] {key} not recognised!")

        return config_params
