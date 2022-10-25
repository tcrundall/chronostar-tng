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
from .mixture.componentmixture import ComponentMixture
from .introducer.simpleintroducer import SimpleIntroducer
from .icpool.simpleicpool import SimpleICPool


class Driver:
    """Top level class of Chronostar which drives the
    entire fitting process

    Parameters
    ----------
    config_file : Union[str, Path]
        A yaml configuration file with sections mixture, icpool,
        introducer and component.
    mixture_class : Type[BaseMixture], default :class:`ComponentMixture`
        A class derived from BaseMixture
    icpool_class : Type[BaseICPool], default :class:`SimpleICPool`
        A class derived from BaseICPool
    introducer_class : Type[BaseIntroducer], default :class:`SimpleIntroducer`
        A class derived from BaseIntroducer
    component_class : Type[BaseComponent], default :class:`SphereSpaceTimeComponent`
        A class derived from BaseComponent

    Attributes
    ----------
    intermediate_dumps : bool, default True
        Whether to write to file the results of mixture model fits, configurable
    savedir : str, default './result'
        Path to the directory of where to store results, configurable
    """

    intermediate_dumps: bool = True
    savedir: str = './result'

    def __init__(
        self,
        config_file: Union[dict, str, Path],
        mixture_class: Type[BaseMixture] = ComponentMixture,
        icpool_class: Type[BaseICPool] = SimpleICPool,
        introducer_class: Type[BaseIntroducer] = SimpleIntroducer,
        component_class: Type[BaseComponent] = SphereSpaceTimeComponent,
    ) -> None:
        if isinstance(config_file, dict):
            self.config_params = config_file
        else:
            self.config_params = self.read_config_file(config_file)

        self.component_class = component_class
        self.mixture_class = mixture_class
        self.icpool_class = icpool_class
        self.introducer_class = introducer_class

        self.configure(**self.config_params["driver"])

        self.component_class.configure(**self.config_params["component"])
        self.mixture_class.configure(**self.config_params["mixture"])
        self.icpool_class.configure(**self.config_params["icpool"])
        self.introducer_class.configure(**self.config_params["introducer"])

    def configure(self, **kwargs) -> None:
        """Set any configurable class attributes
        """
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

    def run(
        self,
        data: NDArray[float64],
        first_init_conds=None
    ) -> BaseMixture:
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

        # Before we start, make sure results directory exists
        if self.intermediate_dumps:
            self.savedir_path = Path(self.savedir)
            self.savedir_path.mkdir(parents=True, exist_ok=True)

        # # If we have covariances, merge with data array
        # if covariances is not None:
        #     data = np.vstack((data.T, covariances.reshape(-1, 36).T)).T

        icpool = self.icpool_class(
            introducer_class=self.introducer_class,
            component_class=self.component_class,
        )

        if first_init_conds is not None:
            print(f"[DRIVER] {first_init_conds[0].parameters_set=}")
            icpool.provide_start(first_init_conds)

        # icpool maintains an internal queue of sets of initial conditions
        # iterate through, fitting to each set, and registering the result
        while icpool.has_next():
            label, init_comps = icpool.get_next()
            print(f"[DRIVER] Fitting {label}")
            print(f"[DRIVER] {init_comps[0].parameters_set=}")

            ncomps = len(init_comps)
            init_weights = np.ones(ncomps)/ncomps
            m = self.mixture_class(
                init_weights,
                init_comps,
            )
            m.fit(data)
            icpool.register_result(label, m, -m.bic(data))

            if self.intermediate_dumps:
                self.dump_mixture_result(label, m, data)

        # loop will end when icpool stops generating initial conditions
        return icpool.best_mixture

    def dump_mixture_result(
        self,
        label: str,
        mixture: BaseMixture,
        data: NDArray[float64],
    ) -> None:
        """Store the result of a mixture to file

        Parameters
        ----------
        label : str
            The unique label of the initial condition of the mixture
        mixture : BaseMixture
            The (fitted) mixture model
        data : NDArray of shape (n_samples, n_features)
            The input data
        """
        # Make directory
        mixture_dir = self.savedir_path / label
        mixture_dir.mkdir(parents=True, exist_ok=True)

        # Save model parameters in numpy arrays
        weights, components = mixture.get_parameters()
        memberships = mixture.estimate_membership_prob(data)
        np.save(str(mixture_dir / 'weights.npy'), weights)
        np.save(str(mixture_dir / 'memberships.npy'), memberships)
        for i, comp in enumerate(components):
            np.save(str(mixture_dir / f"comp_{i:03}_params.npy"), comp.get_parameters())

        # Write information to file for quick overview
        results_file = mixture_dir / f'{label}.txt'
        with open(results_file, 'w') as fp:
            fp.write(f"Results of {label}\n")
            fp.write("----------------------------------------------\n")
            fp.write(f"BIC: {mixture.bic(data)}\n")
            fp.write(f"weights: {weights}\n")
            fp.write(f"total members: {memberships.sum(axis=0)}\n")
            fp.write("\n")
            fp.write("--------------------\n")
            fp.write("Component parameters\n")
            fp.write("--------------------\n")
            for i, comp in enumerate(components):
                fp.write(f"Component {i:03}:\n")
                fp.write("^^^^^^^^^^^^^^^^^\n")
                fp.write(f"{comp.get_parameters()}\n")
                fp.write("\n")

            fp.write("\n")
            fp.write("----------------------\n")
            fp.write("Configuration settings\n")
            fp.write("----------------------\n")
            yaml.dump(self.config_params, fp)

            fp.write("\n")
            fp.write("------------\n")
            fp.write("Classes used\n")
            fp.write("------------\n")
            fp.write(f"{self.component_class=}\n")
            fp.write(f"{self.mixture_class=}\n")
            fp.write(f"{self.introducer_class=}\n")
            fp.write(f"{self.icpool_class=}\n")

    def read_config_file(
        self,
        config_file: Union[str, Path]
    ) -> dict[str, Any]:
        """Read the contents of the config file into a
        dictionary

        Parameters
        ----------
        config_file : Union[str, Path]
            A yaml configuration file with sections mixture, icpool,
            introducer and component.

        Returns
        -------
        dict[str, Any]
            A dictionary of all configuration parameters

        Raises
        ------
        yaml.YAMLError 
            A yaml exception, in the event the file can't be read
        UserWarning
            If the file has an unrecognised key at top-most level
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
            'driver', 'icpool', 'component', 'introducer', 'mixture', 'run',
        ]
        for key in config_params:
            if key == 'run':
                print(f"[CONFIG] {key=} is ignored by driver")
            if key not in acceptable_keys:
                raise UserWarning(f"[CONFIG] {key} not recognised!")

        return config_params
