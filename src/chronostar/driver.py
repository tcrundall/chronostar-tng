from collections import defaultdict
import inspect
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Callable, Optional, Type, Union
import yaml

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from .base import (
    BaseComponent,
    BaseMixture,
    BaseICPool,
)

from .component.spherespacetimecomponent import SphereSpaceTimeComponent
from .mixture.componentmixture import ComponentMixture
from .icpool.simpleicpool import SimpleICPool


def heading_str(heading: str, sym: str = '-', top: bool = True) -> str:
    line = sym * len(heading)
    if top:
        return f'\n{line}\n{heading}\n{line}\n'
    else:
        return f'\n{heading}\n{line}\n'


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
    savedir: str = './result/intermediate'

    def __init__(
        self,
        config_file: Union[dict[str, Any], str, Path],
        mixture_class: Type[BaseMixture] = ComponentMixture,
        icpool_class: Type[BaseICPool] = SimpleICPool,
        component_class: Type[BaseComponent] = SphereSpaceTimeComponent,
    ) -> None:
        if isinstance(config_file, dict):
            self.config_params = config_file
        else:
            self.config_params = self.read_config_file(config_file)

        self.component_class = component_class
        self.mixture_class = mixture_class
        self.icpool_class = icpool_class

        self.configure(**self.config_params["driver"])

        self.component_class.configure(**self.config_params["component"])
        self.mixture_class.configure(**self.config_params["mixture"])
        self.icpool_class.configure(**self.config_params["icpool"])

    def configure(self, **kwargs) -> None:              # type: ignore
        """Set any configurable class attributes
        """
        function_parser: dict[str, Callable] = {}       # type: ignore

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
        start_init_comps: Optional[tuple[BaseComponent, ...]] = None,
        init_resp: Optional[NDArray[float64]] = None,
    ) -> BaseMixture:
        """Run a fit on the input data

        Parameters
        ----------
        data : NDArray[float64] of shape (n_samples, n_features)
            The input data
        start_init_comps : InitialCondition, optional
            Parameters defining a mixture model, which serves as a
            starting point for the entire run.
        init_resp : NDArray[float64] of shape (n_samples, n_comps), optional
            Starting point for component memberships

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

        # Initial conditions for all mixture model fits come from the ICPool,
        # therefore, if we want a specific starting point from input, it must
        # be fed to the ICPool

        # Handle initial conditions defined by membership probability
        if init_resp is not None:
            assert start_init_comps is None
            assert getattr(self.mixture_class, 'init_params') == 'init_resp'
            n_comps = init_resp.shape[-1]
            start_init_comps = tuple(self.component_class() for _ in range(n_comps))

        icpool = self.icpool_class(
            component_class=self.component_class,
            start_init_comps=start_init_comps,
        )

        # icpool maintains an internal queue of sets of initial conditions
        # iterate through, fitting to each set, and registering the result
        while icpool.has_next():
            label, init_comps = icpool.get_next()
            print(f"[DRIVER] Fitting {label}")
            print(f"[DRIVER] {init_comps[0].parameters_set=}")

            # If init_resp was provided, we use it to initialise the *very first* mixture
            if init_resp is not None:
                init_weights = np.copy(init_resp)
                init_resp = None
            else:
                ncomps = len(init_comps)
                init_weights = np.ones(ncomps)/ncomps

            m = self.mixture_class(init_weights, init_comps)
            m.fit(data)
            icpool.register_result(label, m, -m.bic(data))

            if self.intermediate_dumps:
                dump_dir = self.savedir_path / label
                self.dump_mixture_result(dump_dir, label, m, data)

        # loop will end when icpool stops generating initial conditions
        return icpool.best_mixture

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
        config_params: dict[str, dict[str, Any]] = defaultdict(dict)
        with open(config_file, "r") as stream:
            try:
                config_params.update(yaml.safe_load(stream))
            except yaml.YAMLError as exc:
                raise exc

        # Fail loudly if config file improperly structured
        acceptable_keys = [
            'driver', 'icpool', 'component', 'mixture', 'run',
        ]
        for key in config_params:
            if key == 'run':
                print(f"[CONFIG] {key=} is ignored by driver")
            if key not in acceptable_keys:
                raise UserWarning(f"[CONFIG] {key} not recognised!")

        return config_params

    def dump_mixture_result(
        self,
        dump_dir: Path,
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
        dump_dir.mkdir(parents=True, exist_ok=True)

        # Save model parameters in numpy arrays
        weights, components = mixture.get_parameters()
        memberships = mixture.estimate_membership_prob(data)
        np.save(str(dump_dir / 'weights.npy'), weights)
        np.save(str(dump_dir / 'memberships.npy'), memberships)
        for i, comp in enumerate(components):
            np.save(str(dump_dir / f"comp_{i:03}_params.npy"), comp.get_parameters())

        # Write information to file for quick overview
        results_file = dump_dir / f'{label}.txt'
        with open(results_file, 'w') as fp:
            fp.write(f"Results of {label}\n")
            fp.write("----------------------------------------------\n")
            fp.write(f"BIC: {mixture.bic(data)}\n")
            fp.write(f"weights: {weights}\n")
            fp.write(f"total members: {memberships.sum(axis=0)}\n")
            fp.write(heading_str("Component parameters"))
            for i, comp in enumerate(components):
                fp.write(heading_str(f"Component {i:03}:", sym='^', top=False))
                fp.write(f"{comp.get_parameters()}\n")
                fp.write("\n")

            fp.write(heading_str("Configuration settings"))
            self.dump_all_config_params(fp)

    def dump_all_config_params(self, fp: TextIOWrapper) -> None:
        """Get all configurable parameters and write them to file

        This method attempts to identify all class level parameters that
        could be configurable and writes them all to file.

        Parameters
        ----------
        fp : TextIOWrapper
            File pointer to output file
        """

        def get_simple_attributes(target_class: Any) -> list[tuple[str, Any]]:
            # Get everything in class which isn't a routine
            attributes = inspect.getmembers(
                target_class, lambda x: not inspect.isroutine(x)
            )

            # Filter out built in attributes
            attributes = [
                a for a in attributes
                if not (a[0].startswith('__') and a[0].endswith('__'))
            ]

            # Filter for booleans, string, floats, integers and classes
            attributes = [
                a for a in attributes
                if isinstance(a[1], (bool, str, float, int)) or inspect.isclass(a[1])
            ]

            return attributes

        classes = {
            "driver": self,
            "component": self.component_class,
            "mixture": self.mixture_class,
            "icpool": self.icpool_class,
        }
        for cname, cl in classes.items():
            fp.write(f"{cname}:\n")
            for name, value in get_simple_attributes(cl):
                if isinstance(value, str):
                    fp.write(f'    {name}: "{value}"\n')
                elif isinstance(value, float):
                    fp.write(f'    {name}: {value:.1e}\n')
                else:
                    fp.write(f'    {name}: {value}\n')
