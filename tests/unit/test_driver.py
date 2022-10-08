import os
from pathlib import Path

from ..context import chronostar     # noqa

from chronostar.driver.driver import Driver
from .fooclasses import FooComponent, FooICPool, FooIntroducer, FooMixture
from .fooclasses import DATA


def test_construction() -> None:
    test_dir = Path(os.path.dirname(__file__))
    foo_configfile = test_dir / 'test_resources' / 'placeholder_configfile.yml'
    driver = Driver(
        config_file=foo_configfile,
        mixture_class=FooMixture,
        icpool_class=FooICPool,
        introducer_class=FooIntroducer,
        component_class=FooComponent,
        )

    assert isinstance(driver.config_params, dict)


def test_run() -> None:
    # Set up fake config file and data
    test_dir = Path(os.path.dirname(__file__))
    foo_configfile = test_dir / 'test_resources' / 'placeholder_configfile.yml'

    driver = Driver(
        config_file=foo_configfile,
        mixture_class=FooMixture,
        icpool_class=FooICPool,
        introducer_class=FooIntroducer,
        component_class=FooComponent,
        )

    best_mixture = driver.run(DATA)

    assert isinstance(best_mixture, FooMixture)


if __name__ == '__main__':
    test_run()
