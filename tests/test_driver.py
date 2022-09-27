# assume chronostar is on path?
import numpy as np
import os
from pathlib import Path
try:
    from .context import chronostar as c
except ImportError:
    from context import chronostar as c

# Import a bunch of placeholder classes that simulate required behaviour
from fooclasses import FooComponent, FooICPool, FooIntroducer, FooMixture


def test_construction() -> None:
    test_dir = Path(os.path.dirname(__file__))
    FooConfigfile = test_dir / 'test_resources' / 'placeholder_configfile.yml'
    driver = c.driver.driver.Driver(
        config_file=FooConfigfile,
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
    n_samples, n_features = 100, 6
    data = np.random.rand(n_samples, n_features)

    driver = c.driver.driver.Driver(
        config_file=foo_configfile,
        mixture_class=FooMixture,
        icpool_class=FooICPool,
        introducer_class=FooIntroducer,
        component_class=FooComponent,
        )

    best_mixture, memberships = driver.run(data)

    assert isinstance(best_mixture, FooMixture)


if __name__ == '__main__':
    test_run()