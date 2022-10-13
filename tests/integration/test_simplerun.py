import os
from pathlib import Path
import numpy as np

from ..context import chronostar     # noqa

from chronostar.driver import Driver
from chronostar.mixture.componentmixture import ComponentMixture
from chronostar.icpool.simpleicpool import SimpleICPool
from chronostar.introducer.simpleintroducer import SimpleIntroducer
# from chronostar.component.ellipspacetimecomponent import\
#   EllipSpaceTimeComponent
from chronostar.component.spherespacetimecomponent import\
    SphereSpaceTimeComponent
from chronostar.component.spacecomponent import SpaceComponent


def test_simple_spacemixture_run():
    curr_dir = Path(os.path.dirname(__file__))
    config_file = curr_dir / 'test_resources' / 'placeholder_configfile.yml'
    driver = Driver(
        config_file=config_file,
        mixture_class=ComponentMixture,
        icpool_class=SimpleICPool,
        introducer_class=SimpleIntroducer,
        component_class=SpaceComponent,
    )

    seed = 0
    rng = np.random.default_rng(seed)

    bound = 200
    uniform_data = rng.uniform(low=-bound, high=bound, size=(1_000, 6))

    mean1 = np.zeros(6) + 100
    cov1 = np.eye(6) * 10
    gaussian_data1 = rng.multivariate_normal(mean1, cov1, size=500)

    mean2 = np.zeros(6) - 50
    cov2 = np.eye(6) * 20
    gaussian_data2 = rng.multivariate_normal(mean2, cov2, size=300)

    data = np.vstack((uniform_data, gaussian_data1, gaussian_data2))

    best_mixture = driver.run(data)         # noqa F841
    weights, comps = best_mixture.get_parameters()
    assert len(weights) == 3
    return best_mixture, data


def test_simple_spacetimemixture_run():
    curr_dir = Path(os.path.dirname(__file__))
    config_file = curr_dir / 'test_resources' / 'placeholder_configfile.yml'
    driver = Driver(
        config_file=config_file,
        mixture_class=ComponentMixture,
        icpool_class=SimpleICPool,
        introducer_class=SimpleIntroducer,
        component_class=SphereSpaceTimeComponent,
    )

    seed = 0
    rng = np.random.default_rng(seed)

    bound = 200
    uniform_data = rng.uniform(low=-bound, high=bound, size=(1_000, 6))

    mean1 = np.zeros(6) + 100
    cov1 = np.eye(6) * 10
    gaussian_data1 = rng.multivariate_normal(mean1, cov1, size=500)

    mean2 = np.zeros(6) - 50
    cov2 = np.eye(6) * 20
    gaussian_data2 = rng.multivariate_normal(mean2, cov2, size=300)

    data = np.vstack((uniform_data, gaussian_data1, gaussian_data2))

    best_mixture = driver.run(data)         # noqa F841
    weights, comps = best_mixture.get_parameters()
    assert len(weights) == 3

    return best_mixture, data


if __name__ == '__main__':
    best_mixture, data = test_simple_spacetimemixture_run()
