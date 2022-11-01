import os
from pathlib import Path
import numpy as np

from ..context import chronostar     # noqa

from chronostar.driver import Driver
from chronostar.mixture.componentmixture import ComponentMixture
from chronostar.icpool.simpleicpool import SimpleICPool
# from chronostar.component.ellipspacetimecomponent import\
#   EllipSpaceTimeComponent
from chronostar.component.spherespacetimecomponent import\
    SphereSpaceTimeComponent
from chronostar.component.spacecomponent import SpaceComponent


def test_simple_spacemixture_run():
    curr_dir = Path(os.path.dirname(__file__))
    config_file = curr_dir / 'test_resources' / 'integration_configfile.yml'
    driver = Driver(
        config_file=config_file,
        mixture_class=ComponentMixture,
        icpool_class=SimpleICPool,
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
    config_file = curr_dir / 'test_resources' / 'integration_configfile.yml'
    driver = Driver(
        config_file=config_file,
        mixture_class=ComponentMixture,
        icpool_class=SimpleICPool,
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


def test_init_resp():
    curr_dir = Path(os.path.dirname(__file__))
    config_file = curr_dir / 'test_resources' / 'integration_configfile.yml'
    driver = Driver(
        config_file=config_file,
        mixture_class=ComponentMixture,
        icpool_class=SimpleICPool,
        component_class=SpaceComponent,
    )
    driver.mixture_class.configure(init_params='init_resp')

    seed = 0
    rng = np.random.default_rng(seed)

    n_bg_stars = 1_000
    bound = 200
    uniform_data = rng.uniform(low=-bound, high=bound, size=(n_bg_stars, 6))

    n_assoc_stars = 500
    mean1 = np.zeros(6) + 100
    cov1 = np.eye(6) * 10
    gaussian_data = rng.multivariate_normal(mean1, cov1, size=n_assoc_stars)

    resp = np.zeros((n_bg_stars + n_assoc_stars, 2))
    resp[:n_bg_stars, 0] = 1.
    resp[n_bg_stars:, 1] = 1.

    data = np.vstack((uniform_data, gaussian_data))

    best_mixture = driver.run(data, init_resp=resp)
    weights, comps = best_mixture.get_parameters()
    assert len(weights) == 2

    return best_mixture, data


def test_init_comps():
    curr_dir = Path(os.path.dirname(__file__))
    config_file = curr_dir / 'test_resources' / 'integration_configfile.yml'
    driver = Driver(
        config_file=config_file,
        mixture_class=ComponentMixture,
        icpool_class=SimpleICPool,
        component_class=SpaceComponent,
    )
    driver.mixture_class.configure(init_params='init_resp')

    seed = 0
    rng = np.random.default_rng(seed)

    bg_mean = np.zeros(6)
    bg_stdev_pos = 1000.
    bg_stdev_vel = 30.
    bg_cov = np.eye(6)
    bg_cov[:3] *= bg_stdev_pos**2
    bg_cov[3:] *= bg_stdev_vel**2
    n_bg_stars = 1_000
    bg_comp = SpaceComponent(np.hstack((bg_mean, bg_cov.flatten())))
    bg_data = rng.multivariate_normal(bg_mean, bg_cov, size=n_bg_stars)

    n_assoc_stars = 500
    mean1 = np.zeros(6) + 10
    cov1 = np.eye(6) * 10**2
    assoc_data = rng.multivariate_normal(mean1, cov1, size=n_assoc_stars)
    assoc_comp = SpaceComponent(np.hstack((mean1, cov1.flatten())))

    data = np.vstack((bg_data, assoc_data))

    best_mixture = driver.run(data, start_init_comps=(bg_comp, assoc_comp))
    weights, comps = best_mixture.get_parameters()
    assert len(weights) == 2

    return best_mixture, data


if __name__ == '__main__':
    best_mixture, data = test_simple_spacetimemixture_run()
