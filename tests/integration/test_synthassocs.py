import numpy as np
import os
from pathlib import Path

from tests.integration import synthdata
from src.chronostar.driver.driver import Driver
from src.chronostar.component.spacetimecomponent import SpaceTimeComponent
from src.chronostar.icpool.simpleicpool import SimpleICPool
from src.chronostar.introducer.simpleintroducer import SimpleIntroducer
from src.chronostar.mixture.componentmixture import ComponentMixture


def test_twoassocs():
    age1, age2 = 30., 50.
    nstars1, nstars2 = 1_000, 1_000
    stars = synthdata.generate_two_overlapping(age1, age2, nstars1, nstars2)

    curr_dir = Path(os.path.dirname(__file__))
    config_file = curr_dir / 'test_resources' / 'placeholder_configfile.yml'
    driver = Driver(
        config_file=config_file,
        mixture_class=ComponentMixture,
        icpool_class=SimpleICPool,
        introducer_class=SimpleIntroducer,
        component_class=SpaceTimeComponent,
    )

    best_mixture = driver.run(data=stars)
    params = best_mixture.get_params()
    weights = params[0]
    components: list[SpaceTimeComponent] = params[1]

    nstars = len(stars)
    fitted_ages = [c.get_parameters()[2] for c in components]

    try:
        assert np.allclose((nstars1, nstars2), weights*nstars, rtol=0.01)
    except AssertionError:
        assert np.allclose((nstars2, nstars1), weights*nstars, rtol=0.01)

    try:
        assert np.allclose([age1, age2], fitted_ages, atol=1)
    except AssertionError:
        assert np.allclose([age2, age1], fitted_ages, atol=1)

    return best_mixture, stars, (age1, age2), (nstars1, nstars2)


def test_one_assoc_one_gaussian_background():
    DIM = 6
    bg_mean = np.zeros(DIM)
    bg_stdev_pos = 1000.
    bg_stdev_vel = 30.
    bg_cov = np.eye(DIM)
    bg_cov[:3] *= bg_stdev_pos
    bg_cov[3:] *= bg_stdev_vel

    bg_age = 0.
    bg_nstars = 10_000

    bg_stars = synthdata.generate_association(
        bg_mean, bg_cov, bg_age, bg_nstars,
    )

    assoc_mean = np.ones(DIM)
    assoc_stdev_pos = 50.
    assoc_stdev_vel = 1.5
    assoc_age = 30.
    assoc_nstars = 500

    assoc_cov = np.eye(DIM)
    assoc_cov[:3] *= assoc_stdev_pos**2
    assoc_cov[3:] *= assoc_stdev_vel**2

    assoc_stars = synthdata.generate_association(
        assoc_mean, assoc_cov, assoc_age, assoc_nstars,
    )

    stars = np.vstack((assoc_stars, bg_stars))

    curr_dir = Path(os.path.dirname(__file__))
    config_file = curr_dir / 'test_resources' / 'placeholder_configfile.yml'
    driver = Driver(
        config_file=config_file,
        mixture_class=ComponentMixture,
        icpool_class=SimpleICPool,
        introducer_class=SimpleIntroducer,
        component_class=SpaceTimeComponent,
    )

    best_mixture = driver.run(data=stars)

    params = best_mixture.get_params()
    weights = params[0]
    components: list[SpaceTimeComponent] = params[1]

    nstars = len(stars)
    fitted_ages = [c.get_parameters()[2] for c in components]
    assoc_ix = np.argmax(fitted_ages)
    fitted_mean, fitted_cov, fitted_age = components[assoc_ix].get_parameters()
    fitted_assoc_weight = weights[assoc_ix]

    nstars = len(stars)
    assert np.isclose(assoc_nstars, fitted_assoc_weight*nstars, rtol=0.1)
    assert np.isclose(assoc_age, fitted_age, rtol=0.1)

    return best_mixture, stars


def test_one_assoc_one_uniform_background():
    DIM = 6
    # background_centre = np.ones(DIM)
    background_spread_pos = 1000.
    background_spread_vel = 30.
    background_nstars = 10_000

    np.random.seed(0)
    bg_stars = np.random.uniform(
        low=-0.5,
        high=0.5,
        size=(background_nstars, DIM)
    )
    bg_stars[:, :3] *= background_spread_pos
    bg_stars[:, 3:] *= background_spread_vel

    assoc_mean = np.ones(DIM)
    assoc_stdev_pos = 50.
    assoc_stdev_vel = 1.5
    assoc_age = 30.
    assoc_nstars = 100

    assoc_cov = np.eye(DIM)
    assoc_cov[:3] *= assoc_stdev_pos**2
    assoc_cov[3:] *= assoc_stdev_vel**2

    assoc_stars = synthdata.generate_association(
        assoc_mean,
        assoc_cov,
        assoc_age,
        assoc_nstars,
    )

    stars = np.vstack((assoc_stars, bg_stars))

    curr_dir = Path(os.path.dirname(__file__))
    config_file = curr_dir / 'test_resources' / 'placeholder_configfile.yml'
    driver = Driver(
        config_file=config_file,
        mixture_class=ComponentMixture,
        icpool_class=SimpleICPool,
        introducer_class=SimpleIntroducer,
        component_class=SpaceTimeComponent,
    )

    best_mixture = driver.run(data=stars)

    params = best_mixture.get_params()
    weights = params[0]
    components: list[SpaceTimeComponent] = params[1]

    nstars = len(stars)
    fitted_ages = [c.get_parameters()[2] for c in components]
    assoc_ix = np.argmax(fitted_ages)
    fitted_mean, fitted_cov, fitted_age = components[assoc_ix].get_parameters()
    fitted_assoc_weight = weights[assoc_ix]

    nstars = len(stars)
    assert np.isclose(assoc_nstars, fitted_assoc_weight*nstars, rtol=0.1)
    assert np.isclose(assoc_age, fitted_age, atol=1.0)
    return best_mixture, stars


if __name__ == '__main__':
    print("Fitting to the Gaussian one")
    best_mixture, stars = test_one_assoc_one_gaussian_background()
