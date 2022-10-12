import numpy as np
import os
from pathlib import Path

from ..context import chronostar     # noqa

from chronostar.driver import Driver
from chronostar.component.spherespacetimecomponent import\
    SphereSpaceTimeComponent
from chronostar.icpool.simpleicpool import SimpleICPool
from chronostar.introducer.simpleintroducer import SimpleIntroducer
from chronostar.mixture.componentmixture import ComponentMixture
from chronostar import synthdata


def test_twoassocs():
    age1, age2 = 5., 10.
    nstars1, nstars2 = 1_000, 2_000
    stars = synthdata.generate_two_overlapping(age1, age2, nstars1, nstars2)

    true_membership_probs = np.zeros((nstars1 + nstars2, 2))
    true_membership_probs[:nstars1, 0] = 1.
    true_membership_probs[nstars1:, 1] = 1.

    curr_dir = Path(os.path.dirname(__file__))
    config_file = curr_dir / 'test_resources' / 'placeholder_configfile.yml'
    driver = Driver(
        config_file=config_file,
        mixture_class=ComponentMixture,
        icpool_class=SimpleICPool,
        introducer_class=SimpleIntroducer,
        component_class=SphereSpaceTimeComponent,
    )

    best_mixture = driver.run(data=stars)
    params = best_mixture.get_parameters()
    weights = params[0]
    components: list[SphereSpaceTimeComponent] = params[1]

    nstars = len(stars)
    fitted_ages = [c.get_parameters()[2] for c in components]
    fitted_memberships = best_mixture.estimate_membership_prob(X=stars)

    # First, assume component 1 mapped to association 1
    try:
        assert np.allclose((nstars1, nstars2), weights*nstars, rtol=0.05)
        assert np.allclose([age1, age2], fitted_ages, rtol=0.1)
        false_matches = np.sum(
            (true_membership_probs != np.round(fitted_memberships))[:, 0]
        )
        assert false_matches < 0.05 * (nstars)

    # If failed, assume component 1 mapped to association 2
    except AssertionError:
        assert np.allclose((nstars2, nstars1), weights*nstars, rtol=0.05)
        assert np.allclose([age2, age1], fitted_ages, rtol=0.1)
        false_matches = np.sum(
            (
                true_membership_probs[:, :: -1]
                != np.round(fitted_memberships)
             )[:, 0]
        )
        assert false_matches < 0.05 * (nstars)

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

    seed = 0
    rng = np.random.default_rng(seed)

    bg_stars = synthdata.generate_association(
        bg_mean, bg_cov, bg_age, bg_nstars, rng=rng,
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
        assoc_mean, assoc_cov, assoc_age, assoc_nstars, rng=rng,
    )

    stars = np.vstack((assoc_stars, bg_stars))

    curr_dir = Path(os.path.dirname(__file__))
    config_file = curr_dir / 'test_resources' / 'placeholder_configfile.yml'
    driver = Driver(
        config_file=config_file,
        mixture_class=ComponentMixture,
        icpool_class=SimpleICPool,
        introducer_class=SimpleIntroducer,
        component_class=SphereSpaceTimeComponent,
    )

    best_mixture = driver.run(data=stars)

    params = best_mixture.get_parameters()
    weights = params[0]
    components: list[SphereSpaceTimeComponent] = params[1]

    nstars = len(stars)
    fitted_ages = [c.get_parameters()[2] for c in components]
    assoc_ix = np.argmax(fitted_ages)
    fitted_mean, fitted_cov, fitted_age = components[assoc_ix].get_parameters()
    fitted_assoc_weight = weights[assoc_ix]

    nstars = len(stars)
    assert np.isclose(assoc_nstars, fitted_assoc_weight*nstars, rtol=0.1)
    assert np.isclose(assoc_age, fitted_age, rtol=0.1)

    return best_mixture, stars, driver


def test_one_assoc_one_uniform_background():
    DIM = 6
    # background_centre = np.ones(DIM)
    background_spread_pos = 1000.
    background_spread_vel = 30.
    background_nstars = 1_000

    seed = 2
    rng = np.random.default_rng(seed)
    print(f"{seed=}")
    bg_stars = rng.uniform(
        low=-0.5,
        high=0.5,
        size=(background_nstars, DIM)
    )
    bg_stars[:, :3] *= background_spread_pos
    bg_stars[:, 3:] *= background_spread_vel

    # Setting the association right in the centre leads to only one component
    percent_offset = 0.15
    assoc_mean = np.zeros(DIM)
    assoc_mean[:3] += percent_offset * background_spread_pos
    assoc_mean[3:] += percent_offset * background_spread_vel
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
        rng=rng,
    )

    stars = np.vstack((assoc_stars, bg_stars))

    curr_dir = Path(os.path.dirname(__file__))
    config_file = curr_dir / 'test_resources' / 'placeholder_configfile.yml'
    driver = Driver(
        config_file=config_file,
        mixture_class=ComponentMixture,
        icpool_class=SimpleICPool,
        introducer_class=SimpleIntroducer,
        component_class=SphereSpaceTimeComponent,
    )

    best_mixture = driver.run(data=stars)

    params = best_mixture.get_parameters()
    weights = params[0]
    components: list[SphereSpaceTimeComponent] = params[1]

    nstars = len(stars)
    fitted_ages = [c.get_parameters()[2] for c in components]
    assoc_ix = np.argmax(fitted_ages)
    fitted_mean, fitted_cov, fitted_age = components[assoc_ix].get_parameters()
    fitted_assoc_weight = weights[assoc_ix]

    nstars = len(stars)
    assert np.isclose(assoc_nstars, fitted_assoc_weight*nstars, rtol=0.1)
    assert np.isclose(assoc_age, fitted_age, rtol=0.1)
    return best_mixture, stars


if __name__ == '__main__':
    print("Fitting to the uniform one")
    # best_mixture, stars, *extra = test_twoassocs()
    res = test_one_assoc_one_gaussian_background()
