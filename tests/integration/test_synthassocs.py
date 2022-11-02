import numpy as np
import os
from pathlib import Path

from ..context import chronostar     # noqa

from chronostar import synthdata, datatools
from chronostar.utils import coordinate, transform
from chronostar.driver import Driver
from chronostar.traceorbit import trace_epicyclic_orbit
from chronostar.component.spherespacetimecomponent import\
    SphereSpaceTimeComponent
from chronostar.icpool.simpleicpool import SimpleICPool
from chronostar.mixture.componentmixture import ComponentMixture


def test_twoassocs():
    """
    Generate two synthetic associations, then fit to them.
    """
    age_1, age_2 = 5., 10.
    n_stars_1, n_stars_2 = 1_000, 2_000

    mean_now_1 = np.zeros(6)
    mean_now_2 = np.array([5., 10., 20., 1., 2., 3], dtype=float)

    mean_birth_1 = trace_epicyclic_orbit(
        mean_now_1[np.newaxis],
        -age_1,
    ).squeeze()

    mean_birth_2 = trace_epicyclic_orbit(
        mean_now_2[np.newaxis],
        -age_2,
    ).squeeze()

    birth_dxyz = 10.
    birth_duvw = 3

    true_comp_1 = SphereSpaceTimeComponent(
        np.hstack((
            mean_birth_1,
            birth_dxyz,
            birth_duvw,
            age_1,
        ))
    )

    true_comp_2 = SphereSpaceTimeComponent(
        np.hstack((
            mean_birth_2,
            birth_dxyz,
            birth_duvw,
            age_2,
        ))
    )

    seed = 0
    rng = np.random.default_rng(seed)
    stars_1 = rng.multivariate_normal(
        true_comp_1.mean, true_comp_1.covariance, size=n_stars_1
    )
    stars_2 = rng.multivariate_normal(
        true_comp_2.mean, true_comp_2.covariance, size=n_stars_2
    )
    stars = np.vstack((stars_1, stars_2))

    true_membership_probs = np.zeros((n_stars_1 + n_stars_2, 2))
    true_membership_probs[:n_stars_1, 0] = 1.
    true_membership_probs[n_stars_1:, 1] = 1.

    curr_dir = Path(os.path.dirname(__file__))
    config_file = curr_dir / 'test_resources' / 'integration_configfile.yml'
    driver = Driver(
        config_file=config_file,
        mixture_class=ComponentMixture,
        icpool_class=SimpleICPool,
        component_class=SphereSpaceTimeComponent,
    )

    best_mixture = driver.run(data=stars)
    params = best_mixture.get_parameters()
    weights = params[0]
    components = params[1]

    nstars = len(stars)
    fitted_ages = [c.get_parameters()[-1] for c in components]
    fitted_memberships = best_mixture.estimate_membership_prob(X=stars)

    # First, assume component 1 mapped to association 1
    try:
        assert np.allclose((n_stars_1, n_stars_2), weights*nstars, rtol=0.05)
        assert np.allclose([age_1, age_2], fitted_ages, rtol=0.1)
        false_matches = np.sum(
            (true_membership_probs != np.round(fitted_memberships))[:, 0]
        )
        assert false_matches < 0.1 * (nstars)

    # If failed, assume component 1 mapped to association 2
    except AssertionError:
        assert np.allclose((n_stars_2, n_stars_1), weights*nstars, rtol=0.05)
        assert np.allclose([age_2, age_1], fitted_ages, rtol=0.1)
        false_matches = np.sum(
            (
                true_membership_probs[:, :: -1]
                != np.round(fitted_memberships)
            )[:, 0]
        )
        assert false_matches < 0.1 * (nstars)

    return best_mixture, stars, (age_1, age_2), (n_stars_1, n_stars_2)


def test_one_assoc_one_gaussian_background():
    DIM = 6
    bg_mean = np.zeros(DIM)
    bg_stdev_pos = 1000.
    bg_stdev_vel = 30.
    bg_cov = np.eye(DIM)
    bg_cov[:3] *= bg_stdev_pos**2
    bg_cov[3:] *= bg_stdev_vel**2

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
    config_file = curr_dir / 'test_resources' / 'integration_configfile.yml'
    driver = Driver(
        config_file=config_file,
        mixture_class=ComponentMixture,
        icpool_class=SimpleICPool,
        component_class=SphereSpaceTimeComponent,
    )

    best_mixture = driver.run(data=stars)

    params = best_mixture.get_parameters()
    weights = params[0]
    components = params[1]

    nstars = len(stars)
    fitted_ages = [c.get_parameters()[-1] for c in components]
    assoc_ix = np.argmax(fitted_ages)
    assoc_comp = components[assoc_ix]
    # parameters = components[assoc_ix].get_parameters()
    fitted_assoc_weight = weights[assoc_ix]

    nstars = len(stars)
    assert np.isclose(assoc_nstars, fitted_assoc_weight*nstars, rtol=0.1)
    assert np.isclose(assoc_age, assoc_comp.age, rtol=0.1)      # type: ignore

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
    config_file = curr_dir / 'test_resources' / 'integration_configfile.yml'
    driver = Driver(
        config_file=config_file,
        mixture_class=ComponentMixture,
        icpool_class=SimpleICPool,
        component_class=SphereSpaceTimeComponent,
    )

    best_mixture = driver.run(data=stars)

    params = best_mixture.get_parameters()
    weights = params[0]
    components = params[1]

    nstars = len(stars)
    fitted_ages = [c.get_parameters()[-1] for c in components]
    assoc_ix = np.argmax(fitted_ages)
    fitted_comp = components[assoc_ix]
    fitted_assoc_weight = weights[assoc_ix]

    nstars = len(stars)
    assert np.isclose(assoc_nstars, fitted_assoc_weight*nstars, rtol=0.1)
    assert np.isclose(assoc_age, fitted_comp.age, rtol=0.1)     # type: ignore
    return best_mixture, stars


# @pytest.mark.skip("Takes around 1 hour to run")
def test_uncertain_one_assoc_one_gaussian_background():
    DIM = 6
    bg_mean = np.zeros(DIM)
    bg_stdev_pos = 1000.
    bg_stdev_vel = 30.
    bg_cov = np.eye(DIM)
    bg_cov[:3] *= bg_stdev_pos**2
    bg_cov[3:] *= bg_stdev_vel**2

    bg_age = 0.
    bg_nstars = 1_000

    seed = 0
    rng = np.random.default_rng(seed)

    bg_stars = synthdata.generate_association(
        bg_mean, bg_cov, bg_age, bg_nstars, rng=rng,
    )

    assoc_mean = np.ones(DIM)
    assoc_stdev_pos = 50.
    assoc_stdev_vel = 1.5
    assoc_age = 30.
    assoc_nstars = 200

    assoc_cov = np.eye(DIM)
    assoc_cov[:3] *= assoc_stdev_pos**2
    assoc_cov[3:] *= assoc_stdev_vel**2

    assoc_stars = synthdata.generate_association(
        assoc_mean, assoc_cov, assoc_age, assoc_nstars, rng=rng,
    )

    stars = np.vstack((assoc_stars, bg_stars))

    synthdata.SynthData.m_err = 0.6
    astrometry = synthdata.SynthData.measure_astrometry(stars)
    astro_data = datatools.extract_array_from_table(astrometry)
    astro_means, astro_covs = datatools.construct_covs_from_data(astro_data)

    cart_means = np.empty(astro_means.shape)
    cart_covs = np.empty(astro_covs.shape)
    for i in range(len(cart_means)):
        cart_covs[i], cart_means[i] = transform.transform_covmatrix_py(
            cov=astro_covs[i],
            trans_func=coordinate.convert_astrometry2lsrxyzuvw,
            loc=astro_means[i],
        )

    data = np.vstack((cart_means.T, cart_covs.reshape(-1, 36).T)).T

    curr_dir = Path(os.path.dirname(__file__))
    config_file = curr_dir / 'test_resources' / 'uncertainties_configfile.yml'
    driver = Driver(
        config_file=config_file,
        mixture_class=ComponentMixture,
        icpool_class=SimpleICPool,
        component_class=SphereSpaceTimeComponent,
    )

    best_mixture = driver.run(data=data)

    params = best_mixture.get_parameters()
    weights = params[0]
    components = params[1]

    nstars = len(stars)
    fitted_ages = [c.get_parameters()[-1] for c in components]
    assoc_ix = np.argmax(fitted_ages)
    assoc_comp = components[assoc_ix]
    # parameters = components[assoc_ix].get_parameters()
    fitted_assoc_weight = weights[assoc_ix]

    nstars = len(stars)
    assert np.isclose(assoc_nstars, fitted_assoc_weight*nstars, rtol=0.1)
    assert np.isclose(assoc_age, assoc_comp.age, rtol=0.1)      # type: ignore

    return best_mixture, stars, driver


if __name__ == '__main__':
    print("Fitting to the uniform one")
    best_mixture, stars, *extra = test_twoassocs()
    # res = test_one_assoc_one_gaussian_background()
