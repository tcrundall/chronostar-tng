"""Test various treatments of background
"""

from ..context import chronostar        # noqa

from chronostar import synthdata
from chronostar.mixture.componentmixture import ComponentMixture
from chronostar.component.spacecomponent import SpaceComponent
from chronostar.component.uniformcomponent import UniformComponent
from chronostar.component.spherespacetimecomponent import SphereSpaceTimeComponent
import numpy as np


def test_uniform_space():
    DIM = 6
    # background_centre = np.ones(DIM)
    background_spread_pos = 1000.
    background_spread_vel = 100.
    background_nstars = 1_000

    seed = 0
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
    assoc_stdev_vel = 5.
    assoc_nstars = 1_000

    assoc_cov = np.eye(DIM)
    assoc_cov[:3] *= assoc_stdev_pos**2
    assoc_cov[3:] *= assoc_stdev_vel**2

    assoc_stars = synthdata.generate_association(
        mean_now=assoc_mean,
        covariance_birth=assoc_cov,
        age=0.,
        nstars=assoc_nstars,
        rng=rng,
    )

    stars = np.vstack((assoc_stars, bg_stars))

    UniformComponent.configure()
    SpaceComponent.configure()

    upper_bound = np.max(assoc_stars, axis=0)
    lower_bound = np.min(assoc_stars, axis=0)

    # Calculate the average density of the occupied volume with a single star
    unit_density = 1./np.prod(upper_bound - lower_bound)

    comps = (UniformComponent(params=np.array([unit_density])), SpaceComponent())

    ComponentMixture.configure()
    init_weights = np.array([0.5, 0.5])
    mixture = ComponentMixture(init_weights=init_weights, init_components=comps)
    mixture.fit(stars)

    params = mixture.get_parameters()
    weights = params[0]
    components = params[1]

    fitted_comp: SpaceComponent = components[1]     # type: ignore
    fitted_assoc_weight = weights[1]

    nstars = len(stars)
    assert np.isclose(assoc_nstars, fitted_assoc_weight*nstars, rtol=0.3)
    assert np.allclose(assoc_mean, fitted_comp.mean, rtol=0.1)
    assert np.isclose(assoc_stdev_pos, np.sqrt(fitted_comp.covariance[0, 0]), rtol=0.2)
    assert np.isclose(assoc_stdev_vel, np.sqrt(fitted_comp.covariance[3, 3]), rtol=0.2)
    return mixture, stars


def test_uniform_sphere():
    DIM = 6
    # background_centre = np.ones(DIM)
    background_spread_pos = 200.
    background_spread_zpos = 100.
    background_spread_vel = 10.
    background_nstars = 850

    seed = 0
    rng = np.random.default_rng(seed)
    print(f"{seed=}")
    bg_stars = rng.uniform(
        low=-0.5,
        high=0.5,
        size=(background_nstars, DIM)
    )
    bg_stars[:, :2] *= background_spread_pos
    bg_stars[:, 2] *= background_spread_zpos
    bg_stars[:, 3:] *= background_spread_vel

    # Setting the association right in the centre leads to only one component
    # percent_offset = 0.15
    assoc_mean = np.zeros(DIM)
    # assoc_mean[:3] += percent_offset * background_spread_pos
    # assoc_mean[3:] += percent_offset * background_spread_vel
    assoc_stdev_pos = 10.
    assoc_stdev_vel = 1.
    assoc_nstars = 150

    assoc_cov = np.eye(DIM)
    assoc_cov[:3] *= assoc_stdev_pos**2
    assoc_cov[3:] *= assoc_stdev_vel**2

    assoc_age = 20.

    assoc_stars = synthdata.generate_association(
        mean_now=assoc_mean,
        covariance_birth=assoc_cov,
        age=assoc_age,
        nstars=assoc_nstars,
        rng=rng,
    )

    stars = np.vstack((assoc_stars, bg_stars))

    UniformComponent.configure()
    SphereSpaceTimeComponent.configure(
        stellar_uncertainties=False,
        max_age=30.,
    )

    upper_bound = np.max(assoc_stars, axis=0)
    lower_bound = np.min(assoc_stars, axis=0)

    # Calculate the average density of the occupied volume with a single star
    unit_density = 1./np.prod(upper_bound - lower_bound)

    comps = (
        UniformComponent(params=np.array([unit_density])),
        SphereSpaceTimeComponent(),
    )

    ComponentMixture.configure(
        verbose=2,
        verbose_interval=1,
        tol=1.e-5,
    )
    init_weights = np.array([1.e-10, 1 - 1.e-10])
    mixture = ComponentMixture(init_weights=init_weights, init_components=comps)
    mixture.fit(stars)

    params = mixture.get_parameters()
    weights = params[0]
    components = params[1]

    fitted_comp: SphereSpaceTimeComponent = components[1]     # type: ignore
    fitted_assoc_weight = weights[1]

    nstars = len(stars)

    # Note: Very poor at membership recovery
    assert np.isclose(assoc_nstars, fitted_assoc_weight*nstars, rtol=1.)
    assert np.allclose(assoc_mean, fitted_comp.mean, rtol=0.1, atol=3.)
    assert np.isclose(assoc_stdev_pos, fitted_comp.parameters[6], rtol=0.3)
    assert np.isclose(assoc_stdev_vel, fitted_comp.parameters[7], rtol=0.3, atol=0.2)
    assert np.isclose(assoc_age, fitted_comp.age, rtol=0.1)

    return mixture, stars
