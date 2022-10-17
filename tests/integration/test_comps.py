from ..context import chronostar       # noqa

import numpy as np
from chronostar.component.spacecomponent import\
    SpaceComponent


def test_spacecomp():
    """Test various ages of sphere components

    Note that SphereSpaceTimeComponent is age accurate up to
    175 Myr.

    It's fitted covariance matrices remain accurate up to
    150 Myr.

    Returns
    -------
    _type_
        _description_
    """
    true_mean = np.ones(6)
    true_stdevs = np.array([10., 20., 30., 4., 5., 6])
    true_xy_corr = 0.5
    true_yv_corr = 0.3

    true_covariance = np.eye(6, 6) * true_stdevs**2

    true_covariance[0, 1] = true_covariance[1, 0] =\
        true_stdevs[0] * true_stdevs[1] * true_xy_corr

    true_covariance[1, 4] = true_covariance[4, 1] =\
        true_stdevs[1] * true_stdevs[4] * true_yv_corr

    true_params = np.hstack((true_mean, true_covariance.flatten()))

    SpaceComponent.configure(nthreads=1)
    true_comp = SpaceComponent(true_params)

    np.random.seed(0)
    rng = np.random.default_rng()

    n_stars = 1_000
    stars = rng.multivariate_normal(
        true_comp.mean,
        true_comp.covariance,
        size=n_stars
    )

    fitted_comp = SpaceComponent()
    fitted_comp.maximize(stars, resp=np.ones(n_stars))

    assert np.allclose(
        true_mean,
        fitted_comp.mean,
        rtol=0.2,
        atol=2.,
    )
    assert np.allclose(
        sorted(np.linalg.eigvals(true_covariance)),
        sorted(np.linalg.eigvals(fitted_comp.covariance)),
        rtol=0.1,
    )

    return fitted_comp
