from ..context import chronostar        # noqa

import numpy as np
from chronostar.component.spherespacetimecomponent import\
    SphereSpaceTimeComponent


def test_sphereSTC():
    """Test various ages of sphere components

    Note that SphereSpaceTimeComponent is age accurate up to
    175 Myr.

    It's fitted covariance matrices remain accurate up to
    150 Myr.

    These limits are probs due to too tight a constraint on the
    the mean...
    """
    true_mean = np.ones(6)
    true_dxyz = 10.  # pc
    true_duvw = 2.   # km/s
    ages = [0., 10., 30., 75., 120., 180.]

    fitted_comps = []
    SphereSpaceTimeComponent.configure(
        nthreads=4,
        age_offset_interval=1,
        stellar_uncertainties=False,
    )
    for true_age in ages:
        print(f"----- {true_age=} -----")
        true_params = np.hstack((true_mean, true_dxyz, true_duvw, true_age))
        true_comp = SphereSpaceTimeComponent(true_params)

        seed = 0
        rng = np.random.default_rng(seed)

        n_stars = 1_000
        stars = rng.multivariate_normal(
            true_comp.mean,
            true_comp.covariance,
            size=n_stars
        )

        fitted_comp = SphereSpaceTimeComponent()
        fitted_comp.maximize(stars, resp=np.ones(n_stars))
        fitted_comp.maximize(stars, resp=np.ones(n_stars))
        fitted_comp.maximize(stars, resp=np.ones(n_stars))

        # Check ages
        assert np.allclose(
            true_params[-1],
            fitted_comp.parameters[-1],
            rtol=0.1,
            atol=1.,
        )

        fitted_comps.append(fitted_comp)

    return fitted_comps
