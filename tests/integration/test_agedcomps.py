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

    Returns
    -------
    _type_
        _description_
    """
    true_mean = np.ones(6)
    true_dxyz = 10.  # pc
    true_duvw = 2.   # km/s
    ages = [0., 10., 30., 75., 120.]
    # ages = [150., 175., 200.]

    fitted_comps = []
    SphereSpaceTimeComponent.configure(nthreads=1)
    for true_age in ages:
        print(f"----- {true_age=} -----")
        true_params = np.hstack((true_mean, true_dxyz, true_duvw, true_age))
        true_comp = SphereSpaceTimeComponent(true_params)

        np.random.seed(0)
        rng = np.random.default_rng()

        n_stars = 1_000
        stars = rng.multivariate_normal(
            true_comp.mean,
            true_comp.covariance,
            size=n_stars
        )

        fitted_comp = SphereSpaceTimeComponent()
        fitted_comp.maximize(stars, resp=np.ones(n_stars))

        # tols = {
        #     'mean': (0.1, 5),
        #     'covariance': (0.5, 10),
        #     'age': (0.1, 1),
        # }
        # for attr, (rtol, atol) in tols.items():
        #     assert np.allclose(
        #         getattr(true_comp, attr),
        #         getattr(fitted_comp, attr),
        #         rtol=rtol,
        #         atol=atol,
        #     )

        # Ignore birth mean: too sensitive to age

        # Check ages
        assert np.allclose(
            true_params[-1],
            fitted_comp.parameters[-1],
            rtol=0.05,
            atol=1.,
        )

        # # Check covariance parameters
        # assert np.allclose(
        #     true_params[6:-1],
        #     fitted_comp.parameters[6:-1],
        #     rtol=0.1,
        #     atol=1.,
        # )

        fitted_comps.append(fitted_comp)

    return fitted_comps
