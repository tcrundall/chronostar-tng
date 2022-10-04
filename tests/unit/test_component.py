import numpy as np
from scipy.stats import multivariate_normal
from src.chronostar.base import BaseComponent

from src.chronostar.component.spacetimecomponent import SpaceTimeComponent
from src.chronostar.component.spacecomponent import SpaceComponent
from tests.unit.fooclasses import CONFIG_PARAMS, DATA, NSAMPLES


# Component classes and default extra parameters
COMPONENT_CLASSES: dict[type[BaseComponent], tuple] = {
    SpaceComponent: (),
    SpaceTimeComponent: (1.,),
}


def test_construction() -> None:
    for CompClass, extra_params in COMPONENT_CLASSES.items():
        CompClass.configure(**CONFIG_PARAMS['component'])
        params = (
            np.zeros(6),
            np.eye(6),
            *extra_params
        )
        comp = CompClass(params)        # noqa F841


def test_simpleusage() -> None:
    for CompClass in COMPONENT_CLASSES.keys():
        CompClass.configure(**CONFIG_PARAMS['component'])
        comp = CompClass()
        comp.maximize(DATA, np.ones(NSAMPLES))
        result = comp.estimate_log_prob(DATA)
        assert result.shape[0] == NSAMPLES


def test_splitting() -> None:
    dim = 6
    mean = np.zeros(dim)
    stdev = 10.
    primary_stdev = 2 * stdev
    covariance = stdev**2 * np.eye(dim)
    covariance[0, 0] = primary_stdev**2

    true_prim_axis = np.zeros(dim)
    true_prim_axis[0] = 1.

    true_prim_axis_len = primary_stdev

    true_mean_1 = mean + true_prim_axis_len * true_prim_axis / 2.0
    true_mean_2 = mean - true_prim_axis_len * true_prim_axis / 2.0
    true_new_covariance = stdev**2 * np.eye(dim)

    for CompClass, default_params in COMPONENT_CLASSES.items():
        # Skip classes that don't have split
        CompClass.configure(**CONFIG_PARAMS['component'])
        comp = CompClass((mean, covariance, *default_params))
        # comp.set_parameters((mean, covariance, *default_params))
        c1, c2 = comp.split()
        mean_1, new_covariance, *_ = c1.get_parameters()
        mean_2, _, *_ = c2.get_parameters()

        assert np.allclose(true_mean_1, mean_1)
        assert np.allclose(true_mean_2, mean_2)
        assert np.allclose(true_new_covariance, new_covariance)


def test_usage() -> None:
    # Generate simple data set
    true_mean = np.zeros(6)
    true_stdev = 30.
    true_cov = true_stdev**2 * np.eye(6)
    nsamples = 100
    rng = np.random.default_rng()
    data = rng.multivariate_normal(mean=true_mean, cov=true_cov, size=nsamples)

    true_log_probs = multivariate_normal.logpdf(
        DATA,
        mean=true_mean,
        cov=true_cov,       # type: ignore
    )

    # Instantiate, maximize, and check log_probs
    for CompClass in COMPONENT_CLASSES:
        CompClass.configure(**CONFIG_PARAMS['component'])
        comp = CompClass()

        # a log_resp of 0 is a resp of 1 (i.e. full responsibility)
        comp.maximize(data, log_resp=np.zeros(NSAMPLES))

        log_probs = comp.estimate_log_prob(DATA)

        assert np.allclose(log_probs, true_log_probs, rtol=5e-2)

if __name__ == '__main__':
    test_simpleusage()
