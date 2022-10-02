import numpy as np
from scipy.stats import multivariate_normal

# from src.chronostar.component.spacetimecomponent import SpaceTimeComponent
from src.chronostar.component.spacecomponent import SpaceComponent
from tests.fooclasses import CONFIG_PARAMS, DATA, NSAMPLES


COMPONENT_CLASSES = [
    SpaceComponent,
    # SpaceTimeComponent
]


def test_construction():
    for CompClass in COMPONENT_CLASSES:
        comp = CompClass(CONFIG_PARAMS['component'])   # noqa F841


def test_simpleusage():
    for CompClass in COMPONENT_CLASSES:
        comp = CompClass(CONFIG_PARAMS['component'])
        comp.maximize(DATA, np.ones(NSAMPLES))
        result = comp.estimate_log_prob(DATA)
        assert result.shape[0] == NSAMPLES


def test_usage():
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
        cov=true_cov,               # type: ignore
    )

    # Instantiate, maximize, and check log_probs
    for CompClass in COMPONENT_CLASSES:
        comp = CompClass(CONFIG_PARAMS['component'])
        comp.maximize(data, np.ones(NSAMPLES))

        log_probs = comp.estimate_log_prob(DATA)

        assert np.allclose(log_probs, true_log_probs, rtol=5e-2)
