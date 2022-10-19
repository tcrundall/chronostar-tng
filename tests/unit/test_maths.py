import numpy as np
from scipy import stats

from ..context import chronostar        # noqa F401
from chronostar.maths import estimate_log_gaussian_ol_prob, co2


def test_logoverlapsbasic():

    # A super narrow data covariance, should serve as the dirac delta function
    model_mean = np.zeros(6)
    model_cov = np.eye(6)

    n_stars = 100
    data_means = np.random.rand(n_stars, 6)
    data_covs = np.broadcast_to(np.eye(6) / 10_000_000., shape=(n_stars, 6, 6))

    data = np.vstack((data_means.T, data_covs.reshape(-1, 36).T)).T

    log_overlaps = estimate_log_gaussian_ol_prob(data, model_mean, model_cov)

    assert np.allclose(
        np.log(stats.multivariate_normal.pdf(
            data_means, model_mean, model_cov                 # type: ignore
        )),
        log_overlaps,
    )


def test_logoverlapsmedim():
    # Confirm overlap decreases with broader data covariances

    mean = np.zeros(6)
    model_cov = np.eye(6)

    n_stars = 100
    data_covs = np.empty((n_stars, 6, 6))

    for i in range(n_stars):
        data_covs[i] = np.eye(6) * (i+1)**2

    data = np.vstack((
        np.broadcast_to(mean, (n_stars, 6)).T,
        data_covs.reshape(-1, 36).T,
    )).T

    log_overlaps = estimate_log_gaussian_ol_prob(data, mean, model_cov)
    # Confirm each element is greater than the next element
    assert np.all(log_overlaps[:-1] > log_overlaps[1:])


def test_differentimplementations():
    np.random.seed(0)

    # Generate a bunch of full cov matrices by calculating covariance from
    # random points
    n_points = 100
    points = np.random.rand(n_points, 6)

    n_samples = 90
    means = np.empty(shape=(n_samples, 6))
    covs = np.empty(shape=(n_samples, 6, 6))
    for i in range(n_samples):
        means[i] = points[i]
        covs[i] = np.cov(points[i:i+10].T)

    data = np.vstack((means.T, covs.flatten().reshape(-1, 36).T)).T

    ln_ols = estimate_log_gaussian_ol_prob(data, means[0], covs[0])

    ln_ols_co2 = np.empty(n_samples)
    for i in range(n_samples):
        ln_ols_co2[i] = np.log(co2(covs[0], means[0], covs[i], means[i]))

    assert np.allclose(ln_ols, ln_ols_co2)
