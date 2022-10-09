import numpy as np

from ..context import chronostar        # noqa
from chronostar import datatools


def test_replace_covs():
    np.random.seed(0)

    dim = 6
    n_samples = 1_000

    data_mean = np.array([10., 10., 10., 20., 20., 20.])
    data_std = np.array([40., 50., 60., 70., 80., 90.])

    data = np.random.uniform(-10, 10, size=(n_samples, dim))
    data = np.random.randn(n_samples, dim) * data_std + data_mean

    stdevs = np.array([1., 2., 3., 4., 5., 6.])
    covariance = np.eye(6) * stdevs**2

    covariances = np.repeat(covariance[np.newaxis], n_samples, axis=0)

    n_draws = 100
    new_data = datatools.replace_cov_with_sampling(
        data,
        covariances,
        n_draws=n_draws,
        dim=dim
    )

    assert np.allclose(data_mean, np.mean(new_data, axis=0), atol=3.)
    assert np.allclose(data_std, np.std(new_data, axis=0), atol=3.)

    new_covariance = np.cov(new_data[:n_draws].T)
    assert np.allclose(covariance, new_covariance, rtol=0.2, atol=3.)

    return new_data
