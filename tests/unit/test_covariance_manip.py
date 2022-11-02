"""test covariance manipulation"""

import numpy as np


def test_construction_from_eigenvals():
    dim = 6
    nsamples = 100

    true_mean = np.zeros(dim)
    true_stdev = 30.
    true_cov = true_stdev**2 * np.eye(dim)
    nsamples = 100
    rng = np.random.default_rng(seed=0)
    data = rng.multivariate_normal(mean=true_mean, cov=true_cov, size=nsamples)
    return data


if __name__ == '__main__':
    data = test_construction_from_eigenvals()

    mean = np.mean(data, axis=0)
    covariance = np.cov(data.T)
    eigvals, eigvecs = np.linalg.eigh(covariance)

    prim_axis_length = np.sqrt(np.max(eigvals))
    prim_axis = eigvecs[np.argmax(eigvals)]

    new_mean_1 = mean + prim_axis_length * prim_axis / 2.0
    new_mean_2 = mean - prim_axis_length * prim_axis / 2.0

    new_eigvals = eigvals[:]
    new_eigvals[np.argmax(eigvals)] /= 2.0

    D = np.eye(6) * eigvals

    new_covariance = np.dot(eigvecs,  np.dot(D,  eigvecs.T))
