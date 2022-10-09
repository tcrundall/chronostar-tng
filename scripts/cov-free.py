#!/usr/bin/env python
# coding: utf-8


import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def slow_get_lnoverlaps(g_cov, g_mn, st_covs, st_mns, dummy=None):
    """
    A pythonic implementation of overlap integral calculation.
    Left here in case swigged _overlap doesn't work.

    Parameters
    ---------
    g_cov: ([6,6] float array)
        Covariance matrix of the group
    g_mn: ([6] float array)
        mean of the group
    st_covs: ([nstars, 6, 6] float array)
        covariance matrices of the stars
    st_mns: ([nstars, 6], float array)
        means of the stars
    dummy: {None}
        a place holder parameter such that this function's signature
        matches that of the c implementation, which requires an
        explicit size of `nstars`.

    Returns
    -------
    ln_ols: ([nstars] float array)
        an array of the logarithm of the overlaps
    """
    lnols = []
    dim = len(g_mn)
    for st_cov, st_mn in zip(st_covs, st_mns):
        res = 0
        res -= dim * np.log(2*np.pi)
        res -= np.log(np.linalg.det(g_cov + st_cov))
        stmg_mn = st_mn - g_mn
        stpg_cov = st_cov + g_cov
        res -= np.dot(stmg_mn.T, np.dot(np.linalg.inv(stpg_cov), stmg_mn))
        res *= 0.5
        lnols.append(res)
    return np.array(lnols)


def stochastic_lnoverlap(g_cov, g_mn, st_covs, st_mns, nsamples=100):
    lnols = []
    rng = np.random.default_rng()
    for st_mn, st_cov in zip(st_mns, st_covs):
        # sample a bunch of points from star
        pts = rng.multivariate_normal(st_mn, st_cov, size=nsamples)
        lnol = np.log(np.mean(stats.multivariate_normal.pdf(pts, g_mn, g_cov)))
        lnols.append(lnol)

    return np.array(lnols)


def compare_cov_free(dim, nsamples, n_stars=100):
    N_STARS, DIM = 10, 6
    group_mean = np.zeros(DIM)
    group_cov = np.eye(DIM)

    star_means = np.random.rand(N_STARS * DIM).reshape(N_STARS, DIM)
    star_covs = np.broadcast_to(np.eye(DIM), (N_STARS, DIM, DIM))

    integrated_ols = slow_get_lnoverlaps(
        group_cov, group_mean, star_covs, star_means
    )
    stochastic_ols = stochastic_lnoverlap(
        group_cov, group_mean, star_covs, star_means, nsamples=nsamples
    )

    average_error = np.mean(np.abs(
        np.exp(integrated_ols) - np.exp(stochastic_ols)     # type: ignore
    ) / np.exp(integrated_ols))
    return average_error


for dim in [1, 6, 10]:
    nsamples = [10, 25, 100, 1_000, 10_000]

    average_errors = [
        compare_cov_free(dim, nsample, n_stars=1_000_000)
        for nsample in nsamples
    ]
    plt.plot(nsamples, average_errors, label=dim)

plt.legend()
plt.xscale('log')
plt.xlabel("Number of random samples")
plt.ylabel("Relative error")
plt.savefig("cov-free.png")
