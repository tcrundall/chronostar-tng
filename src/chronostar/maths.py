import numpy as np
from numpy.typing import NDArray
from numpy import float64
from numba import njit, prange


# @njit(parallel=True)
@njit
def extract_gaussian_pars(
    X: NDArray[float64],
) -> tuple[NDArray[float64], NDArray[float64]]:
    """Convert a single array where each row is a mean and flattened covariance
    into an array of means and an array of covariance matrices

    Parameters
    ----------
    X : NDArray[float64]
        A single array encoding many means and covariance matrices. For a
        given row, the first 6 elements are the mean, the next 36 elements
        are the covariance matrix, flattened

    Returns
    -------
    array of shape (nstars, 6), array of shape (nstars, 6, 6)
        The means and the covariance matrices
    """
    means = np.copy(X[:, :6])
    covs = np.copy(X[:, 6:]).reshape(-1, 6, 6)
    return means, covs


@njit(parallel=True)
# @njit
def estimate_log_gaussian_ol_prob(
    X: NDArray[float64],
    mean: NDArray[float64],
    covariance: NDArray[float64],
) -> NDArray[float64]:
    """
    A pythonic implementation of overlap integral calculation.

    Parameters
    ---------
    X: ndarray of shape (n_stars, 42)
        The first 6 columns are the mean of each star, the remaining
        36 are the flattened covariance matrix of each star
    mean: ndarray of shape (6)
        Mean of component Gaussian model (current day)
    covariance: ndarray of shape (6, 6)
        Covariance of component Gaussian model (current day)

    Returns
    -------
    ln_ols: ([nstars] float array)
        an array of the logarithm of the overlaps
    """
    data_means, data_covs = extract_gaussian_pars(X)
    lnols = np.empty(data_means.shape[0])
    for i in prange(len(data_means)):
        res = 0.
        res -= 6. * np.log(2*np.pi)
        res -= np.log(np.linalg.det(data_covs[i] + covariance))
        diff = data_means[i] - mean
        comb_cov = data_covs[i] + covariance
        res -= np.dot(diff.T, np.dot(np.linalg.inv(comb_cov), diff))
        res *= 0.5
        lnols[i] = res
    return lnols


def extract_gaussian_pars_py(
    X: NDArray[float64],
) -> tuple[NDArray[float64], NDArray[float64]]:
    """Convert a single array where each row is a mean and flattened covariance
    into an array of means and an array of covariance matrices

    Parameters
    ----------
    X : NDArray[float64]
        A single array encoding many means and covariance matrices. For a
        given row, the first 6 elements are the mean, the next 36 elements
        are the covariance matrix, flattened

    Returns
    -------
    tuple[NDArray[float64], NDArray[float64]]
        _description_
    """
    means = np.copy(X[:, :6])
    covs = np.copy(X[:, 6:]).reshape(-1, 6, 6)
    return means, covs


def estimate_log_gaussian_ol_prob_py(
    X: NDArray[float64],
    mean: NDArray[float64],
    covariance: NDArray[float64],
) -> NDArray[float64]:
    """
    A pythonic implementation of overlap integral calculation.
    Left here in case swigged _overlap doesn't work.

    Parameters
    ---------
    X: ndarray of shape (n_stars, 42)
        The first 6 columns are the mean of each star, the remaining
        36 are the flattened covariance matrix of each star
    mean: ndarray of shape (6)
        Mean of component Gaussian model (current day)
    covariance: ndarray of shape (6, 6)
        Covariance of component Gaussian model (current day)

    Returns
    -------
    ln_ols: ([nstars] float array)
        an array of the logarithm of the overlaps
    """
    # data_means, data_covs = extract_gaussian_pars(X)
    data_means, data_covs = extract_gaussian_pars_py(X)
    # data_covs = np.copy(data_means)
    lnols = np.empty(data_means.shape[0])
    for i, (data_mean, data_cov) in enumerate(zip(data_means, data_covs)):
        res = 0.
        res -= 6. * np.log(2*np.pi)
        res -= np.log(np.linalg.det(data_cov + covariance))
        diff = data_mean - mean
        comb_cov = data_cov + covariance
        res -= np.dot(diff.T, np.dot(np.linalg.inv(comb_cov), diff))
        res *= 0.5
        lnols[i] = res
    return lnols


def co2(A, a, B, b):
    """
    This is an alternative derivation of the overlap integral between
    two multivariate gaussians. This is the version implemented
    in the swigged C module.

    Parameters
    ----------
    A : (n x n) np.float array
        Covariance matrix of first Gaussian distribution
    a : (n) np.float array
        Mean of first Gaussian distribution
    B : (n x n) np.float array
        Covariance matrix of second Gaussian distribution
    b : (n) np.float array
        Mean of second Gaussian distribution
    """
    ApB = (A + B)
    ApB_det = np.linalg.det(ApB)
    ApB_i = np.linalg.inv(ApB)
    overlap = np.exp(-0.5 * (np.dot(a - b, np.dot(ApB_i, a - b))))
    overlap *= 1.0 / ((2 * np.pi) ** 3.0 * np.sqrt(ApB_det))
    return overlap
