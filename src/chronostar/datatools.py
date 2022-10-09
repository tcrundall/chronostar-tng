import numpy as np
from numpy.typing import NDArray
from numpy import float64


def construct_covs_from_data(X, dim=6):
    """Reconstruct covariance matrices from data rows

    Parameters
    ----------
    X : NDArray[float64] of shape (n_samples, n_features)
        Input data

    Notes
    -----
    Structure of data is assumed to be:
    X, Y, Z, U, V, W, DX, DY, DZ, DU, DV, DW,
    XY_cov, XZ_cov, XU_cov, XV_cov, XW_cov,
            YZ_cov, YU_cov, YV_cov, ...
    etc.
    """
    n_samples = len(X)

    sample_means = X[:, :dim]

    # Fill in covariance matrices
    sample_covs = np.empty((n_samples, dim, dim))

    # Fill in errors
    error_col_offset = dim
    for i in range(dim):
        sample_covs[:, i, i] = X[:, error_col_offset + i]**2

    # Fill in all covariances
    # Assume covariances begin after dim feature cols and dim error cols
    cov_col_offset = 2 * dim
    indices = np.triu_indices(dim, 1)
    for data_col, (i, j) in enumerate(zip(indices[0], indices[1])):
        sample_covs[:, i, j] = X[:, cov_col_offset + data_col]
        sample_covs[:, j, i] = X[:, cov_col_offset + data_col]

    return sample_means, sample_covs


def replace_cov_with_sampling(
    data,
    covs=None,
    n_draws=100,
    dim=6,
) -> NDArray[float64]:

    if covs is None:
        sample_means, sample_covs = construct_covs_from_data(data, dim)
    else:
        sample_means, sample_covs = data, covs

    n_samples = len(data)

    new_data = np.empty((n_samples * n_draws, dim))

    print("Sampling star covariances")
    for i, (mean, cov) in enumerate(zip(sample_means, sample_covs)):
        if i % 1000 == 0:
            print(f"Sampled {i/n_samples*100:5.1f}% of input", end='\r')
        new_data[i * n_draws:(i+1) * n_draws] = \
            np.random.multivariate_normal(mean, cov, size=n_draws)
    print("\nDone")

    return new_data
