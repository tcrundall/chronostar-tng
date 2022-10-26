from astropy.table import Table
from typing import Optional
import numpy as np
from numpy.typing import NDArray
from numpy import float64


def construct_covs_from_data(
    X: NDArray[float64],
    dim: int = 6,
) -> tuple[NDArray[float64], NDArray[float64]]:
    """Reconstruct covariance matrices from data rows

    Parameters
    ----------
    X : NDArray[float64] of shape (n_samples, n_features)
        Input data
    dim : int, default 6
        The dimensions of the means and covariance matrices

    Notes
    -----
    Structure of data is assumed to be (in the case of `dim=6`):
    .. code::

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
    data: NDArray[float64],
    covs: Optional[NDArray[float64]] = None,
    n_draws: int = 100,
    dim: int = 6,
) -> NDArray[float64]:
    r"""Replace uncertainty covariances with a random sampling of the
    implied distribution

    Parameters
    ----------
    data : NDArray[float64] of shape (n_samples, n_features)
        Input data. If ``covs`` is None, then ``data`` should have
        ``dim`` + "\ ``dim``\ th triangle number" columns,
        with the final ``dim``\ th triangle number columns encoding
        covariance matrices
    covs : Optional[NDArray[float64]] of shape (n_samples, 6, 6), optional
        An array of covariance matrices, by default None
    n_draws : int, optional
        the number of random draws to take from each star's distribution, by default 100
    dim : int, optional
        dimensions, by default 6

    Returns
    -------
    NDArray[float64] of shape (n_samples \* n_draws, dim)
        A pseudo data set, where each initial sample is replaced by a
        swarm of samples
    """

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


def extract_array_from_table(
    table: Table, msk: Optional[NDArray] = None,
) -> NDArray[float64]:                                                  # type: ignore

    # Extract astrometry data, 6 measurements, 6 errors, 15 correlations
    # Assume gaia formatting
    if msk is None:
        msk = np.where(np.isfinite(table['radial_velocity']))           # type: ignore
    astrodata: NDArray[float64] = np.zeros((len(msk[0]), 6 + 6 + 15))   # type: ignore

    column_stems = [
        'ra', 'dec', 'parallax', 'pmra', 'pmdec', 'radial_velocity'
    ]
    dim = len(column_stems)
    # Insert measurements
    for i, stem in enumerate(column_stems):
        astrodata[:, i] = table[stem][msk]                      # type: ignore

    # Insert errors
    error_col_offset = dim
    for i, stem in enumerate(column_stems):
        error_col_name = f"{stem}_error"
        astrodata[:, error_col_offset + i] = table[error_col_name][msk]  # type: ignore

    # Insert correlations
    corr_col = 2 * dim
    for i in range(len(column_stems)):
        for j in range(i+1, len(column_stems)):
            corr_col_name = f"{column_stems[i]}_{column_stems[j]}_corr"
            if corr_col_name in table.keys():
                astrodata[:, corr_col] = table[corr_col_name][msk]  # type: ignore
            corr_col += 1

    # Insert covariances
    corr_col = 2 * dim
    for i in range(len(column_stems)):
        for j in range(i+1, len(column_stems)):
            corr_col_name = f"{column_stems[i]}_{column_stems[j]}_corr"
            error_1 = f"{column_stems[i]}_error"
            error_2 = f"{column_stems[j]}_error"
            if corr_col_name in table.keys():
                astrodata[:, corr_col] = (
                    table[corr_col_name][msk]                           # type: ignore
                    * table[error_1][msk]                               # type: ignore
                    * table[error_2][msk]                               # type: ignore
                )
            corr_col += 1

    return astrodata                                                    # type: ignore
