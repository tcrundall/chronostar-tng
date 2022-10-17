"""
A module which aids in the transformation of a covariance matrix
between two coordinate frames.

These functions are used to convert astrometric covariance matrices into
cartesian covariance matrices. Also to project cartesian Gaussian distributions
forward (or backward) through the Galactic potential.
"""

import numpy as np
# from numba import jit
from numba import njit


# @jit(nopython=True)
@njit(parallel=True, cache=True)
def calc_jacobian(trans_func, loc, dim=6, h=1e-3, args=()):
    """
    Calculate the Jacobian of the coordinate transfromation `trans_func` about
    `loc`.

    `trans_func` should take a vector of dimension `dim` to a new vector
    of same dimension. This function then calculates the 2nd order partial
    derivative at point `loc`. Extra arguments for `trans_func` can be
    provided as a tuple to `args`.

    Parameters
    ----------
    trans_func : function
        Transformation function taking us from the initial coordinate frame
        to the final coordinate frame
    loc : [dim] float array
        The position (in the initial coordinte frame) around which we are
        calculating the jacobian
    dim : int {6}
        The dimensionality of the coordinate frames
    h : float {1e-3}
        The size of the increment, smaller values maybe run into numerical
        issues
    args : tuple {None}
        Extra arguments required by `trans_func`

    Returns
    -------
    [dim,dim] float array
        A jacobian matrix
    [dim] float array
        Transformed location

    Notes
    -----
        OPTIMISATION TARGET
    The application of `trans_func` is the bottleneck of Chronostar
    (at least when `trans_func` is traceorbit.trace_cartesian_orbit).
    Since this is a loop, there is scope for parallelisation.
    """

    jac = np.zeros((dim, dim))

    # Even with epicyclic, this constitutes 90% of chronostar work
    # so, we pass all 12 required positions to the trans_func as
    # one array, to exploit numpy's faster array operations
    start_pos = np.zeros((2*dim + 1, dim))
    for i in range(dim):
        offset = np.zeros(dim)
        offset[i] = h
        loc_pl = loc + offset
        loc_mi = loc - offset
        start_pos[2*i] = loc_pl
        start_pos[2*i + 1] = loc_mi

    # Also transform the location
    start_pos[-1] = loc

    final_pos = trans_func(start_pos, *args)
    # final_pos = trans_func(start_pos, 1.)

    for i in range(dim):
        jac[:, i] = (final_pos[2*i] - final_pos[2*i + 1]) / (2*h)

    return jac, final_pos[-1]


# @jit(nopython=True)
@njit(cache=True)
def transform_covmatrix(
    cov,
    trans_func,
    loc,
    dim=6,
    h=1e-3,
    args=(),
):
    """
    Transforming a covariance matrix from one coordinate frame to another

    Parameters
    ----------
    cov : [dim,dim] float array
        Covariance matrix in the initial frame
    trans_func : function
        Transformation function taking us from the initial
        coordinate frame to the final coordinate frame. Output must be
        mutable, i.e. single value, or an array
    loc : [dim] float array
        The position (in the initial coordinate frame)
        around which we are calculating the jacobian
        (i.e. the mean, in the example of a Gaussian distribution)
    dim : integer {6}
        The dimensionality of the coordinate frame
    h : float {1e-3}
        The size of the increment, smaller values maybe run into numerical
        issues
    args : tuple
        extra args to be passed to trans_func. E.g. for traceOrbitXYZUVW
        args = (age,) [for traceforward] or args = (-age,) [for traceback]

    Returns
    -------
    [dim,dim] float array
        The transformed covariance matrix
    [dim] float array
        The transformed location
    """

    jac, final_loc = calc_jacobian(trans_func, loc, dim=dim, h=h, args=args)
    return np.dot(jac, np.dot(cov, jac.T)), final_loc
