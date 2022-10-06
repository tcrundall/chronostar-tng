import numpy as np
# from numba import jit


# @jit(nopython=True)
def convert_cart2curvilin(data, ro=8., vo=220.):
    """
    MZ (2020 - 01 - 17)

    Converts cartesian coordinates XYZUVW (given with respect to the
    LSR) to the curvilinear system. Curvilinear system is corotating
    so its radial component xi is always pointing towards the galactic
    center. Coordinates in the curvilinear system are
    [xi, eta, zeta, xidot, etadot, zetadot]

    Parameters
    ----------
    data: [6, (npoints)] float np.array
        [pc, pc, pc, km/s,km/s,km/s]
        [X,  Y,  Z,  U,   V,   W]

    Returns
    -------
    curvilin_coordinates: [6, (npoints)] float np.array
        xi     : radial distance from the origin in LSR
        eta    :
        zeta   : vertical distance from plane
        xidot  :
        etadot :
        zetadot:

    """
    # data = np.array(data)

    X, Y, Z, U, V, W = data.T

    R0 = ro * 1000.0  # pc
    Omega0 = vo / R0  # km/s / pc

    # Place the velocities in a rotating frame
    U = U - Y * Omega0
    V = V + X * Omega0

    R = np.sqrt(Y**2 + (R0-X)**2)
    phi = np.arctan2(Y, R0-X)

    xi = R0 - R
    eta = phi * R0
    zeta = Z
    xidot = U * np.cos(phi) - V * np.sin(phi)
    etadot = R0 / R * (V * np.cos(phi) + U * np.sin(phi))
    zetadot = W

    curvilin_coordinates = np.vstack((xi, eta, zeta, xidot, etadot, zetadot))

    return curvilin_coordinates.T


# @jit(nopython=True)
def convert_curvilin2cart(data, ro=8., vo=220.,
                          lsr_centered=True):
    """
    MZ (2020 - 01 - 17)

    Returns
    -------

    """

    xi, eta, zeta, xidot, etadot, zetadot = data.T

    R0 = ro * 1000.0

    R = R0 - xi
    phi = eta / R0

    X = xi * np.cos(phi) + R0 * (1.0 - np.cos(phi))  # R0 - R*np.cos(phi)
    Y = R * np.sin(phi)
    Z = zeta

    U = xidot * np.cos(phi) + R / R0 * etadot * np.sin(phi)
    V = - xidot * np.sin(phi) + R / R0 * etadot * np.cos(phi)
    W = zetadot

    # Convert to a non-rotating observed frame
    Omega0 = vo / R0  # km/s / pc
    U = U + Y * Omega0
    V = V - X * Omega0

    cart_coordinates = np.vstack((X, Y, Z, U, V, W))

    return cart_coordinates.T


# @jit(nopython=True)
def epicyclic_approx(data, times=None, sA=0.89, sB=1.15, sR=1.21):
    """
    MZ (2020 - 01 - 17)

    Epicyclic approximation following the Makarov et al. 2004 paper
    in the curvilinear coordinate system:
    The radial component xi is pointing towards the Galactic center
    at all times and equals 0 at R0.
    The circular component eta circles around the Galaxy; eta = phi*R.
    The vertical component is defined as a displacement from the
    Galactic plane.

    This approximation works close to the LSR.

    Parameters
    ------------
    data : [pc, pc*, pc, km/s, km/s, km/s]
           xi, eta, zeta, xidot, etadot, zetadot

        # *parsecs in the eta component are scales parsecs...
    """
    xi0, eta0, zeta0, xidot0, etadot0, zetadot0 = data.T

    # Bovy 2017
    A0 = 15.3  # km/s/kpc
    B0 = -11.9  # km/s/kpc

    # Bovy 2017. Scale factors to match MW2014.
    # Mike's scale factors
    # A0 = 0.97*15.3 # km/s/kpc
    # B0 = 1.12*(-11.9) # km/s/kpc

    # Marusa's factors
    A0 = A0 * sA  # km/s/kpc
    B0 = B0 * sB  # km/s/kpc

    # Unit conversion: convert from km/s/kpc to Myr-1
    A = A0 * 0.0010227121650537077  # Myr-1
    B = B0 * 0.0010227121650537077  # Myr-1

    # Fine tuning rho
    rho_scale_factor = sR  # 1.36
    rho = rho_scale_factor * 0.0889  # M0/pc3
    Grho = rho * 0.004498502151575285  # Myr-2; rho should be given in M0/pc3
    kappa = np.sqrt(-4.0 * B * (A - B))  # Myr-1
    nu = np.sqrt(4.0 * np.pi * Grho + (A + B) * (A - B))  # Myr-1

    t = times

    kt = kappa * t
    nt = nu * t

    xi = xi0 \
        + xidot0 / kappa * np.sin(kt) \
        + (etadot0 - 2.0 * A * xi0) * (1.0 - np.cos(kt)) / (2.0*B)

    eta = eta0 \
        - xidot0 * (1.0 - np.cos(kt)) / (2.0*B) \
        + etadot0 * (A*kt - (A-B) * np.sin(kt)) / (kappa * B) \
        - xi0 * 2.0 * A * (A - B) * (kt - np.sin(kt)) / (kappa * B)

    zeta = zeta0 * np.cos(nt) + zetadot0 / nu * np.sin(nt)

    xidot = xidot0 * np.cos(kt) \
        + (etadot0 - 2.0 * A * xi0) * kappa * np.sin(kt) / (2.0 * B)

    etadot = -xidot0 * kappa/(2.0 * B) * np.sin(kt) \
        + etadot0/B * (A - (A - B) * np.cos(kt)) \
        - 2.0 * A * xi0 * (A-B) * (1.0 - np.cos(kt)) / B

    zetadot = -zeta0 * nu * np.sin(nt) + zetadot0 * np.cos(nt)

    new_position = np.vstack((xi, eta, zeta, xidot, etadot, zetadot))
    new_position = new_position.T
    return new_position


# @jit(nopython=True)
def trace_epicyclic_orbit(
    xyzuvw_start,
    time=None,
    sA=0.89,
    sB=1.15,
    sR=1.21,
    ro=8.,
    vo=220.
):
    """
    MZ (2020 - 01 - 17)

    Given a star's XYZUVW relative to the LSR (at any time), project its
    orbit forward (or backward) to each of the time listed in *time*
    using epicyclic approximation. This only works close to the LSR.

    Positive time --> traceforward
    Negative time --> traceback

    Parameters
    ----------
    xyzuvw : [pc,pc,pc,km/s,km/s,km/s]
    time : (float) or ([ntime] float array)
        Myr - time of 0.0 must be present in the array. time need not be
        spread linearly.
    #TODO: time 0.0 really? [TC: this was true for galpy]
    single_age: (bool) {True}
        Set this flag if only providing a single age to trace to
        This is there for the plotting purposes.

    Returns
    -------
    xyzuvw_tf : [ntime, 6] array
        [pc, pc, pc, km/s, km/s, km/s] - the traced orbit with positions
        and velocities
    """
    # if single_age:
    #     # replace 0 with some tiny number
    #     try:
    #         if time == 0.:
    #             time = 1e-15
    #         # time = np.array([0., time])
    #     except ValueError as err:
    #         if not err.args:
    #             err.args = ('',)
    #         err.args = err.args + ('WARNING: comparing array to float? '
    #                                'Did you leave single_age as True?',)
    #         raise

    # else:
    #     raise UserWarning('Multi age orbit integation no longer supported')
    #     time = np.array(time)

    # Make sure numbers are floats, and reshape into 2d
    assert len(xyzuvw_start.shape) == 2
    # xyzuvw_start = np.atleast_2d(xyzuvw_start)  # .astype(float))

    # Units: Velocities are in km/s, convert into pc/Myr
    xyzuvw_start[:, 3:] = xyzuvw_start[:, 3:] * 1.0227121650537077  # pc/Myr

    # Transform to curvilinear
    curvilin = convert_cart2curvilin(xyzuvw_start, ro=ro, vo=vo)

    # Trace orbit with epicyclic approx.
    new_position = epicyclic_approx(curvilin, times=time, sA=sA, sB=sB, sR=sR)

    # Transform back to cartesian
    xyzuvw_new = convert_curvilin2cart(new_position, ro=ro, vo=vo)

    # Units: Transform velocities from pc/Myr back to km/s
    xyzuvw_new[:, 3:] /= 1.0227121650537077

    # Remove empty dimensions
    return xyzuvw_new
    # return np.squeeze(xyzuvw_new)
