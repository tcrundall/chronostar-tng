"""
Module for tracing orbits using the epicyclic approximation
"""
import numpy as np
from numba import jit

from .utils.coordinate import convert_curvilin2cart, convert_cart2curvilin


@jit(nopython=True)
def epicyclic_approx(data, times=None, sA=0.89, sB=1.15, sR=1.21):
    r"""
    MZ (2020 - 01 - 17)

    Epicyclic approximation following the Makarov et al. 2004 paper
    in the curvilinear coordinate system:
    The radial component xi is pointing towards the Galactic center
    at all times and equals 0 at R0.
    The circular component eta circles around the Galaxy; eta = phi\*R.
    The vertical component is defined as a displacement from the
    Galactic plane.

    This approximation works close to the LSR.

    Parameters
    ------------
    data : [pc, pc\*, pc, km/s, km/s, km/s]
           xi, eta, zeta, xidot, etadot, zetadot

        # \*parsecs in the eta component are scales parsecs...
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
    kappa = np.sqrt(-4.0 * B * (A-B))  # Myr-1
    nu = np.sqrt(4.0*np.pi*Grho + (A+B)*(A-B))  # Myr-1

    t = times

    kt = kappa * t
    nt = nu * t

    xi = xi0 \
        + xidot0/kappa * np.sin(kt) \
        + (etadot0 - 2.0*A*xi0) * (1.0-np.cos(kt)) / (2.0*B)

    eta = eta0 \
        - xidot0 * (1.0-np.cos(kt)) / (2.0*B) \
        + etadot0 * (A*kt - (A-B) * np.sin(kt)) / (kappa*B) \
        - xi0 * 2.0 * A * (A-B) * (kt-np.sin(kt)) / (kappa*B)

    zeta = zeta0 * np.cos(nt) + zetadot0 / nu * np.sin(nt)

    xidot = xidot0 * np.cos(kt) \
        + (etadot0 - 2.0*A*xi0) * kappa * np.sin(kt) / (2.0*B)

    etadot = -xidot0 * kappa/(2.0*B) * np.sin(kt) \
        + etadot0/B * (A - (A-B) * np.cos(kt)) \
        - 2.0 * A * xi0 * (A-B) * (1.0-np.cos(kt)) / B

    zetadot = -zeta0*nu*np.sin(nt) + zetadot0 * np.cos(nt)

    new_position = np.vstack((xi, eta, zeta, xidot, etadot, zetadot))
    new_position = new_position.T
    return new_position


@jit(nopython=True)
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
    xyzuvw_start : float array of shape (6)
        Starting point of orbit: x, y, z, u, v, w with pc units for position
        and km/s units for velocity

    time : (float) or ([ntime] float array)
        One (or many) times by which to calculate the orbit

    Returns
    -------
    xyzuvw_tf : [ntime, 6] array
        [pc, pc, pc, km/s, km/s, km/s] - the traced orbit with positions
        and velocities
    """
    # In order for Jit to work, we remove all array shape checks

    # Units: Velocities are in km/s, convert into pc/Myr
    # This is fine to do inplace, because we undo it at end of function
    xyzuvw_start[:, 3:] *= 1.0227121650537077  # pc/Myr

    # Transform to curvilinear
    curvilin = convert_cart2curvilin(xyzuvw_start, ro=ro, vo=vo)

    # Trace orbit with epicyclic approx.
    new_position = epicyclic_approx(curvilin, times=time, sA=sA, sB=sB, sR=sR)

    # Transform back to cartesian
    xyzuvw_new = convert_curvilin2cart(new_position, ro=ro, vo=vo)

    # Units: Transform velocities from pc/Myr back to km/s
    xyzuvw_new[:, 3:] /= 1.0227121650537077

    # Undo inplace conversion
    xyzuvw_start[:, 3:] /= 1.0227121650537077  # pc/Myr

    return xyzuvw_new
