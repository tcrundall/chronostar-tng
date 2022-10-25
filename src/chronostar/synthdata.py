"""
synthdata.py
Used for testing only.

A class plus a couple of functions helpful for generating synthetic data sets.

synthesiser
Used to generate realistic data for one (or many) synthetic association(s)
with multiple starburst events along with a background as desired.
From a parametrised gaussian distribution, generate the starting
XYZUVW values for a given number of stars
TODO: accommodate multiple groups
"""
from typing import Union
import numpy as np
from numpy import float64
from numpy.typing import NDArray
from chronostar.base import BaseComponent

from chronostar.utils.transform import transform_covmatrix
from .traceorbit import trace_epicyclic_orbit

from astropy.table import Table

from .utils import coordinate
from .component.spherespacetimecomponent import SphereSpaceTimeComponent
# from . import traceorbit


def generate_association(
    mean_now,
    covariance_birth,
    age,
    nstars=100,
    rng=None,
) -> NDArray[float64]:
    mean_birth = trace_epicyclic_orbit(mean_now[np.newaxis], -age)
    covariance_aged, _ = transform_covmatrix(
        covariance_birth,
        trace_epicyclic_orbit,
        mean_birth,
        args=(age,),
    )

    if rng is None:
        rng = np.random.default_rng()
    # import ipdb; ipdb.set_trace()
    return rng.multivariate_normal(mean_now, covariance_aged, size=nstars)


def generate_two_overlapping(
    age1,
    age2,
    nstars1=1_000,
    nstars2=1_000,
    rng=None
):
    dim = 6
    X_OFFSET = 50.
    V_OFFSET = 5.
    DV = 3.
    mean1 = np.zeros(dim)
    mean2 = np.copy(mean1)
    mean2[0] = X_OFFSET
    mean2[4] = V_OFFSET

    stdevs1 = np.array([age1*DV] * 3 + [DV] * 3)
    cov1 = np.eye(dim) * stdevs1

    stdevs2 = np.array([age2*DV] * 3 + [DV] * 3)
    cov2 = np.eye(dim) * stdevs2

    if rng is None:
        rng = np.random.default_rng()

    stars1 = generate_association(mean1, cov1, age1, nstars=nstars1, rng=rng)
    stars2 = generate_association(mean2, cov2, age2, nstars=nstars2, rng=rng)

    all_stars = np.vstack((stars1, stars2))
    return all_stars


MaybeListCompClass = Union[list[type[BaseComponent]], type[BaseComponent]]
ListCompClass = list[type[BaseComponent]]


class SynthData():
    # BASED ON MEDIAN ERRORS OF ALL GAIA STARS WITH RVS
    # AND <20% PARALLAX ERROR
    GERROR = {
        'ra_error': 0.05,  # deg
        'dec_error': 0.05,  # deg
        'parallax_error': 0.06,  # e_Plx [mas]
        'radial_velocity_error': 2.5,  # e_RV [km/s]
        'pmra_error': 0.06,  # e_pm [mas/yr]
        'pmdec_error': 0.06,  # e_pm [mas/yr]
    }

    DEFAULT_ASTR_COLNAMES = (
        'ra', 'dec', 'parallax', 'pmra', 'pmdec', 'radial_velocity',
    )

    DEFAULT_NAMES = (
        # 'name', 'component', 'age',
        # 'x0', 'y0', 'z0', 'u0', 'v0', 'w0',
        # 'x_now', 'y_now', 'z_now', 'u_now', 'v_now', 'w_now',
        'source_id',
        'ra', 'ra_error', 'dec', 'dec_error', 'parallax', 'parallax_error',
        'pmra', 'pmra_error', 'pmdec', 'pmdec_error',
        'radial_velocity', 'radial_velocity_error',
    )

    # DEFAULT_DTYPES = tuple(['S20', 'S2']
    #                        + (len(DEFAULT_NAMES)-2) * ['float64'])
    DEFAULT_DTYPES = tuple(['int'] + (len(DEFAULT_NAMES) - 1) * ['float64'])

    # Define here, because apparently i can't be trusted to type a string
    # in a correct order
    cart_labels = 'xyzuvw'
    # m_err = 1.0
    m_err = 0.5

    def __init__(
        self,
        pars,
        starcounts,
        measurement_error=1.0,
        ComponentClass=SphereSpaceTimeComponent,
        # savedir=None,
        # tablefilename=None,
        # background_density=None,
        # bg_span_scale=1.2,
        component_config=None,
    ):
        """
        Generates a set of astrometry data based on multiple star bursts with
        simple, Gaussian origins.
        Parameters
        ----------
            pars : list of length (ncomps)
                A list of parameters for each component, where each
                element of list is a 1D numpy array
            starcounts : [ncomps] int array
                Number of stars generated for each component
            measurement_error : float
                proportion of measurement error to incorporate.
                1. -> gaia
                2. -> twice as bad as Gaia
                0.5 -> twice as good as Gaia
                See GERROR attribute for reference values
            Components : Component class {SphereComponent}
                The component class to construct component objects
                from input parameters `pars`
            savedir : string {None}
                Directory in which to save generated dataset
            tablefilename : string {None}
                Filename to save generated dataset to
            background_density : float {None}
                Density of background stars, number of stars per pc^3 (km/s)^3
        """
        # Tidying input and applying some quality checks
        if type(pars) is not list:
            raise UserWarning("Expected a list of numpy arrays for 'pars'")

        # self.pars = pars      # Can different rows of pars be
        #                       # provided in different forms?

        self.ncomps = len(pars)
        self.starcounts = starcounts

        assert len(self.starcounts) ==\
            self.ncomps,\
            'starcounts must be same length as pars dimension. Received' \
            'lengths starcounts: {} and pars: {}'.format(
                len(self.starcounts),
                self.ncomps,
        )

        self.m_err = measurement_error

        if component_config is not None:
            ComponentClass.configure(**component_config)
        self.components = [ComponentClass(p) for p in pars]

        self.table = Table(names=self.DEFAULT_NAMES,
                           dtype=self.DEFAULT_DTYPES)

    def generate_init_cartesian(
        self,
        component: SphereSpaceTimeComponent,
        starcount,
        component_name='',
        seed=None
    ):
        """Generate initial xyzuvw based on component"""
        # For testing reasons, can supply a seed
        if seed:
            np.random.seed(seed)

        init_xyzuvw = np.random.multivariate_normal(
            mean=component.mean, cov=component.covariance,
            size=starcount,
        )

        return init_xyzuvw

    def generate_all_cartesian(self) -> NDArray[float64]:
        all_final_cartesian = []
        for ix, comp in enumerate(self.components):
            init_cartesian = self.generate_init_cartesian(
                comp, self.starcounts[ix], component_name=str(ix)
            )
            all_final_cartesian.append(comp.trace_orbit_func(init_cartesian, comp.age))
        all_cartesian_arr = np.vstack(all_final_cartesian)

        return all_cartesian_arr

    @classmethod
    def measure_astrometry(cls, cartesian):
        """
        Convert current day cartesian phase-space coordinates into astrometry
        values, with incorporated measurement uncertainty.
        """
        print(cls.m_err)
        errors = cls.m_err * np.array([
            cls.GERROR[colname + '_error']
            for colname in cls.DEFAULT_ASTR_COLNAMES
        ])

        # Get perfect astrometry
        astr = coordinate.convert_many_lsrxyzuvw2astrometry(cartesian)

        # Measurement errors are applied homogenously across data so we
        # can just tile to produce uncertainty
        # TODO: Add some variation in amount of error per datum
        nstars = len(astr)
        raw_errors = np.tile(errors, (nstars, 1))

        # Generate and apply a set of offsets from a 1D Gaussian with std
        # equal to the measurement error for each value
        offsets = raw_errors * np.random.randn(*raw_errors.shape)
        astr_w_offsets = astr + offsets
        # insert into Table
        temp = Table(names=cls.DEFAULT_NAMES, dtype=cls.DEFAULT_DTYPES)
        table = Table(
            names=cls.DEFAULT_NAMES,
            dtype=cls.DEFAULT_DTYPES,
            data=np.zeros((len(astr_w_offsets), 13), dtype=temp.dtype),
        )

        table['source_id'] = np.arange(len(astr_w_offsets))

        for ix, astr_name in enumerate(cls.DEFAULT_ASTR_COLNAMES):
            table[astr_name] = astr_w_offsets[:, ix]
            table[astr_name + '_error'] = raw_errors[:, ix]
        return table

    def synthesise_everything(self, savedir=None, filename=None, overwrite=False):
        """
        Uses self.pars and self.starcounts to generate an astropy table with
        synthetic stellar measurements.
        """
        cartesian = self.generate_all_cartesian()

        table = self.measure_astrometry(cartesian)
        return table
