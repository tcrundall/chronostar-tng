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
        # if type(starcounts) is not np.ndarray:
        #     if type(starcounts) in (list, tuple):
        #         self.starcounts = np.array(starcounts, dtype=int)
        #     else:
        #         self.starcounts = np.array([starcounts], dtype=int)
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

        # self.background_density = background_density
        # self.bg_span_scale = bg_span_scale
        self.table = Table(names=self.DEFAULT_NAMES,
                           dtype=self.DEFAULT_DTYPES)

    # def extract_data_as_array(self, colnames=None, table=None):
    #     result = []
    #     if table is None:
    #         table = self.table
    #     for colname in colnames:
    #         result.append(np.array(table[colname]))
    #     return np.array(result).T

    # @staticmethod
    # def generate_synth_data_from_file():
    #     """Given saved files, generate a SynthData object"""
    #     raise NotImplementedError

    # def append_extra_cartesian(self, extra_xyzuvw, component_name='',
    #                            component_age=0., init=True):
    #     """
    #     Append rows of stellar initial (or final) positions to current table
    #     """
    #     # construct new table with same fields as self.astr_table,
    #     # then append to existing table
    #     init_size = len(self.table)
    #     starcount = len(extra_xyzuvw)

    #     names = np.arange(init_size, init_size + starcount).astype(str)
    #     new_data = Table(
    #             data=np.zeros(starcount, dtype=self.table.dtype)
    #     )

    #     new_data['name'] = names
    #     new_data['component'] = starcount * [component_name]
    #     new_data['age'] = starcount * [component_age]
    #     if init:
    #         colnames = [dim+'0' for dim in self.cart_labels]
    #     else:
    #         # Final data
    #         colnames = [dim+'_now' for dim in self.cart_labels]

    #     for col, colname in zip(extra_xyzuvw.T, colnames):
    #         new_data[colname] = col

    #     self.table = vstack((self.table, new_data))

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

        # # Append data to end of table
        # self.append_extra_cartesian(
        #     init_xyzuvw,
        #     component_name=component_name,
        #     component_age=component.age,
        #     init=True,
        # )

#     def generate_background_stars(self):
#         """
#         Embed association stars in a sea of background stars with
#         `self.bg_span_scale` * the span as current data
#         """

#         # Get the 6D means of all stars currently in table
#         means_now = tabletool.build_data_dict_from_table(
#                 self.table, main_colnames=[dim+'_now' for dim in self.cart_labels],
#                 only_means=True,
#         )
# #         init_means = tabletool.build_data_dict_from_table(
# #                 self.table, main_colnames=[el+'0' for el in 'xyzuvw'],
# #                 only_means=True,
# #         )

#         # Identify 6D box centred on component stars, with twice the span
#         # TC: Edited, to only have 10% margin on every side
#         data_upper_bound = np.max(means_now, axis=0)
#         data_lower_bound = np.min(means_now, axis=0)
#         box_centre = (data_upper_bound + data_lower_bound) / 2.

#         data_span = data_upper_bound - data_lower_bound
#         box_span = self.bg_span_scale * data_span
#         box_low = box_centre - 0.5*box_span
#         box_high = box_centre + 0.5*box_span

#         # Calculate number of background stars required to reach given density
#         bg_starcount = self.background_density * np.product(box_span)

#         # Generate a uniform sampling within box
#         # bg_xyzuvw = np.random.uniform(low=-data_span, high=data_span,
#         # TC: This was incorrect. Fixed it now.
#         bg_xyzuvw = np.random.uniform(low=box_low, high=box_high,
#                                            size=(int(round(bg_starcount)),6))
#         # bg_xyzuvw += box_centre
#         self.bg_starcount = bg_starcount
#         if bg_starcount > 0:
#             self.append_extra_cartesian(bg_xyzuvw, component_name='bg', init=False)
#         else:
#             print('WARNING: No synthetic background stars generated')

    def generate_all_cartesian(self) -> NDArray[float64]:
        # self.table = Table(names=self.DEFAULT_NAMES,
        #                    dtype=self.DEFAULT_DTYPES)
        all_final_cartesian = []
        for ix, comp in enumerate(self.components):
            init_cartesian = self.generate_init_cartesian(
                comp, self.starcounts[ix], component_name=str(ix)
            )
            all_final_cartesian.append(comp.trace_orbit_func(init_cartesian, comp.age))
        all_cartesian_arr = np.vstack(all_final_cartesian)

        return all_cartesian_arr

    # def project_stars(self):
    #     """Project stars from xyzuvw then to xyzuvw now based on their age"""
    #     if self.trace_orbit_func is None:
    #         raise UserWarning(
    #                 'Need to explicitly set trace orbit function '
    #                 'i.e. with mysynthdata.trace_orbit_func = trace_epicyclic_orbit')
    #     for star in self.table:
    #         mean_then = self.extract_data_as_array(
    #             table=star,
    #             colnames=[dim+'0' for dim in self.cart_labels],
    #         )
    #         xyzuvw_now = self.trace_orbit_func(mean_then, times=star['age'])
    #         for ix, dim in enumerate(self.cart_labels):
    #             star[dim+'_now'] = xyzuvw_now[ix]
    #     # if self.background_density is not None:
    #     #     self.generate_background_stars()

    @classmethod
    def measure_astrometry(cls, cartesian):
        """
        Convert current day cartesian phase-space coordinates into astrometry
        values, with incorporated measurement uncertainty.
        """
        # # Grab xyzuvw data in array form
        # xyzuvw_now_colnames = [dim + '_now' for dim in self.cart_labels]
        # xyzuvw_now = self.extract_data_as_array(colnames=xyzuvw_now_colnames)

        # Build array of measurement errors, based on Gaia DR2 and scaled by
        # `m_err`
        # [ra, dec, plx, pmra, pmdec, rv]
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

    # def store_table(self, savedir=None, filename=None, overwrite=False):
    #     """
    #     Store table on disk.
    #     Parameters
    #     ----------
    #     savedir : str {None}
    #         the directory to store table file in
    #     filename : str {None}
    #         what to call the file (can also just use this and provide whole
    #         path)
    #     overwrite : boolean {False}
    #         Whether to overwrite a table in the same location
    #     """
    #     if savedir is None:
    #         savedir = self.savedir
    #     # ensure singular trailing '/'
    #     if savedir != '':
    #         savedir = savedir.rstrip('/') + '/'
    #     if filename is None:
    #         filename = self.tablefilename
    #     self.table.write(savedir + filename, overwrite=overwrite)

    def synthesise_everything(self, savedir=None, filename=None, overwrite=False):
        """
        Uses self.pars and self.starcounts to generate an astropy table with
        synthetic stellar measurements.
        """
        cartesian = self.generate_all_cartesian()
        # if self.background_density is not None:
        #     self.generate_background_stars()

        table = self.measure_astrometry(cartesian)
        return table
        # if filename is not None:
        #     self.store_table(savedir=savedir, filename=filename,
        #                      overwrite=overwrite)
