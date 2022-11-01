.. _settings:

Configuration Settings
======================

Chronostar has many configurable settings. You can configure them via a config
file or by calling each class's :code:`configure` class (static) method.

See :ref:`Command-Line Interface<cli>` for example config files.

Here we provide an exhaustive list of all configurable settings and their uses.

Components
__________

:class:`~chronostar.component.spacecomponent.SpaceComponent`
------------------------------------------------------------

reg_covar : float, default 1.e-6
   A regularisation constant added to the diagonals
   of the covariance matrix

:class:`~chronostar.component.spherespacetimecomponent.SphereSpaceTimeComponent`
--------------------------------------------------------------------------------

reg_covar : float, default 1.e-6
   A regularisation constant added to the diagonals
   of the covariance matrix
minimize_method: str, default 'Nelder-Mead'
   The method used by ``scipy.optimize.minimize``. Must be one of

   - 'Nelder-Mead' (recommended)
   - 'Powell' (not receommended)

nthreads : int, optional
   Manually restrict how many threads openMP tries to use when
   executing optimized numpy functions
trace_orbit_func: Callable: f(start_loc, time), default :func:`trace_epicyclic_orbit`
   A function that traces a position by `time` Myr. Positive `time`
   traces forward, negative `time` backwards
age_offset_interval: int, default 20
   After how many calls to :func:`maximize` age offsets should be explored
stellar_uncertainties: bool, default True
   Whether data covariance matrices are encoded in final 36 columns of
   input data X

Mixtures
________

.. _config_mixture:

:class:`~chronostar.mixture.componentmixture.ComponentMixture`
--------------------------------------------------------------

tol : float, default 1e-3
    Used to determine convergence by sklearn's EM algorithm.
    Convergence determined if "change" between EM iterations is
    less than tol, where change is the difference between the
    average log probability of each sample
reg_covar : float, default 1e-6
    A regularization constant added to the diagonals of
    covariance matrices
max_iter : int, default 100
    The maximum iterations for sklearn's EM algorithm
n_init : int, default 1
    (included only for sklearn API compatbility, ignored)
init_params : str, default 'random'
    The initialization approach used by sklearn if component
    parameters aren't pre set. Must be one of

    - 'init_resp' : responsibilites are taken from input
    - 'kmeans' : responsibilities are initialized using kmeans.
    - 'k-means++' : use the k-means++ method to initialize.
    - 'random' : responsibilities are initialized randomly.
    - 'random_from_data' : initial means are randomly selected data points.

random_state: int, default None
    Controls the random seed given to the method chosen to
    initialize the parameters (see init_params). In addition, it
    controls the generation of random samples from the fitted
    distribution. Pass an int for reproducible output across multiple
    function calls.
warm_start: bool, default True
    (leave True for correct interactions between `self` and
    `self.sklmixture`)
verbose: int, default 0
    Whether to print sklearn statements:

    - 0 : no output
    - 1 : prints current initialization and each iteration step
    - 2 : same as 1 but also prints log probability and execution time

verbose_interval: int, default 10
    If `verbose > 0`, how many iterations between print statements

Introducers
___________

:class:`~chronostar.introducer.simpleintroducer.SimpleIntroducer`
-----------------------------------------------------------------

None...

ICPools
_______

:class:`~chronostar.icpool.simpleicpool.SimpleICPool`
-----------------------------------------------------

max_components : int, default 100
    The max components in an initial condition provided by
    `SimpleICPool`

Drivers
_______

:class:`~chronostar.driver.Driver`
----------------------------------

intermediate_dumps : bool, default True
    Whether to write to file the results of mixture model fits
savedir : str, default './result'
    Path to the directory of where to store results

Runs
____

The three command line tools for fitting
(
:ref:`fit-chronostar<cli_chron>`,
:ref:`fit-mixture<cli_mix>`,
and
:ref:`fit-component<cli_comp>`
)
have "run level" parameters.

nthreads : int, default 1
    Provided for future high-level parallelism. Currently nothing is implemented, so
    leave this at 1
savedir : str, default "output"
    The output directory for the final results. This can be the same directory that 
    :ref:`ComponentMixture<config_mixture>` uses to store intermediate dumps.
