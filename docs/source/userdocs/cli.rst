.. role:: bash(code)
   :language: bash

.. role:: py(code)
   :language: python

.. _cli:

Command-Line Interface
----------------------
Chronostar provides three command line tools for analyzing data: :code:`fit-component`, :code:`fit-mixture`
and :code:`fit-chronostar`. Each of these tools serve as entry points to different
levels of complexity.

There are also two helper tools :code:`prepare-data` and :code:`plot-features`.
:code:`prepare-data` assists users in converting fits tables from Gaia into the
appropriate format expected by Chronostar. :code:`plot-features` is a convenience
tool that plots various 2D projections of the data's feature space, as well as
colour magnitude diagrams.

If you'd prefer to stay in python, sample scripts are explained in
:doc:`Scripts <scripts>`, or you may peruse the pre-written provided scripts
on `github <https://github.com/tcrundall/chronostar-tng/tree/main/bin>`_.

Each command  assumes you have a data as numpy array of shape
:code:`(n_samples, n_features)` stored as a :code:`.npy` file.

Data Input
^^^^^^^^^^
Chronostar's :class:`Driver` accepts data only as a single 2D array of shape
`(n_samples, n_features)`. For example, for the current default implementation,
in the absence of uncertainties, the input is a `(n_stars, 6)` array where the
6 columns are `XYZUVW`.

If you wish to provide uncertainties in the form of a covariance matrix, the
matrices must be flattened and appended to the data array:

.. code::

   input_data = np.vstack((means.T, covariances.flatten().T)).T

However, if you're using the command line tool :ref:`fit-chronostar<cli_chron>`
this is handled for you, you need only provide the path to a stored
:code:`(n_stars, 6, 6)` numpy array of covariance matrices.

.. _dataprep:

Converting to Cartesian Coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The default implementation of Chronostar-TNG fits to data in a right handed
6D cartesian coordinate system, centred on the local standard of rest.

Since stellar data is typically provided as observables, we provide a command
line tool for converting fits files of observables into numpy arrays of cartesian
values.

Currently table column names are expected to be in the default form provided by 
Gaia. First ensure your column names match that of Gaia:

.. code::

   ['source_id', 'ra', 'ra_error', 'dec', 'dec_error', 'parallax', 'parallax_error',
   'pmra', 'pmra_error', 'pmdec', 'pmdec_error', 'ra_dec_corr', 'ra_parallax_corr', ...
    'radial_velocity', 'radial_velocity_error']

Then run

.. code::

   prepare-data path/to/data.fits

This will try to convert each star (skipping those that have non-finite radial velocities)
from astrometry to cartesian. Upon completion you will have 3 files in the working directory.
The cartesian means will be :code:`data_means.npy`, the uncertainty covariance matrices will be
:code:`data_covs.npy`, and the :code:`source_id` will be in :code:`ids.npy`.

By default, you'll also get a :code:`data_all.npy` file, which has the covariances
flattened and appended to the data. If you don't want this file, you can skip it by calling

.. code::
   
   prepare-data --nomerge path/to/data.fits

.. note::

   The `ids` array is useful for identifying which rows of the table constitute your
   data sample: :code:`subset_table = t[np.where(np.isin(t['source_id'], ids))]`

.. _cli_comp:

Fitting a component
^^^^^^^^^^^^^^^^^^^
This is for fitting a single component (e.g. a 6-D Gaussian) to the data.

.. code::

   $ fit-component path/to/data.npy

One can optionally include membership probabilities. The membership
probabilities are expected as a numpy array of shape :code:`(n_samples)`
and stored in a :code:`.npy` file.

.. code::

   $ fit-component path/to/data.npy path/to/memberships.npy

The behaviour can be configured by providing a config file:

.. code::

   $ fit-component -c path/to/config.yml path/to/data.npy

An empty config file is valid. An example :code:`config.yml` file is:

.. code::

   # define which classes to import
   modules:
      # A 6D Gaussian with age and spherical birth covariance
      component: "SphereSpaceTimeComponent"

   # any configuration parameters specific to the component class
   component:
      minimize_method: "Nelder-Mead"  # The scipy.optimize method
      reg_covar: 1.e-6                # Covariance matrix regularization constant
      nthreads: 4                     # How many numba threads to use for overlap integral calc
      stellar_uncertainties: True     # If the final 36 columns of data file are covariances

   # Specifics for the actual run
   run:
      savedir: "output"         # save directory for the results


.. note::

   Make sure to include the decimal point when using scientific
   notation, or :code:`yaml` will treat the value as a string.

.. _cli_mix:

Fitting a mixture
^^^^^^^^^^^^^^^^^
This is for fitting a fixed number of components to some data.

For example, to fit 5 components one would do:

.. code::

   $ fit-mixture 5 path/to/data.npy

Similarly, one can provide a config file:

.. code::

   $ fit-mixture -c /path/to/config.yml 5 path/to/data.npy

One can also provide a membership probability file. Make sure the dimensions
match (n_stars, n_components)

.. code:: bash

   $ fit-mixture -c path/to/config.yml 5 path/to/data.npy path/to/membprobs.npy

Since a mixture model utilises components, one can provide component
configuration parameters along with those for the mixture:

.. code::

   modules:
      component: "SpaceComponent"      # 6D Gaussian with no age
      mixture: "ComponentMixture"      # Default mixture (currently no alterantives anyway)

   mixture:
      max_iter: 100     # Max number of EM iterations
      tol: 1e-4         # Tolerance for convergence

      # How a fresh mixture is initialized:
      #  - 'init_resp': use input membership probabilities
      #  - 'random': memberships are initialized randomly
      #  - 'kmeans': memberships are initialized using kmeans
      #  - 'k-means++': use the k-means++ method to initialize
      init_params: 'random'
      # Get SKLearn to print messages. 0 - nothing, 1 - a little, 2 - a lot
      verbose: 1
      # how many iterations to wait between SKLearn print messages
      verbose_interval: 10

   component:
      reg_covar: 1.e-5
      minimize_method: 'Nelder-Mead'
      trace_orbit_func: 'epicyclic'
      age_offset_interval: 20         # After how many M-steps offsets for a component's age are tried

   run:
      savedir: "result"

.. _cli_chron:

Finding the best mixture
^^^^^^^^^^^^^^^^^^^^^^^^
This is full Chronostar.
Chronostar begins with fitting a single component to the
data, then progressively introduces more components, fitting
more complex mixtures, until extra components cease improving
the fit.

.. code::

   $ fit-chronostar -c path/to/config.yml path/to/data.npy
      or
   $ fit-chronostar -c path/to/config.yml path/to/means.npy --covs path/to/covs.npy

An example config file is:

.. code::

   modules:
      component: "SpaceComponent"      # 6D Gaussian with no age
      mixture: "ComponentMixture"      # Default mixture (currently no alterantives anyway)
      introducer: "SimpleIntroducer"   # Determines how components are introduced into future fits
      icpool: "SimpleICPool"           # Manages a pool of initial conditions of arbitrary number of comps

   mixture:
      max_iter: 100     # Max number of EM iterations
      tol: 1e-4         # Tolerance for convergence

   component:
      reg_covar: 1.e-5
      # Unnecessary parameters will be ignored, e.g. the following two
      # parameters are for SphereSpaceTimeComponent, SpaceComponent will
      # print a warning, then continue
      minimize_method: 'Nelder-Mead'
      trace_orbit_func: 'epicyclic'
   
   # introducer:     # A title may be missing

   icpool: {}        # But a title cannot point to nothing. An empty dictionary is allowed.

   run:
      savedir: "result"

Plotting
^^^^^^^^
A CLI tool for plotting is provided. It has two key functions so far. One is to plot features against features. Another is to plot CMDs. In both instances points are coloured by membership.

.. _feat:

Features
~~~~~~~~
Here is an example of plotting 6 phase-space planes ('XY, XZ, YZ, XU, YV, ZW') and saving the plot in a directory `plots`.

.. code::

   plot-features -f '0,1.0,2.1,2.0,3.1,4.2,5' -m path/to/data.npy -z path/to/membership_probs.npy -o plots

Each phase-space subplot is separated in the command by a period, i.e.
:code:`plot1-xaxis,plot1-yaxis.plot2-xaxis,plot2-yaxis` etc. You may add as many
phase-space pairs as you like, and they will be arranged top to bottom, left to right,
with as close to a square layout as possible.

.. _cmd:

CMD
~~~
Here is an example of plotting a CMD. Since the fits file likely featured rows with incomplete data, there will likely not be a one to one mapping from the membership probability table to the astrometry table. Hence `source_ids.npy` is used. `source_ids.npy` should be of shape `(n_stars)`
and has the gaia source id of each star in `membership_probs.npy`.

.. code::

   plot-features --photom -d path/to/gaia/data.fits -z path/to/membership_probs.npy -s path/to/source_ids.npy

