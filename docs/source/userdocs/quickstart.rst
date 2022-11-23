.. role:: bash(code)
   :language: bash

.. role:: py(code)
   :language: python

Quickstart
==========

Preparing a run
---------------

Data prep
^^^^^^^^^
The first step is to prepare your data.

Prepare your stellar cartesian data as two stored numpy arrays of measurements
(means) with dimension (``n_stars, 6``) and uncertainties (stored as covariance matrices) with dimension (``n_stars, 6, 6``). Alternatively, means and uncertainties can be folded into the same array. See
:ref:`Converting to Cartesian Coordinates<dataprep>` for more details.

.. note::

   If your data represents a subset of a Gaia astrometry fits file, we recommend storing the corresponding ``source_id`` of the stars in the cartesian data set e.g. ``subset_sourceids.npy``. This can be used to extract the corresponding photometric data for plotting with the results.

Configuration
^^^^^^^^^^^^^
Each class in Chronostar has a set of configurable parameters. Chronostar
reads them in via a configuration file.

Prepare a configuration file e.g. ``config.yml`` with contents:

.. code-block::

   module:
      component: "SphereSpaceTimeComponent"
   
   mixture:
      max_iter: 100
      tol: 1e-4
      verbose: 2
      verbose_interval: 1

   component:
      stellar_uncertainties: True
      nthreads: 4    # or however many cores you have at disposal
   
   driver:
      intermediate_dumps: True
      savedir: "results/intermediate"

   run:
      savedir: "results/final"

See :ref:`Command-Line Interface<cli>` for details about other command-line tools and :ref:`Configuration Settings<settings>` for a complete list of configurable parameters.

Performing a run
----------------
Now run:

.. code-block::

   $ fit-chronostar -c config.yml path/to/means.npy --covs path/to/covs.npy

Chronostar will perform multiple Gaussian Mixture model fits to the data, with differing initial conditions and ever increasing number of components. Chronostar determines convergence by comparing `BICs <https://en.wikipedia.org/wiki/Bayesian_information_criterion>`_. Chronostar's :class:`Driver` stores the results of each fitted mixture model in the provided ``savedir``, which is given above as ``results_intermediate``. The Driver stores each fit's result in an informatively named subdirectory. See :ref:`Finding the best mixture<cli_chron>` for details on how to interpret the names of these subdirectories.

.. note::

   Make sure to add these details

Depending on the size of the data, this may take up several days to converge, so it is suggested to run this with ``nohup``, e.g.:

.. code-block::

   $ nohup fit-chronostar -c config.yml path/to/means.npy --covs path/to/covs.npy > chron-run.log &

If all goes well, you'll have the following files in ``./results/final``:

.. code-block::

   # A text file summarising the results, where 'XXX' is the number of components
   final-fit-XXXcomps.txt

   # membership probabilities of each star to each component
   membership.npy

   # The weights of each component (normalised to all sum to one)
   weights.npy

   # Files for each component with best fitting parameters
   comp_000_params.npy
   comp_001_params.npy
   ...

Plotting
--------

CMDs
^^^^
You can plot some results using the membership probabilities. To plot the astrometry (using Gaia's ``phot_g_mean_mag`` and ``g_rp``):

.. code-block::

   plot-features --photom -d path/to/gaia.fits -z result_final/membership.npy -s path/to/subset_sourceids.npy

This plot automatically colours each star by component membership, and sizes each point by it's component's weight, thereby making smaller components more visible. See :ref:`CMD<cmd>` for more info.

.. note::

   This plot tries to be a "one size fits all" tool, which utlimately doesn't fit any. Users are encouraged to develop their own plotting tools based on
   `those provided <https://github.com/tcrundall/chronostar-tng/tree/main/bin>`_.

Cartesian Space
^^^^^^^^^^^^^^^
To plot stars in cartesian phase-space:

.. code-block::

   plot-features -f 0,1.0,3.1,4.2,5 -m path/to/means.npy -z result_final/membership.npy

The argument following ``-f`` represents each phase-space subplot. For example here, we will plot X against Y, X against U, Y against V and Z against W. See :ref:`Features<feat>` for more info.

.. note::

   For large data sets ( > 1,000) plotting may take multiple minutes.
