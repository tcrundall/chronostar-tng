.. role:: bash(code)
   :language: bash

.. role:: py(code)
   :language: python

Quickstart
----------
Chronostar provides three command line tools: :code:`fit-component`, :code:`fit-mixture`
and :code:`fit-chronostar`. Each of these tools serve as entry points to different
levels of complexity.

If you'd prefer to stay in python, sample scripts are explained in
:doc:`Scripts <scripts>`, or you may peruse the pre-written provided scripts
on `github <https://github.com/tcrundall/chronostar-tng/tree/main/bin>`_.

Each command  assumes you have a data as numpy array of shape
:code:`(n_samples, n_features)` stored as a :code:`.npy` file.

Data preparation
^^^^^^^^^^^^^^^^
Chronostar's front end expects data to be in a 2D array of shape
`(n_samples, n_features)`. For current implementations, this translates to
a `(n_stars, 6)` array where the 6 columns are `XYZUVW`.

A CLI tool is under development to assist users in this, but it isn't ready yet.

If you're using a subset of a Gaia fits file, we recommend saving an array of the source ids for the stars actually used,
as this will prove useful mapping Chronostar results back to the fits table and is explicitly used
by `plot-features`.

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

   # Specifics for the actual run
   run:
      savedir: "output"         # save directory for the results


.. note::

   Make sure to include the decimal point when using scientific
   notation, or :code:`yaml` will treat the value as a string.


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

   module:
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

   run:
      savedir: "result"

Finding the best mixture
^^^^^^^^^^^^^^^^^^^^^^^^
This is full Chronostar.
Chronostar begins with fitting a single component to the
data, then progressively introduces more components, fitting
more complex mixtures, until extra components cease improving
the fit.

.. code::

   $ fit-chronostar -c path/to/config.yml path/to/data.npy

An example config file is:

.. code::

   module:
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

Features
++++++++
Here is an example of plotting 6 phase-space planes ('XY, XZ, YZ, XU, YV, ZW') and saving the plot in a directory `plots`.

.. code::

   plot-features -f '0,1.0,2.1,2.0,3.1,4.2,5' -m path/to/data.npy -z path/to/membership_probs.npy -o plots

CMD
+++
Here is an example of plotting a CMD. Since the fits file likely featured rows with incomplete data, there will likely not be a one to one mapping from the membership probability table to the astrometry table. Hence `source_ids.npy` is used. `source_ids.npy` should be of shape `(n_stars)`
and has the gaia source id of each star in `membership_probs.npy`.

.. code::

   plot-features --photom -d path/to/gaia/data.fits -z path/to/membership_probs.npy -s path/to/source_ids.npy

