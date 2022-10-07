.. chronostar-dev documentation master file, created by
   sphinx-quickstart on Sat Sep 17 16:22:57 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to chronostar-dev's documentation!
==========================================
**Chronostar** is a Python library for discovering and characterising
unbound stellar associations (aka "moving groups") and their members using 
features such as kinematics, age markers, chemical composition. The code 
can be found on `github <https://github.com/tcrundall/chronostar-tng/>`_.

Installing
----------

.. code::

   git clone https://github.com/tcrundall/chronostar-tng.git
   conda env create -n chron -f environment.yml    # set up py39 environment
   pip install .

Quickstart
----------
Chronostar provides three command line tools: `fit-component`, `fit-mixture`
and `fit-chronostar`. Each of these tools serve as entry points to different
levels of complexity.

Each command  assumes you have a data as numpy array of shape
`(n_samples, n_features)` stored as a `.npy` file.

Fitting a component
^^^^^^^^^^^^^^^^^^^
This is for fitting a single component (e.g. a 6-D Gaussian) to the data.

.. code::

   >> fit-component path/to/data.npy

One can optionally include membership probabilities. The membership
probabilities are expected as a numpy array of shape `(n_samples)`
and stored in a `.npy` file.

.. code::

   >> fit-component path/to/data.npy path/to/memberships.npy

The behaviour can be configured by providing a config file:

.. code::

   >> fit-component -c path/to/config.yml path/to/data.npy

An empty config file is valid. An example `config.yml` file is:

.. code::

   # define which classes to import
   modules:
      component: "SpaceTimeComponent"  # A 6D Gaussian with age

   # any configuration parameters specific to the component class
   component:
      minimize_method: "brent"  # The scipy.optimize method
      reg_covar: 1.e-5          # Covariance matrix regularization constant

   # Specifics for the actual run
   run:
      savedir: "output"         # save directory for the results


.. note::

   Make sure to include the decimal point when using scientific
   notation, or `yaml` will treat the value as a string.


Fitting a mixture
^^^^^^^^^^^^^^^^^
This is for fitting a fixed number of components to some data.

For example, to fit 5 components one would do:

.. code::

   >> fit-mixture 5 path/to/data.npy

Similarly, one can provide a config file:

.. code::

   >> fit-mixture -c /path/to/config.yml 5 path/to/data.npy

.. note::

   Currently there is no implementation for initialising a mixture
   with membership probabilities. This is on the todo list.

Since a mixture model utilises components, one can provide component
configuration parameters along with those for the mixture:

.. code::

   module:
      component: "SpaceComponent"      # 6D Gaussian with no age
      mixture: "ComponentMixture"      # Default mixture (currently no alterantives anyway)

   mixture:
      max_iter: 100     # Max number of EM iterations
      tol: 1e-3         # Tolerance for convergence

   component:
      reg_covar: 1.e-5
      minimize_method: 'golden'
      trace_orbit_func: 'epicyclic'
      morph_cov_func: 'elliptical'     # The assumptions placed on birth-site covariance

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

   fit-chronostar -c path/to/config.yml path/to/data.npy

An example config file is:

.. code::

   module:
      component: "SpaceComponent"      # 6D Gaussian with no age
      mixture: "ComponentMixture"      # Default mixture (currently no alterantives anyway)
      introducer: "SimpleIntroducer"   # Determines how components are introduced into future fits
      icpool: "SimpleICPool"           # Manages a pool of initial conditions of arbitrary number of comps

   mixture:
      max_iter: 100     # Max number of EM iterations
      tol: 1e-3         # Tolerance for convergence

   component:
      reg_covar: 1.e-5
      minimize_method: 'golden'
      trace_orbit_func: 'epicyclic'
      morph_cov_func: 'elliptical'
   
   # introducer:     # A title may be missing

   icpool: {}        # But a title cannot point to nothing. An empty dictionary is allowed.

   run:
      savedir: "result"

..
   Check out the :doc:`usage` section for further information, including how to :ref:`install <installation>` the project.

.. note::

   This project is under active development

.. toctree::
   :maxdepth: 1
   :caption: Contents:
   :glob:

..
   .. toctree::
      :maxdepth: 2
      :caption: Lost docs:
      :glob:

      uncertaintyfinder


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
