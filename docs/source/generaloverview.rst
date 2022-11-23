================
General Overview
================

Chronostar: The Next Generation
-------------------------------

.. toctree::
   :maxdepth: 1
   :caption: Guides:
   :glob:

   guides/componentguide
   guides/mixtureguide
   guides/icpoolguide
   guides/driverguide

Here we provide a deep dive into the various parts of Chronostar, detailing how they work and hopefully providing enough guidance such that users can implement their own custom classes in order to augment Chronostar's functionality.

The core framework of Chronostar TNG consists of 4 classes. A :doc:`Driver <guides/driverguide>`, an :doc:`Initial Conditions Pool <guides/icpoolguide>`, a :doc:`Mixture <guides/mixtureguide>` and a :doc:`Component <guides/componentguide>`.

.. image:: images/simple_snapshot.svg
  :width: 800
  :alt: A graphical representation of how the classes connect.

Here we detail each element in a top-down approach. If you prefer a bottom-up approach,
feel free to read this page backwards.

Summary of Driver
^^^^^^^^^^^^^^^^^
The :doc:`Driver <guides/driverguide>` is the top level manager of the entire process. It parses the config file, instantiates classes where appropriate, passes on the initialization parameters as necessary, and then runs the primary loop.
In a multiprocessor implementation, the Driver could also manage the MPI Pool, providing each worker with a mixture model to maximize.

Driver-ICPool interface
^^^^^^^^^^^^^^^^^^^^^^^^^
The :doc:`Initial Conditions Pool <guides/icpoolguide>` or *ICPool* serves as a queue of initial conditions (the name will be updated in the next code refactor). An ICPool generates set after set of plausible :class:`~chronostar.base.InitialCondition`\ s. An ``InitialCondition`` is a labelled tuple of :doc:`Components <guides/componentguide>`. For each set of initial conditions the ``Driver`` initializes a :doc:`Mixture Model <guides/mixtureguide>` (Gaussian or otherwise). The Mixture Model fits itself to the data by maximizing the parameters of its list of `Components`. Once fit the Mixture Model calculates its final score (AIC, BIC, etc.). The ``Driver`` registers each fit to the ``ICPool`` along with its score. The primary loop repeats and the Driver acquires the next set of initial conditions. When the ``ICPool`` runs out of plausible initial conditions, the primary loop ends and the ``Driver`` returns the best fit.

.. note::
  The interaction between Driver and ICPool lends itself well to parallelisation, an aspect sorely lacking in the original Chronostar. The number of processes that can fit a mixture model in parallel is only capped by the number of sets of initial conditions sitting in the ICPool.

ICPool internal
^^^^^^^^^^^^^^^
Lets explore how the ``ICPool`` maintains this queue of initial conditions and determines when to trigger an end to the primary loop. The ``ICPool`` builds up a *registry* of previous fits and their scores as registered by the ``Driver``. The ``ICPool`` has an internal queue, to which it adds the next ``InitialCondition``\ s in batches called *generations*. The precise mechanism by which the next generation of ``InitialCondition``\ s is generated is unique to each ``ICPool``.

..
  One simple approach is the :class:`~chronostar.icpool.simpliecpool.SimpleICPool`. This implementation produces the next generation of ``InitialCondition``\ s by
  taking the current best fit, and generating one ``InitialCondition`` per component. The ``InitialCondition`` generated for the ``i``\ th component will split that
  component into two overlapping ones.

Summary of Mixture
^^^^^^^^^^^^^^^^^^
Lets focus now on how a Mixture Model fits itself to the data. We acknowledge the expert craftspersonship of `sklearn.mixture.GaussianMixture <https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture>`_ and so our Mixture Models follow their interface closely, with our example implementation even inheriting from `sklearn.mixture.BaseMixture <https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator>`_. Our Mixture Model expects the inputdata to be array-like of floats. The Mixture Model runs an Expectation-Maximization (EM) algorithm on the data and its list of Components. The interesting parts of the EM algorithm (how the various features in the data is interpreted, how the Component parameters are maximized) are delegated to the Components themselves. The Mixture Model handles only the boring, tedious, delicate business of membership probabilities and EM convergence. It is unlikely the user will need to write their own Mixture Model.

Summary of Component
^^^^^^^^^^^^^^^^^^^^
The Component class is where the most variation will likely appear. The Component implementation determines both what features we're fitting to, and how we find the best parameters. For example, :class:`~chronostar.component.spherespacetimecomponent.SphereSpaceTimeComponent` fits to the cartesian positions and velocities of the stars and
fits a spherical 6D Gaussian which is transformed through the galactic potential via an epicyclic approximation.
However many alternative implementations are viable.
The features are determined by your input data and could include position, velocity, age approximators, chemical composition etc.
The method of finding the best parameters could involve ``emcee``, ``scipy.optimize``. You can customize what distribution your component takes, for example a 6D Gaussian with or without an incorporated age projection.

As long as the data has the columns your Component expects, your Component can do whatever it likes, completely independent from the rest of Chronostar.
