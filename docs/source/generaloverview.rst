================
General Overview
================

Chronostar: The Next Generation
-------------------------------

.. toctree::
   :maxdepth: 1
   :caption: Guides:
   :glob:

   guides/*

.. note::

    All the guides are slightly out of date with the code. The general approach and algorithms are correct, but the specific module and class names may slightly differ.

The next generation of Chronostar prioritises simplicity, flexibility and extensibility. The default behaviour addresses the simplest scenario but the means of extending the capabilties for more complex scenarios is straight forward; by default Chronostar-TNG (will) provide abstract base classes which clearly dictate required interfaces as well as many different example reifications for the most basic scenarios. How one actually extends these example implementations to address complex scenarios is left to the user. ;)

The framework of Chronostar TNG consists of 5 classes. A :doc:`Driver <guides/driverguide>`, a :doc:`Initial Conditions Pool <guides/icpoolguide>`, an :doc:`Introducer <guides/introducerguide>`, a :doc:`Mixture <guides/mixtureguide>` and a :doc:`Component <guides/componentguide>`.

.. image:: images/simple_snapshot.svg
  :width: 800
  :alt: A graphical representation of how the classes connect.


The goal of Chronostar is to be a flexible framework for fitting Gaussian
Mixture Models to astronomical data. This will be achieved by utilising
"injected dependencies". The :doc:`Driver <guides/driverguide>` is composed of a collection
of 4 classes (:doc:`ICPool <guides/icpoolguide>`, :doc:`Introducer <guides/introducerguide>`, :doc:`Mixture <guides/mixtureguide>` and :doc:`Component <guides/componentguide>`) for which
Chronostar provides many implementations. Anyone wishing to modify 
aspects of Chronostar (e.g. input data, fitting method, models of 
components) simply needs to provide the :doc:`driver <guides/driverguide>` with their own
class that matches the required interface, whether that be by writing
the class from scratch, or by using inheritance to extend pre-existing
classes.

Here we provide a :doc:`general overview <generaloverview>` of the framework. 

Summary of Driver
^^^^^^^^^^^^^^^^^
The :doc:`Driver <guides/driverguide>` is the top level manager of the entire process. It parses the config file, instantiates classes where appropriate, passes on the initialization parameters as necessary, and then runs the primary loop.
The primary loop is only 4 lines long because the majority of the work is neatly delegated to the other classes. In a multiprocessor implementation, the Driver would also manage the MPI Pool.

Driver-ICPool interface
^^^^^^^^^^^^^^^^^^^^^^^^^
The :doc:`Initial Conditions Pool <guides/icpoolguide>` or *ICPool* serves as (perhaps surprisingly) a pool of initial conditions. An ICPool generates set after set of plausible initial conditions. An initial conditions set is a list of :doc:`Components <guides/componentguide>`. For each set of initial conditions the `Driver` initializes a :doc:`Mixture Model <guides/mixtureguide>` (Gaussian or otherwise). The Mixture Model fits itself to the data by maximizing the parameters of its list of `Components`. Once fit the Mixture Model calculates its final score (AIC, BIC, etc.). The `Driver` registers each fit to the `ICPool` along with its score. The primary loop repeats and the Driver acquires the next set of initial conditions. When the `ICPool` runs out of plausible initial conditions, the primary loop ends and the `Driver` returns the best fit.

.. note::
  The interaction between Driver and ICPool lends itself well to parallelisation, an aspect sorely lacking in the original Chronostar. The number of processes that can fit a mixture model in parallel is only capped by the number of sets of initial conditions sitting in the ICPool.

ICPool-Introducer interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Lets explore how the `ICPool` cultivates this pool of initial conditions and determines when to trigger an end to the primary loop. The `ICPool` builds up a *registry* of previous fits and their scores as registered by the `Driver`. The `ICPool` generates the next initial conditions based on this registry, perhaps by introducing one ore more extra components. The precise mechanism by which the `ICPool` introduces `Components` is determined by the :doc:`Introducer <guides/introducerguide>`. The `Introducer` takes one or more previous fits, and returns one or more plausible sets of initial conditions to the `ICPool`. The `ICPool` in turn *yields* each set to the `Driver`. By using the keyword *yield* we've turned `ICPool` (or more precisely one of its methods) into an iterable object which can be looped over. By inspecting the scores in the registry, the `ICPool` can determine when scores are conisitently failing to improve. At this point the `ICPool` stops yielding which triggers a ``Stop Iteration`` exception, thereby ending the primary loop.

Summary of Mixture
^^^^^^^^^^^^^^^^^^
Lets focus now on how a Mixture Model fits itself to the data. We acknowledge the expert craftspersonship of `sklearn.mixture.GaussianMixture <https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture>`_ and so our Mixture Models follow their interface closely, with our example implementation even inheriting from `sklearn.mixture.BaseMixture <https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator>`_. Our Mixture Model expects the inputdata to be array-like of floats. The Mixture Model runs an Expectation-Maximization (EM) algorithm on the data and its list of Components. The interesting parts of the EM algorithm (how the various features in the data is interpreted, how the Component parameters are maximized) are delegated to the Components themselves. The Mixture Model handles only the boring, tedious, delicate business of membership probabilities and EM convergence. It is unlikely the user will need to write their own Mixture Model.

Summary of Component
^^^^^^^^^^^^^^^^^^^^
The Component class is where the most variation will likely appear. The Component implementation determines what features we're fitting to, and how we find the best parameters. If you want age dependency, you got it. If you want ``emcee``, you got it. If you want Nelder-Mead, you got it. If you want a flat background component, you got it. If you want a background component that just reads off a column in the input data, you got it. If you want your stars to have uncertainties and correlations stored in the input data which you can use to constract star covariance matrices, you got it. If you want to propogate through time with galpy, epicyclic, or just a straight bloody line, you got it.

As long as the data has the columns your Component expects, your Component can do whatever it likes, completley independently from the Chronostar.

.. note::
  TODO: Finish this section...
