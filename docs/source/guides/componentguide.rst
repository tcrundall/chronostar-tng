===============
Component Guide
===============

A Component object models a simple distribution, typically a mutli-variate normal (i.e. Gaussian) distribution. A Component class encapsulates the parameters that define the distribution, the methods that determine the best parameters given input data and how the input data is interpreted.

For example, a :class:`~chronostar.component.spherespacetimecomponent.SphereSpaceTimeComponent` is defined by an age, birth mean and birth covariance matrix, parameterised by a single array `parameters` (as used in Paper 1). It also has methods for calculating the log probability of stars :func:`~chronostar.component.spherespacetimecomponent.SphereSpaceTimeComponent.estimate_log_prob` and estimating its parameters :func:`~SpaceTimeComponent.maximize`. For more details, see the API. It also is able to reconstruct data covariance matrices from the input
data rows.

Components can accept a list of membership probabilities, thereby the contribution
of any datapoint is weighted accordingly during fits.

For example usages see :ref:`Fitting a component <scripts-comp>`.

Implemented Components
----------------------

.. _guide-spacecomp:

:class:`~chronostar.component.spacecomponent.SpaceComponent`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The space component fits a free 6D Gaussian to a set of 6D data points.
This component cannot handle uncertainties and ignores any data beyond the
first 6 columns.

This component class is useful for fitting to data where either uncertainties are not
significant (e.g. when characterising the background) or for getting a first order
characterisation of the environment of an association, the results of which
may be used to initialise a more nuanced fit, thereby skipping a lot of computation.

This component is lightning quick to fit, because it doesn't perform any parameter
exploration. It calculates its mean and covariance using simple numpy functions.

The parameters array of a SpaceComponent is ``42`` elements long, the first ``6``
being the mean, the next ``36`` are the covariance matrix, flattened, i.e.::

  parameters = np.hstack((mean, covariance.flatten()))


.. _guide-spherecomp:

:class:`~chronostar.component.spherespacetimecomponent.SphereSpaceTimeComponent`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The SphereSpaceTimeComponent has an age (hence time) and assumes the
distribution of an association at birth to be spherical. The birth distribution
is therefore a 6D Gaussian, spherical in position and spherical in velocity.
The current day distribution (which determines the actual log probabilities of the data)
is the birth distribution projected forward through the galactic potential by *age* Myr.

The parameters array
parameterise the birth mean (first ``6`` elements), the birth covariance (the next ``2``) and the age. A standard deviation in position ``dxyz`` and in velocity ``duvw`` parameterise the birth
covariance matrix. Therefore the parameters array looks like::

  parameters = np.array([x, y, z, u, v, w, dxyz, duvw, age])

The component calculates its current day mean using the epicyclic approximation
:func:`~chronostar.traceorbit.trace_epicyclic_orbit`, and transforms the birth covariance
to the current day covariance using a jacobian matrix :func:`~chronostar.utils.transform.transform_covmatrix`.

Implementing Custom Components
------------------------------
All component classes derive from :class:`~chronostar.base.BaseComponent`.
Any custom derived component classes should also derive from this base class.
The base class has some abstract methods and properties that your new class must implement:

- :func:`~chronostar.base.BaseComponent.maximize`: given input data, determine the best values for the component's parameters
- :func:`~chronostar.base.BaseComponent.estimate_log_prob`: given your component's current distribution, estimate the log probability of a data point. i.e. if the distribution is a normalised PDF, evaluate it at the location of the data point, and take the log of the result.
- :func:`~chronostar.base.BaseComponent.split`: split the current component into two similar, overlapping components that when combined more or less describe the original. This is the key mechanism used to by :class:`~chronostar.introducer.simpleintroducer.SimpleIntroducer` to introduce new components.
- :func:`~chronostar.base.BaseComponent.set_parameters` - set the parameters
- :func:`~chronostar.base.BaseComponent.get_parameters` - get the parameters
- :func:`~chronostar.base.BaseComponent.n_params` - the number of parameters used. This is not necessarily the same as ``len(self.get_parameters())``, for example :ref:`SpaceComponent <guide-spacecomp>` has duplicates of the covariance values in its parameter array. Calculations of BIC and AIC need to know this.

You may find it convenient to define further properties that convert the raw
parameter array into the component aspects. For example, :ref:`SphereSpaceTimeComponent <guide-spherecomp>` has as properties ``mean``, ``covariance`` and ``age``.
These properties should only depend on the component's ``parameters``. Defining
them as properties (and not attributes) guarantees that the state of the component
is defined in one place only, the ``parameters`` and removes the temptation of the
user to attempt to modfiy the current day mean and covariance.

.. note::

  The ``maximize()`` method should never make use of these properties, for two reasons.
  Firstly it doesn't make algorithmic sense because it is trying to find the best values for ``parameters`` and update them, not evaluate how good the current values are.
  Secondly, there would be potential performance issues by redoing a potentially expensive
  operation during the many iterations.
