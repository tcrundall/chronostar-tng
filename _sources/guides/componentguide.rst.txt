===============
Component Guide
===============

A Component object models a simple distribution, typically a mutli-variate normal (i.e. Gaussian) distribution. A Component class encapsulates the parameters that define the distribution and the methods that determine the best parameters given input data.

For example, a :class:`~chronostar.component.spherespacetimecomponent.SphereSpaceTimeComponent` is defined by an age, birth mean and birth covariance matrix, parameterised by a single array `parameters` (as used in Paper 1). It also has methods for calculating the log probability of stars :func:`~chronostar.component.spherespacetimecomponent.SphereSpaceTimeComponent.estimate_log_prob` and estimating its parameters :func:`~SpaceTimeComponent.maximize`. For more details, see the API.

.. note::

  The intention is to not differentiate between *Background* components and *stellar association* components. The difference will be implicit in the components' parameterization and fitting method. For example, an association component could have an age parameter, where as a background component would have none. Some other methods could be employed such as enforcing a large minimum position spread for background components and a small minimum position spread for association components. If one wishes to get fancy, one could choose to implement a :class:`BaseIntroducer` which can convert background components to association components and vica verse, based on some "failure" criteria.


Example SphereSpaceTimeComponent Usage
--------------------------------------

A simple usage will look something like this::

  resp = # ... assume we have responsibilities (i.e. Z or membership probabilites)
  X = # ... assume we have data, array-like with shape (n_samples, n_features)
  config_params = # ... assume we have Component specific config params as a dict

  SphereSpaceTimeComponent.configure(config_params)

  c = SphereSpaceTimeComponent()
  c.maximize(X, resp) 
  log_probs = c.estimate_log_prob(X)

  # If you have a specific parameters for the component (i.e. taken from a paper)
  # you can set the parameters yourself, but note that SphereSpaceTimeComponent
  paper_mean = np.array([])
  params = np.array([...])  # parameterise as best you can


If you desire a specific set of parameters, e.g. because you would like to match a component to that provided in paper, you can initialise a component by parameters. Note that this is non-trivial, since the components are typically parameterised by their *birth* distribution, and projecting arbitrary covariance matrices backwards through time rarely leads to a contracting distribution in space. This is how I would propose to do it::

  # Define current day mean and covariance
  paper_mean = np.array([...])
  paper_covariance = np.array([[...]])
  paper_age = ...

  birth_mean = SphereSpaceTimeComponent.trace_orbit_func(paper_mean, -paper_age)

  # For birth velocity spread, estimate current day spherical velocity spread
  # and assume minimal change
  birth_duvw = np.sqrt(np.mean(np.linalg.eigvals(paper_covariance[3:, 3:])))

  # For birth position spread, estimate current day spherical position spread
  # and assume linear expansion
  paper_dxyz = np.sqrt(np.mean(np.linalg.eigvals(paper_covariance[:3, :3])))
  birth_dxyz = paper_xyz - age * birth_duvw
  assert birth_dxyz > 0.

  params = np.hstack((birth_mean, birth_dxyz, birth_duvw, paper_age))

  # Configuring is likely unnecessary, since we're not fitting to data
  # SphereSpaceTimeComponent.configure(config_params)
  
  paper_comp = SphereSpaceTimeComponent(params)
  log_probs = paper_comp.estimate_log_prob(X)


Deriving New Component Classes
------------------------------
The hope is that implementing a new component class is relatively straightforward.

A new component class must inherit from :class:`~chronostar.base.BaseComponent`. If you have an IDE (e.g. Visual Studio Code) and are using type hints, then the IDE will prompt you into completing all the necessary aspects.

:class:`~chronostar.base.BaseComponent` has three abstract methods and one property you must implement:

- :func:`~chronostar.base.BaseComponent.estimate_log_prob`: given your component's current distribution, estimate the log probability of a data point. i.e. if the distribution is a normalised PDF, evaluate it at the location of the data point, and take the log of the result.
- :func:`~chronostar.base.BaseComponent.maximize`: given input data, determine the best values for the component's parameters
- :func:`~chronostar.base.BaseComponent.split`: split the current component into two similar, overlapping components that when combined more or less describe the original. This is the key mechanism used to by :class:`~chronostar.introducer.simpleintroducer.SimpleIntroducer` to introduce new components.
- :func:`~chronostar.base.BaseComponent.n_params`: define how many parameters are needed to parameterise the component. Calculations of BIC and AIC need to know this.

Check the API for the required function signatures and return types.
