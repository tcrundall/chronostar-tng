=============
Mixture Guide
=============

Chronostar's Mixture Models will match the interface of `scikit-learn's mixture models <https://scikit-learn.org/stable/modules/mixture.html#gmm>`_ as closely as possible, with the default mixture model :class:`ComponentMixture` even inheriting from `sklearn.mixture.BaseMixture <https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator>`_. For reference of a class that inherits from BaseMixture, see `sklearn.mixture.GaussianMixture <https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture>`_.

I predict there will only need to be one implementation of this class, as customisation of data features, component features, fitting approaches and overall model extensions can be done elsewhere (i.e. in :class:`Component` or :class:`BaseICPool`).

Our Mixture Models will be a linear combination of components, fit to an input data array of shape `(n_samples, n_features)` where a "sample" is a single star (or can be treated as such, e.g. a combined binary) and a feature is any data we're trying to fit to (position, velocity, magnitude, apparent-magnitude, abundances, inferred-age, etc.). The Mixture Model will remain completely agnostic of the features being fit. The implementation of how features are fit to is left to the :class:`Component` class.

This linear combination of components can include background components. In fact, this is the only (intended) means of including information about the background. The implementation of the background components is completely arbitrary. For example, it could be a single flat distribution, or a collection of fixed 6D Gaussians, or a collection of free 6D Gaussians that are fit alongside components intended for *stellar associations*.

Conceivably, if users wish to replicate Crundall et al. (2019)'s treatment of the background, one could precompute background log likelihoods, store them in the input data array, and have a background component that simply retrieves that column of data.

.. note::
  The precise interface will likely need to be tweaked as the first implementation is written, however I expect implementation specific configuration parameters will be passed through the chronostar framework as a dictionary read from a config file (see :doc:`Driver Guide <driverguide>`) to the Mixture class's constructor method. Therefore, despite my intentions to only implement a Mixture Class that inherits from `sklearn.mixture.BaseMixture <https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator>`_, I will ensure the interface remains simple and allows for alternate implementations. To ensure a clear interface I will write an abstract base class :class:`ChronBaseMixture` which all mixtures will inherit from.

Data treatment
--------------

To achieve first the **minimal working example**, I propose disregarding data uncertainties completely. RV's are pretty good (compared to the occupied velocity volumn of associations), and datasets with no RV's are too large for Chronostar to successfully fit in it's current implementation anyway. By disregarding data uncertainties, the parameter space to be explored for each (traditional) component is restricted to one dimension only, the age.

Note that this framework can still accommodate data uncertainties. The uncertainties (and correlations) would be just another feature column and it will be up to the Component to interpet correctly.

.. note::
  Intepreting uncertainties and correlations at the component level will lead to repeated calculations (e.g. constructing a covariance matrix each time) however I deem this a worthy trade off for the simplicity and flexibility of the framework as a whole. Especially since the covariance reconstruction will likely occur alongside some expensive parameter exploration method.

Example Mixture Usage
---------------------

A simple usage will look something like this::

    # ...
    # initial conditions is a list of Component objects with pars set sensibly
    inital_conditions = get_some_valid_initial_conditions(ComponentClass)

    # config params is a dictionary
    config_params = read_relevant_config_params(config_file)

    # data is array-like of shape (n_samples, n_features)
    data = read_in_preprocessed_data(data_file)

    # Note: the Component objects know how to interpret the data

    # Initialse model, and fit to data
    m = ComponentMixture(**config_params)
    m.set_params(initial_conditions)
    m.fit(data)


Suggested ComponentMixture implementation
-----------------------------------------

The bulk of the implementation is within sklearn's BaseMixture. The key abstract methods that ComponentMixture must implement are :func:`ComponentMixture.set_params` and :func:`ComponentMixture_estimate_log_prob`.

These could be done as follows::
  
  def set_params(self, components):
    self.components = components

  def _estimate_log_prob(self, X):
    n_samples = X.shape[0]
    n_components = len(self.components)

    log_probs = np.zeros((n_samples, n_components))
    for k, component in enumerate(self.components):
      log_probs[:, k] = component.estimate_log_prob(X)
    
    return log_probs

  def _m_step(self, X, log_resp):
    for component in self.components:
      # X is the data, log_resp is log of responsibilities (a.k.a. mebership probs, a.k.a. Z)
      component.maximize(X, log_resp)

.. note::
  TODO: Confirm that responsibilities are indeed exactly membership probabilites. There might be a chance responsibilities aren't normalised to sum to 1 for each sample. `Sci-kit's docs <https://github.com/scikit-learn/scikit-learn/blob/36958fb240fbe435673a9e3c52e769f01f36bec0/sklearn/mixture/_gaussian_mixture.py#L740>`_ use "responsiblity" and "posterior probability" as synonyms.

.. note::
  There are some other abstract methods of `BaseMixture` that would need to be implemented, but from what I can see these will be trivial.