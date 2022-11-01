===============
Component Guide
===============

A Component object encapsulates the current estimated parameters and the methods required to fit to data.

For example, a :class:`SpaceTimeComponent` would store its weight (a.k.a. amplitude), age, initial mean and initial covariance matrix. It would also have methods for calculating the log probability of stars :func:`~SpaceTimeComponent.estimate_log_prob` and fitting its parameters :func:`~SpaceTimeComponent.maximize`. It also has an :func:`~SpaceTimeComponent._project_orbit` method for calculating orbits. If a user wishes to use a custom orbit method, they could write a new class that inherits form :class:`SpaceTimeComponent` and override the :func:`_project_orbit` method.

.. note::
  The intention is to not differentiate between *Background* components and *stellar association* components. The difference will be implicit in the components' parameterization and fitting method. For example, an association component could have an age parameter, where as a background component would have none. Some other methods could be employed such as enforcing a large minimum position spread for background components and a small minimum position spread for association components. If one wishes to get fancy, one could choose to implement a :class:`BaseIntroducer` which can convert background components to association components and vica verse, based on some "failure" criteria.

Age fitting with a twist
------------------------

In this section I explain how we can implement a Component class that reproduce the same fitting power of Paper I and II components, but with only one free parameter (age), thereby drastically simplifying (and hence speeding up) parameter maximization.

If we ignore data uncertainties, the age is the only free parameter in the maximisation step. To explain this further, consider no age parameter (or equivalently fixing the age at 0). In this case we could use `sklearn.mixture.GaussianMixture` out of the box. There is no parameter exploration needed, because the maximisation step estimates the mean :math:`\mu_c'` and covariance :math:`\Sigma_c'` through dot products and element-wise arithematic. I use :math:`'` to denote values calculated or infered from the data without any parameter maximization.

:class:`SpaceTimeComponent` introduces a free age parameter :math:`t` to a component. Lets say we go ahead and calculate :math:`\mu_c'` and :math:`\Sigma_c'` like above. The key remaining **time dependent** parameters of a component are its initial mean :math:`\mu_0` and initial covariance matrix :math:`\Sigma_0`. :math:`\mu_0` is constrained completely by the age and data, since we just project :math:`\mu_c'` back in time by :math:`t`. From similar logic, :math:`\mu_c` is constrained completely by the data and :math:`\mu_c = \mu_c'`. :math:`\Sigma_0` is less apparent, but with some assumptions can be equally constrained. We have :math:`\Sigma_c'` from the assigned members. We project this back in time by :math:`t` (e.g. by using a Jacobian). We now have a :math:`\Sigma_0'`.

:math:`\Sigma_0'` cannot actually represent the origin of an association, because it may have position-velocity correlations imprinted by the time projection that are unrealistic for an unbound stellar association.

Therefore the penultimate step is to impose the core assumptions of a Component - e.g. that there is no positional-velocity correlation - and impose assumptions specific to the chosen component class - e.g. the :class:`SphereComponent` assumes spherical in position space and spherical in velocity space. Therefore in this case we can derive :math:`\Sigma_0` by finding the total space-volume of :math:`\Sigma_0'` and inferring the required radius of an equal volume sphere in position space. With similar treatment we can derive the radius in velocity space.

The final step would be to project :math:`\Sigma_0` forward by :math:`t` to attain :math:`\Sigma_c`, a covariance matrix centred on :math:`\mu_c`, that matches the extent and shape of :math:`\Sigma_c'` as closely as possible, but with the appropriate position-velocity correlations imprinted by :math:`t` Myrs of unbound dispersal through the Galaxy. In relevant log probability calculations, :class:`SpaceTimeComponent` would use :math:`\mu_c` and :math:`\Sigma_c`.

Hence, :math:`\mu_0` and :math:`\Sigma_0` are constrained completely by the data and by the age parameter :math:`t`. Therefore, any parameter exploration or maximisation approach need only explore one (easily charecterisable) parameter.


Example SpaceTimeComponent Usage
--------------------------------

A simple usage will look something like this::

  resp = # ... assume we have responsibilities (i.e. Z or membership probabilites)
  X = # ... assume we have data, array-like with shape (n_samples, n_features)
  config_params = # ... assume we have Component specific config params, e.g. which feature is in which column

  c = SpaceTimeComponent(config_params)
  c.maximize(X, resp) 
  log_probs = c.estimate_log_prob(X)

.. note::
  work out when (and why) `log_resp` is used vs `resp`.

Suggested SpaceComponent implementation
-------------------------------------------

This would effectively be the same as some of the calculation in `sklearn.mixture.GaussianMixture`. The difference is they store weights, means and covariances in the Mixture object itself, and use a module level function to maximize these parameters from the data. In a :class:`BaseComponent` implementation, each component tracks its own weight, means and covariances, and uses its own method to maximize the parameters.

.. note::
  Perhaps `weight` should be stored in the :class:`BaseMixture` implementation...

These could be done as follows::

  def maximize(self, X, log_resp):
    resp = np.exp(log_resp)
    self.mean = np.dot(resp, X) / np.sum(resp)
    diff = X - self.mean
    self.covariance = np.dot(resp * diff.T, diff) / nk[k]
    self.prec_chol = self._compute_precision_cholesky(self.covariance)

.. note::
  `sklearn.mixture.GaussianMixture` uses the cholesky decomposition of precisions. A precision is the inverse of covariance. For simplicity I duplicate their calcualtions here.

Estimating the log probability of each sample would be calculated by evaluating the Gaussain at the point of sample::

  def estimate_log_prob(X):
    log_det = _compute_log_det_cholesky(precisions_chol, self.n_features)

    y = np.dot(X, self.prec_chol) - np.dot(mu, prec_chol)
    log_prob = np.sum(np.square(y), axis=1)
    return -0.5 * (self.n_features * np.log(2 * np.pi) + log_prob) + log_det

