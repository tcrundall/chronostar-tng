=============================
Initial Conditions Pool Guide
=============================

An Initial Conditions Pool manages a pool of sets of initial conditions which could be
used to initialise a :class:`ComponentMixture` object. A *set* of initial conditions is
a list of :class:`Component`\ s. An IC Pool yields sets of
initial conditions to another object (e.g. :class:`Driver`) and expects a fit
to be performed, and the score to be registered with the IC Pool. Using a fit
and score registry, the IC Pool can keep the pool populated with reasonable
sets of initial conditions for future fits.

The usage of IC Pool is very general. Objects that access the pool remain ignorant of how many components each set of intial conditions will contain, until the set is yielded. As a consequence, implementations of :class:`BaseICPool` are free to increase component counts however they like.

Furthermore, the design of IC Pool's interface allows for asynchronous, parallel fits. For example, a multiprocessor :class:`Driver` could iterate over the IC Pool's pool, passing each set of initial conditions to one of its child processors. These processors perform fits based on these initial conditions and report the score of the completed fit back to the IC Pool, upon which they receive their next set of initial conditions. Whenever the IC Pool runs out of sets of initial conditions, it repopulates the pool by combining or extending completed fits.

To achieve this generality, IC Pool utilises the perhaps unfamiliar ``yield`` key word in the :func:`BaseICPool.pool` method. Calls to this method return a *generator*
which can be iterated over, with each iteration providing a new set of initial conditions.

To aid the explanation I outline a potential concrete example:
a simple implementation :class:`SimpleICPool` identifies the best mixture with :math:`k`
components and uses this to generate a set of :math:`k+1`-component initial conditions
(i.e. how original Chronostar did things). :ref:`See below<icpool-usage>`
for example usage and implementation.

However, more advanced approaches are possible with this interface design. For example
a (as of yet not implemented) :class:`FancyICPool` could take the best 3 :math:`k`-component
fits and combine them to produce a :math:`k+3`-component initial condition set.

.. note::
  Need to work out where the very first initial conditions come from. Is `ICPool` responsible, or is an `Introducer` object?

.. _icpool-usage:

Example SimpleICPool Usage
--------------------------

The usage will look something like this::

  icpool = SimpleICPool(SphereComponent, SimpleIntroducer)

  for ident, init_conds in icpool.pool():
    m = Mixture() 
    m.set_parameters(init_conds)
    m.fit(data)
    icpool.register_result(ident, m, m.bic(data))

  # loop will end when icpool is no longer able to generate reasonable initial conditions
  best_mixture = icpool.get_best_mixture()

See? So elegant. Behind the scenes, the ICPool object is tracking which mixtures 
performed the best, and therefore which mixtures are the best candidates for 
further exploration. The ICPool can be as clever or as simple as the user 
desires. The method of how a new component is introduced is also left 
completely free.

Suggested SimpleICPool implementation
-------------------------------------

A *serial* ICPool implementation could look a little like this::
  
  class SimpleICPool():

    def __init__(self, Component, Introducer):
      self.component_class = Component
      self.introducer = Introducer()

    def initial_conditions(self):

      best_mixture = None
      prev_best_score = None
      best_score = -np.inf
      while prev_best_score is None or best_score > prev_best_score:
        prev_best_score = best_score
        self.registry = {}

        # Loop over the next generation of initial conditions
        for ix, init_conditions in enumerate(self.introducer.next_gen(best_mixture)):
          yield ix, init_conditions

        # Once all initial conditions are provided, look for best one in registry
        best_mixture, best_score = max(self.registry.values() key=lambda x: x[1])

        # Using best fit, repeat until score ceases to improve
      
      self.best_mixture = best_mixture

    def register_result(self, ident, mixture, score):
      self.registry[ident] = (mixture, score)

    def get_best_mixture(self):
      # Maybe should return a deep copy?
      return self.best_mixture