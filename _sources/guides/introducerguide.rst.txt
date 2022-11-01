================
Introducer Guide
================

An Introducer takes one (or more) previous model fit(s) and determines multiple plausible locations to introduce another :class:`Component`. The inserter communicates all plausible locations by returning a set of independent initial conditions.

A simple example would be: take a previous fit, for each component generate a new set of initial conditions by replacing the target component by two components of identical current day mean and initial covariance matrix, but with differing ages (i.e. how Crundall et al. (2019) introduced components).


Example SimpleIntroducer Usage
------------------------------

A simple usage will look something like this::

    # ...
    best_model = one_comp_model        # Assume we have fit a one-component model
    prev_best_score = one_comp_model.score

    # Unclear if this needs to be a class. Perhaps a function is sufficient...
    introducer = SimpleIntroducer()

    # loop until our score worsens
    while best_score > prev_best_score: 
        # Produce the next generation and loop over them
        for next_init_cond in introducer.next_gen(best_model)
            m = Model(config_pars)
            m._set_parameters(next_init_cond)
            m.fit()
            if m.score > prev_best_score:
                best_model = m
                best_score = m.score


.. note::
  TODO: decide if `Introducer` must be a class, or if a stand-alone function is sufficient. For now I'll perserver with a class in case encapsulating a "state" yields benefits. At the very least, a class design might be necessary to allow consistency checks between selected class variants. For example we might need to check the chosen introduction method is compatible with the intended :doc:`component model<componentguide>`.

Suggested SimpleIntroducer implementation
-----------------------------------------

A *serial* ICG implementation could look a little like this::
  
  class InitialConditionsGenerator():

    def __init__(self, ComponentClass, InsertionApproach):
      self.component_class = ComponentClass
      self.insertion_approach = InsertionApproach()

    def initial_conditions(self):

      best_model = None
      prev_best_score = None
      best_score = -np.inf
      while prev_best_score is None or best_score > prev_best_score:
        prev_best_score = best_score
        self.registry = {}

        # Loop over the next generation of initial conditions
        for ix, init_conditions in enumerate(self.insertion_approach.next_gen(best_model)):
          yield ix, init_conditions

        # Once all initial conditions are provided, look for best one in registry
        best_model, best_score = max(self.registry.values() key=lambda x: x[1])

        # Using best fit, repeat until score ceases to improve

    def register_result(self, ident, model, score):
      self.registry[ident] = (model, score)
