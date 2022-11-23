=============================
Initial Conditions Pool Guide
=============================

A  Initial Conditions Pool (ICPool) inherits from :class:`~chronostar.base.BaseICPool` 
and manages a queue of :class:`~chronostar.base.InitialCondition`\ s.

An :class:`~chronostar.base.InitialCondition` is
a tuple of :class:`~chronostar.base.BaseComponent`\ s combined with a unique, informative
label stored as a string and is
used to initialise a :class:`~chronostar.base.BaseMixture` object.

The ICPool attempts to keep its queue populated with reasonable InitialConditions.
It does this by generating a new generation of InitialConditions each time the
queue becomes empty. Precisely how the new generation is generated depends
on the specific implementation.

The ICPool expects the result of a fit to a given InitialCondition to be
registered back to it with :func:`~chronostar.base.BaseICPool.register_result`
along with the fit's score.
The ICPool can then use the scores to determine the best fit, and use that to
generate the next generation.

See the `source code` for :class:`~chronostar.driver.Driver` for a full example of
how to utilise an ICPool.

Implemented ICPools
-------------------

.. _guide-simpleicp:

:class:`~chronostar.icpool.simpleicpool.SimpleICPool`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A simple implementation ``SimpleICPool`` identifies the best mixture with :math:`k`
components and uses this to generate a set of :math:`k+1`-component initial conditions. This is the algorithm used in Chronostar Paper I.

The set of :math:`k+1`-component initial conditions is formed thusly:
for each component (named the *target* component) in the best fitting :math:`k`-component mixture, make
a new initial condition. This new initial condition has an identical set of components as the :math:`k`-component mixture, except the target component.
The target component is :func:`~chronostar.base.BaseComponent.split` into
two similar, overlapping components that when combined more or less describe the target component.

This is a safe approach that guarantees the discovery of the best Mixture for
a given number of components. However, it involves a lot of repeated computation, since in order to go from :math:`k` components to :math:`k+1` compoents,
:math:`k` fits must be performed. One can see that this results in multiple splitting of the majority of the components. Indeed this algorithm scales
quadratically with the number of components.

.. _guide-greedyicp:

:class:`~chronostar.icpool.greedycycleicp.GreedyCycleICP`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A more efficient approach is the ``GreedyCycleICP``. It is named the ``GreedyCycleICP`` because each time it finds a component that improves the fit's
score, this ICPool immediately incorporates it into its best fit.

This approach is more efficient, because fewer fits are needed before a
new component is found. This approach can miss the best solution, however,
by prematurely accepting a component in one region of the dataspace, when
a different region would have benefitted more, i.e. the solution gets stuck in
a local minimum/maximum.
