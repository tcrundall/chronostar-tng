Driver Guide
============

The :class:`~chronostar.driver.Driver` manages the
entire process of fitting a chronostar model to data. This includes parsing the config file, initialising provided modules/classes, handling any mpi pool setup (not yet implemented), setting up logging (also not yet implemented), and performing fits until convergence is signalled.

The main loop of :class:`~chronostar.driver.Driver` will get initial conditions from :class:`~chronostar.base.BaseICPool`, initialise a :class:`~chronostar.base.BaseMixture` object, run the :func:`~chronostar.base.BaseMixture.fit` method, report the score of the fit to ``BaseICPool`` and track the best fit seen so far. The main loop continues until ``BaseICPool`` ceases to yield initial conditions. The final fit is the best fit seen so far.

A ``Driver`` takes as arguments the various classes you wish to use in the run.
Therefore, if you have defined your own custom class for a specific dataset, you
need only provide that to the ``Driver`` when initialising it.
See :ref:`Fit Chronostar <scripts-chron>` for an example script which demonstrates this.
