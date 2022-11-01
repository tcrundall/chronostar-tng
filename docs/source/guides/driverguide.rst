Driver Guide
============
Manage entire process of fitting a model to data. This includes parsing the config file, initialising provided modules/classes, handle any mpi pool setup, set up logging, and perform fit until convergence is signalled.

The interface is defined in :class:`Driver`.

The main loop of :class:`Driver` will get initial conditions from :class:`BaseICPool`, initialise a :class:`ComponentMixture` object, run the :func:`ComponentMixture.fit` method, report the score of the fit to :class:`BaseICPool` and track the best fit seen so far. The main loop continues until :class:`BaseICPool` ceases to yield initial conditions. The final fit is the best fit seen so far.

Example Driver usage
--------------------

A rough usage could look like this::

    from chronostar.driver import Driver
    from chronostar.component import SpaceTimeComponent
    from chronostar.mixture import ComponentMixture
    from chronostar.icpool import SimpleICPool
    from chronostar.introducer import SimpleIntroducer
    # from chronostar.uncertaintyfinder import UncertaintyFinder

    driver = Driver(
        config_file='some_config_file.yml',
        component_class=SpaceTimeComponent,
        mixture_class=ComponentMixture,
        icg_class=InitialConditionsGenerator,
        inserter_class=LazyInserter,
        # uncertaintyfinder_class=UncertaintyFinder,
    )

    driver.run(data)

Note that we pass the required classes to the driver when initialising it.
This is an example of `dependency injection <https://en.wikipedia.org/wiki/Dependency_injection>`_ and has the consequence that Driver
doesn't need to know any specifics about the SpaceTimeComponent, the Driver only
expects the SpaceTimeComponent to adhere to a prescribed interface. This interface can be defined either as a "protocol" or by requiring all ``Component`` classes to inherit from the abstract ``BaseComponent``.

.. note::
    Dependency injection seems to be more about injecting objects and not classes. So this section will need to be updated as I explore the implementation further.

This provides maximal flexibility and extensibility, as anyone can write their
own component class or model class, and substitute this class into their own
script.

Proposed Driver implementation
-------------------------------
A rough implementation of :class:`Driver` could look like this::

    class Driver:
        def __init__(
            self,
            config_file,
            component_class,
            mixture_class,
            icpool_class,
            introducer_class,
        ):
            # Set all attributes
            self.config_file = config_file
            # etc ...

            # Initialise each class from pars in config file by calling static methods
            self.config_params = parse_config_file(self.config_file)

            self.component_class.config(self.config_params['component'])
            self.mixture_class.config(self.config_params['mixture'])
            # etc ...

        def run(self, data):
            # Unclear whether icpool gets the introducer class or an instantiated object
            # Also unclear how to pass 
            icpool = self.icpool_class(
                self.component_class,
                self.introducer_class,
            )

            # This design is fancy and ambitious, but provides a clean, easily parallelised interface
            for unique_id, init_conds in icpool.pool():
                m = self.mixture_class() 
                m.set_params(initial_conditions)
                m.fit(data)
                icpool.register_result(unique_id, m, m.bic(data))

            # loop will end when icg is no longer able to generate reasonable initial conditions
            best_model = icpool.best_model()

            return best_model, best_model.memberships

            # After the best model is found, uncertainties for parameters can be found...
            # uncertaintyfinder = self.uncertaintyfinder_class()
            # best_model, uncertainties, memberships = uncertaintyfinder(best_model, data)

            # return best_model, uncertainties, memberships



