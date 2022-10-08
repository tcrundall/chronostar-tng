Scripts
=======

Fitting a component
-------------------

.. code-block:: python

    import numpy as np
    import yaml

    from chronostar.component.spacetimecomponent import SpaceTimeComponent

    # prepare your data array, e.g. by loading from file
    data = np.load('path/to/data.npy')

    # Optionally can prepare your membership array, e.g. by loading from file
    memb_probs = np.load('path/to/membership.npy')

    # Configuration parameters are expected as a dictionary.
    # We recommend storing them in a .yaml file and loading them like so:
    with open('path/to/config.yml', 'r') as stream:
        config_params = yaml.safe_load(stream)

    # But you may build your config dictionary however you like
    # Before usage, the SpaceTimeComponent class must be configured
    # Here are three different options

    # With config parameter dictionary
    SpaceTimeComponent.configure(**config_params)

    # With explicit parameter setting
    SpaceTimeComponent.configure(
        minimize_method='brent',
        trace_orbit_func=some_func_you_imported,
        )

    # Or with no configuration
    SpaceTimeComponent.configure()

    # Now we can instantiate the class and perform the fit
    comp = SpaceTimeComponent()
    comp.maximize(data, np.log(memb_probs))

    # The best fitting parameters can be accessed as a tuple
    params = comp.get_parameters()

    # In the case of SpaceTimeComponent, the parameters are:
    mean, covariance, age = params

    # But this varies depending on the implementation of Component

.. note::

    Currently maximize accepts the log of :code:`memb_probs`. I plan
    to change that to be just :code:`memb_probs`. I must remember
    to update docs accordingly.

Fitting a Mixture
-----------------

.. code-block:: python

    import numpy as np
    import yaml

    from chronostar.component.spacetimecomponent import SpaceTimeComponent
    from chronostar.mixture.componentmixture import ComponentMixture

    # prepare your data array, e.g. by loading from file
    data = np.load('path/to/data.npy')

    # Configuration parameters are expected as a dictionary.
    # We recommend storing them in a .yaml file and loading them like so:
    with open('path/to/config.yml', 'r') as stream:
        config_params = yaml.safe_load(stream)
    # An example yaml file is shown below

    # With config parameter dictionary
    SpaceTimeComponent.configure(**config_params["component"])
    ComponentMixture.configure(**config_params["mixture"])

    # Set up initial conditions
    # In an ideal world, we would be able to initialise with membership
    # probabilites. Alas, that isn't yet implemented. Instead you must
    # set initial weights (amplitudes) and set the parameters of each
    # component. Or you can let things be initialised randomly by the
    # mixture class.

    # Say we know of 4 components with stellar counts 100, 300, 400, 500,
    init_weights = np.array([100., 300., 400., 500.])
    init_weights /= np.sum(init_weights)

    # We initialise components
    init_comps = [SpaceTimeComponent() for _ in range(4)]
    means = [
        #X   Y   Z   U   V   W
        [0., 0., 0., 0., 0., 0.],       # comp 1
        [1., 1., 1., 1., 1., 1.],       # comp 2...
        [2., 2., 2., 2., 2., 2.],
        [3., 3., 3., 3., 3., 3.],
    ]
    covs = [np.eye(6) for _ in range(4)]

    for c, mean, cov in zip(init_comps, means, covs):
        c.set_parameters((mean, cov, 0.))

    # Now we can instantiate the class and perform the fit
    mixture = ComponentMixture(init_weights, init_comps)
    mixture.fit(data)

    # Get the fitted weights and components 
    weights, components = mixture.get_parameters()

    # Do with this information what you wish
    print(weights)
    for c in components:
        for param in c.get_parameters:
            print(param)

Example config.yml file
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    mixture: {}

    component:
        reg_covar: 1.e-5
        minimize_method: 'golden'
        trace_orbit_func: 'epicyclic'
        morph_cov_func: 'elliptical'


Fit Chronostar
--------------

.. code-block:: python

    import numpy as np
    import yaml

    # Import driver
    from chronostar.driver import Driver

    # Import any desired module substitutes
    from chronostar.component.spacecomponent import SpaceComponent

    # Prepare data
    data = np.load('path/to/data.npy')

    # Instantiate the driver class
    driver = Driver(
        config_file='path/to/config.yml',
        component_class=SpaceComponent,
    )

    # Run Chronostar
    best_mixture = driver.run(data)

    # Analyse the resulting best fit however you like
    weights, comps = best_mixture.get_parameters()
    print(f"{weights=}")
    for i, comp in enumerate(comps):
        print(f"---- Component {i} -----")
        for j, param in enumerate(comp.get_parameters()):
            print(f"  -- param {j} --")
            print(param)

