Scripts
=======

.. _scripts-comp:

Fitting a component
-------------------

.. code-block:: python

    import numpy as np
    import yaml

    from chronostar.component.spherespacetimecomponent import SphereSpaceTimeComponent
    from chronostar.component.spherespacetimecomponent import construct_params_from_cov

    # prepare your data array, e.g. by loading from file
    data = np.load('path/to/data.npy')

    # Optionally can prepare your membership array, e.g. by loading from file
    memb_probs = np.load('path/to/membership.npy')

    # Configuration parameters are expected as a dictionary.
    # We recommend storing them in a .yaml file and loading them like so:
    with open('path/to/config.yml', 'r') as stream:
        config_params = yaml.safe_load(stream)

    # But you may build your config dictionary however you like
    # Before usage, the SpaceTimeComponent class may be configured
    # Here are two examples

    # With config parameter dictionary
    SphereSpaceTimeComponent.configure(**config_params)

    # With explicit parameter setting
    SphereSpaceTimeComponent.configure(
        minimize_method='Nelder-Mead',
        trace_orbit_func=some_func_you_imported,
        )

    # Now we can instantiate the class and perform the fit
    comp = SphereSpaceTimeComponent()
    comp.maximize(data, memb_probs)

    # The best fitting parameters can be accessed as an array
    params = comp.get_parameters()
    birth_mean = params[:6]
    birth_dxyz = params[6]
    birth_duvw = params[7]
    age = params[8]

    # In the case of SphereSpaceTimeComponent, the derived parameters are:
    current_day_mean = comp.mean
    current_day_covariance = comp.covariance
    age = comp.age

    # But this varies depending on the implementation of Component

Example config.yaml file
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    component:
        minimize_method: 'Nelder-Mead'
        reg_covar: 1e-6
        trace_orbit_func: 'epicyclic'

.. _scripts-mixture:

Fitting a Mixture
-----------------

.. code-block:: python

    import numpy as np
    import yaml

    from chronostar.component.spacetimecomponent import SphereSpaceTimeComponent
    from chronostar.mixture.componentmixture import ComponentMixture

    # prepare your data array, e.g. by loading from file
    data = np.load('path/to/data.npy')

    # Configuration parameters are expected as a dictionary.
    # We recommend storing them in a .yaml file and loading them like so:
    with open('path/to/config.yml', 'r') as stream:
        config_params = yaml.safe_load(stream)
    # An example yaml file is shown below

    # With config parameter dictionary
    SphereSpaceTimeComponent.configure(**config_params["component"])
    ComponentMixture.configure(**config_params["mixture"])

    # --------------------------------------------------
    # Set up initial conditions
    # --------------------------------------------------
    # initial conditions can be set by
    #   (1) membership probabilities,
    #   (2) by component parameters
    #   (3) by sklearn methods

    # 1. -----------------------------------------------
    # Initialising by membership probabilities:

    # Load in an array of shape (n_stars, n_components)
    init_weights = np.load('path/to/membership.npy')
    n_stars, n_comps = init_weights.shape

    # Construct list of raw components
    init_comps = [SphereSpaceTimeComponent() for _ in range(n_comps)]

    # Initialise mixture
    mixture = ComponentMixture(init_weights, init_comps)

    # Run the fit
    mixture.fit(data)

    # Access results
    weights, components = mixture.get_parameters()
    membership_probs = mixture.estimate_membership_probs(data)

    # 2. --------------------------------------------------
    # Initialising by components

    # This is less straight forward, especially since time components
    # are parameterised by their bith-mean and -covariance. Typically
    # this approach would only be used if you have the output of a previous
    # fit. Of course you could take current day means, expected ages, trace
    # those means back, and take a guess at their birth covs.

    # Say we know of 4 components with stellar counts 100, 300, 400, 500,
    n_comps = 4
    init_weights = np.array([100., 300., 400., 500.])
    init_weights /= np.sum(init_weights)

    # With birth means, birth covs and ages:
    init_comps = []
    for i in range(n_comps):
        # Load in the parameters stored as a one dimensional array
        comp_pars = np.load(f'path/to/prev/result/comp_{i:03}/pars.npy')
        init_comps.append(SphereSpaceTimeComponent(comp_pars))

    # Now we can instantiate the class and perform the fit
    mixture = ComponentMixture(init_weights, init_comps)
    mixture.fit(data)

    # Get the fitted weights and components 
    weights, components = mixture.get_parameters()
    membership_probs = mixture.estimate_membership_probs(data)

    # Do with this information what you wish
    print(weights)
    for c in components:
        for param in c.get_parameters:
            print(param)

    # 3. --------------------------------------------------
    # Letting sklearn initialise things

    n_comps = 4
    init_weights = np.ones(len(n_comps)) / n_comps

    init_comps = [SphereSpaceTimeComponent() for _ in range(n_comps)]

    # Notice that we gave the components no parameters
    # Components will therefore have the attribute .parameters_set = False
    # ComponentMixture will detect this and prime SKLMixture to 
    # run one of its initialisation routines, as determined by `init_params`
    # in the config file

    mixture = ComponentMixture(init_weights, init_comps)
    mixture.fit(data)

    weights, components = mixture.get_parameters()
    membership_probs = mixture.estimate_membership_probs(data)


Example config.yml file
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    mixture:
        tol: 1.e-4
        verbose: 2
        verbose_interval: 1

        # (1) Set this if initialising with membership probs!
        init_params: 'init_resp'
        # --------------------------------------------------
        # (2) If initialising with components, init_params is ignored
        # --------------------------------------------------
        # (3) If letting sklearn initialise the fit, pick a method
        init_params: 'kmeans'

    component:
        reg_covar: 1.e-5
        minimize_method: 'Nelder-Mead'
        trace_orbit_func: 'epicyclic'

.. _scripts-chron:

Fit Chronostar
--------------

Running full chronostar is the simplest script of them all, because
the :class:`Driver` handles everything. The default classes used are
:class:`SphereSpaceTimeComponent`, :class:`ComponentMixture` and
:class:`SimpleICPool`.
If you wish to use an alternative (either included in Chronostar or
a custom class of your own) simply import it and pass it to the :class:`Driver`.

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

Example config.yml file
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    driver: {}

    icpool:
        max_components: 100

    mixture:
        tol: 1.e-4
        verbose: 2
        verbose_interval: 1
        # Initialisation mode isn't relevant, since chronostar
        # typically begins with a one component fit
        # So lets just avoid choosing anything that might lead to
        # unnecessary computation (i.e. avoid 'kmeans')
        init_params: 'random'

    component:
        reg_covar: 1.e-5
