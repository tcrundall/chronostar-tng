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


