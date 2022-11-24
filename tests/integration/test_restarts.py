from pathlib import Path
import os

from ..context import chronostar

import numpy as np
from chronostar.driver import Driver
from chronostar import synthdata
from chronostar.icpool.simpleicpool import SimpleICPool
from chronostar.component.spacecomponent import SpaceComponent
from chronostar.mixture.componentmixture import ComponentMixture


def test_restart():
    """
    Set up a 4 (space-only) component dataset
    Fit up to 3 components, then halt run (by setting max_components)
    Restart with the intermediate results dir
    
    (if possible, try and crash run on the second 3 component fit....)
    """
    ##################################################
    #######   Generate a 4-component dataset    ######
    ##################################################
    DIM = 6
    stdev_pos = 10.
    stdev_vel = 3.
    cov = np.eye(DIM)
    cov[:3] *= stdev_pos**2
    cov[3:] *= stdev_vel**2

    all_nstars = [100, 200, 300, 400]

    all_means = [
        np.zeros(DIM) + 0.,
        np.zeros(DIM) + 10.,
        np.zeros(DIM) + 20.,
        np.zeros(DIM) + 30.,
    ]

    # seed = 0 --> leads to some weird behaviour where assoc retains an age of 0.
    seed = 0
    rng = np.random.default_rng(seed)

    all_stars = np.zeros((0, DIM))

    for mean, nstars in zip(all_means, all_nstars):
        stars = synthdata.generate_association(
            mean, cov, age=0, nstars=nstars, rng=rng,
        )
        all_stars = np.vstack((all_stars, stars))

    test_dir = Path(os.path.dirname(__file__))
    initial_configfile = test_dir / 'test_resources' / 'initial_configfile.yml'
    restart_configfile = test_dir / 'test_resources' / 'restart_configfile.yml'


    ##################################################
    #######   Run a fit up until 3 components   ######
    ##################################################
    driver = Driver(
        config_file=initial_configfile,
        icpool_class=SimpleICPool,
        component_class=SpaceComponent,
        mixture_class=ComponentMixture,
    )

    driver.run(data=all_stars)


    ##################################################
    #######   Run a fit restarting from 3 comps ######
    ##################################################
    del driver
    # config_params = driver.read_config_file(restart_configfile)
    # driver.configure(**config_params['driver'])
    driver = Driver(
        config_file=restart_configfile,
        icpool_class=SimpleICPool,
        component_class=SpaceComponent,
        mixture_class=ComponentMixture,
    )

    best_mixture = driver.run(data=all_stars)
    del driver
    assert False
