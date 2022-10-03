import os
from pathlib import Path
import numpy as np

from src.chronostar.driver.driver import Driver
from src.chronostar.mixture.componentmixture import ComponentMixture
from src.chronostar.icpool.simpleicpool import SimpleICPool
from src.chronostar.introducer.simpleintroducer import SimpleIntroducer
# from src.chronostar.component.spacetimecomponent import SpaceTimeComponent
from src.chronostar.component.spacecomponent import SpaceComponent


def test_simple_spacemixture_run():
    curr_dir = Path(os.path.dirname(__file__))
    config_file = curr_dir / 'test_resources' / 'placeholder_configfile.yml'
    driver = Driver(
        config_file=config_file,
        mixture_class=ComponentMixture,
        icpool_class=SimpleICPool,
        introducer_class=SimpleIntroducer,
        component_class=SpaceComponent,
    )

    data = np.random.rand(100, 6)

    best_mixture = driver.run(data)         # noqa F841
    return best_mixture


if __name__ == '__main__':
    best_mixture = test_simple_spacemixture_run()
