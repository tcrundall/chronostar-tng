import numpy as np

from src.chronostar.mixture.componentmixture import ComponentMixture
from tests.unit.fooclasses import FooComponent, CONFIG_PARAMS, DATA


def test_construction() -> None:
    cm = ComponentMixture(                          # noqa F841
        CONFIG_PARAMS['mixture'],
        init_weights=np.ones(1),
        init_components=[FooComponent(CONFIG_PARAMS['component'])]
    )


def test_simple_usage() -> None:
    cm = ComponentMixture(                          # noqa F841
        CONFIG_PARAMS,
        init_weights=np.ones(5)/5,
        init_components=[FooComponent(CONFIG_PARAMS) for _ in range(5)],
    )
    cm.fit(DATA)
    score = cm.bic(DATA)                            # noqa F841


if __name__ == '__main__':
    test_construction()
