import numpy as np

from src.chronostar.mixture.componentmixture import ComponentMixture
from tests.fooclasses import CONFIG_PARAMS, DATA, FooComponent


def test_construction() -> None:
    cm = ComponentMixture(                          # noqa F401
        CONFIG_PARAMS['mixture'],
        init_weights=[1.],
        init_components=[FooComponent(CONFIG_PARAMS['component'])]
    )


def test_simple_usage() -> None:
    comps = [FooComponent(CONFIG_PARAMS) for _ in range(5)]
    cm = ComponentMixture(                          # noqa F401
        CONFIG_PARAMS,
        init_weights=list(np.ones(5)),
        init_components=comps,
    )
    cm.fit(DATA)
    score = cm.bic(DATA)        # noqa F401


if __name__ == '__main__':
    test_construction()
