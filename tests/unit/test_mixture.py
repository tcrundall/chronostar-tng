import numpy as np

from ..context import chronostar     # noqa

from chronostar.mixture.componentmixture import ComponentMixture
from .fooclasses import FooComponent, CONFIG_PARAMS, DATA


def test_construction() -> None:
    FooComponent.configure(**CONFIG_PARAMS['component'])
    ComponentMixture.configure(**CONFIG_PARAMS['mixture'])

    cm = ComponentMixture(                          # noqa F841
        init_weights=np.ones(1),
        init_components=[FooComponent(params=None)]
    )


def test_simple_usage() -> None:
    FooComponent.configure(**CONFIG_PARAMS['component'])
    ComponentMixture.configure(**CONFIG_PARAMS['mixture'])

    cm = ComponentMixture(                          # noqa F841
        init_weights=np.ones(5)/5,
        init_components=[FooComponent(params=None) for _ in range(5)],
    )

    cm.fit(DATA)
    score = cm.bic(DATA)                            # noqa F841


if __name__ == '__main__':
    test_construction()
