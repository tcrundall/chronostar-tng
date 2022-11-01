import numpy as np

from ..context import chronostar        # noqa F401

from chronostar.icpool.simpleicpool import SimpleICPool
from .fooclasses import FooComponent, FooMixture
from .fooclasses import CONFIG_PARAMS, DATA


def test_construction() -> None:
    SimpleICPool.configure(**CONFIG_PARAMS["icpool"])
    FooComponent.configure(**CONFIG_PARAMS["component"])
    icpool = SimpleICPool(    # noqa F841
        component_class=FooComponent,
    )


def test_simple_usage() -> None:
    """
    The score decreases each iteration. Depending on FooIntroducer
    shouldn't 'fit' more than two mixtures
    """
    SimpleICPool.configure(**CONFIG_PARAMS["icpool"])
    FooComponent.configure(**CONFIG_PARAMS["component"])
    FooMixture.configure(**CONFIG_PARAMS["mixture"])

    icpool = SimpleICPool(
        component_class=FooComponent,
    )

    score = 10.
    while icpool.has_next():
        (unique_id, init_conds) = icpool.get_next()
        ncomps = len(init_conds)
        init_weights = np.ones(ncomps) / ncomps
        m = FooMixture(init_weights, init_conds)
        ncomps = len(init_conds)
        m.fit(DATA)
        icpool.register_result(unique_id, m, score)

        score -= 1.
