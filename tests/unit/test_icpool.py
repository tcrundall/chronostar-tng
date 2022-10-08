import numpy as np

from ..context import chronostar     # noqa

from chronostar.icpool.simpleicpool import SimpleICPool
from .fooclasses import FooComponent, FooIntroducer, FooMixture
from .fooclasses import CONFIG_PARAMS


def test_construction() -> None:
    SimpleICPool.configure(**CONFIG_PARAMS["icpool"])
    FooIntroducer.configure(**CONFIG_PARAMS["introducer"])
    FooComponent.configure(**CONFIG_PARAMS["component"])
    icpool = SimpleICPool(    # noqa F841
        introducer_class=FooIntroducer,
        component_class=FooComponent,
    )


def test_simple_usage() -> None:
    """
    The score decreases each iteration. Depending on FooIntroducer
    shouldn't 'fit' more than two mixtures
    """
    SimpleICPool.configure(**CONFIG_PARAMS["icpool"])
    FooIntroducer.configure(**CONFIG_PARAMS["introducer"])
    FooComponent.configure(**CONFIG_PARAMS["component"])
    FooMixture.configure(**CONFIG_PARAMS["mixture"])

    icpool = SimpleICPool(
        introducer_class=FooIntroducer,
        component_class=FooComponent,
    )

    score = 10.
    while icpool.has_next():
        (unique_id, init_conds) = icpool.get_next()
        ncomps = len(init_conds)
        init_weights = np.ones(ncomps) / ncomps
        m = FooMixture(init_weights, init_conds)
        ncomps = len(init_conds)
        m.set_parameters((np.ones(ncomps)/ncomps, init_conds))
        icpool.register_result(unique_id, m, score)

        score -= 1.
        print(score)
