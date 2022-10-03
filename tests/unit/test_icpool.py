import numpy as np

from src.chronostar.icpool.simpleicpool import SimpleICPool
from tests.unit.fooclasses import FooComponent, FooIntroducer, FooMixture
from tests.unit.fooclasses import CONFIG_PARAMS


def test_construction() -> None:
    icpool = SimpleICPool(    # noqa F841
        config_params=CONFIG_PARAMS,
        introducer_class=FooIntroducer,
        component_class=FooComponent,
    )


def test_simple_usage() -> None:
    """
    The score decreases each iteration. Depending on FooIntroducer
    shouldn't 'fit' more than two mixtures
    """
    icpool = SimpleICPool(
        config_params=CONFIG_PARAMS,
        introducer_class=FooIntroducer,
        component_class=FooComponent,
    )

    score = 10.
    for (unique_id, init_conds) in icpool.pool():
        ncomps = len(init_conds)
        init_weights = np.ones(ncomps) / ncomps
        m = FooMixture(CONFIG_PARAMS, init_weights, init_conds)
        ncomps = len(init_conds)
        m.set_params((np.ones(ncomps)/ncomps, init_conds))
        icpool.register_result(unique_id, m, score)

        score -= 1.
        print(score)
