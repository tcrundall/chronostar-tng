import numpy as np

from ..context import chronostar        # noqa F401

from chronostar.icpool.simpleicpool import SimpleICPool
from chronostar.icpool.greedycycleicp import GreedyCycleICP
from .fooclasses import FooComponent, FooMixture
from .fooclasses import CONFIG_PARAMS, DATA


ICP_CLASSES = [SimpleICPool, GreedyCycleICP]


def test_construction() -> None:
    SimpleICPool.configure(**CONFIG_PARAMS["icpool"])
    FooComponent.configure(**CONFIG_PARAMS["component"])
    for icpool_class in ICP_CLASSES:
        icpool = icpool_class(    # noqa F841
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

    for icpool_class in ICP_CLASSES:
        icpool = icpool_class(
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

        assert icpool.best_score_ == 10.    # type: ignore


def test_nuanced_usage() -> None:
    """
    The score increases initially than decreases. Depending on FooIntroducer
    this should terminate on the 6th mixture.
    """
    SimpleICPool.configure(**CONFIG_PARAMS["icpool"])
    FooComponent.configure(**CONFIG_PARAMS["component"])
    FooMixture.configure(**CONFIG_PARAMS["mixture"])

    for icpool_class in [SimpleICPool, GreedyCycleICP]:
        icpool = icpool_class(
            component_class=FooComponent,
        )

        scores = np.array([5., 6, 7, 8, 9, 10, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8])
        score_ix = 0
        while icpool.has_next():
            (unique_id, init_conds) = icpool.get_next()
            ncomps = len(init_conds)
            init_weights = np.ones(ncomps) / ncomps
            m = FooMixture(init_weights, init_conds)
            ncomps = len(init_conds)
            m.fit(DATA)
            icpool.register_result(unique_id, m, scores[score_ix])

            score_ix += 1

        assert icpool.best_score_ == 10.        # type: ignore
