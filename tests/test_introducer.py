import numpy as np

from src.chronostar.base import BaseComponent
from src.chronostar.introducer.simpleintroducer import SimpleIntroducer
from tests.fooclasses import CONFIG_PARAMS, DATA, FooComponent, FooMixture


def test_construction() -> None:
    introducer = SimpleIntroducer(   # noqa F401
        config_params=CONFIG_PARAMS,
        component_class=FooComponent,
    )


def test_comp_count_increase() -> None:
    introducer = SimpleIntroducer(
        config_params=CONFIG_PARAMS,
        component_class=FooComponent,
    )

    comp_set_1 = [FooComponent(CONFIG_PARAMS) for _ in range(1)]
    assert isinstance(comp_set_1[0], BaseComponent)
    next_gen = introducer.next_gen(comp_set_1)      # type: ignore
    for comp_set in next_gen:
        assert len(comp_set) == 2

    comp_set_2 = next_gen[0]
    next_gen = introducer.next_gen(comp_set_2)
    for comp_set in next_gen:
        assert len(comp_set) == 3


def test_full_usage() -> None:
    introducer = SimpleIntroducer(
        config_params=CONFIG_PARAMS,
        component_class=FooComponent,
    )

    best_mixture = FooMixture(CONFIG_PARAMS)
    best_mixture.set_params((np.ones(1), [FooComponent(CONFIG_PARAMS)]))
    best_score = best_mixture.bic(DATA)
    prev_best_score = -np.inf

    while best_score > prev_best_score:
        # Produce the next generation and loop over them
        for next_init_cond in introducer.next_gen(
            best_mixture.get_components()
        ):
            m = FooMixture(CONFIG_PARAMS)
            ncomps = len(next_init_cond)
            m.set_params((np.ones(ncomps)/ncomps, next_init_cond))
            m.fit(DATA)
            if m.bic(DATA) > prev_best_score:
                prev_best_score = best_score
                best_mixture = m
                best_score = m.bic(DATA)
        print(best_score)
