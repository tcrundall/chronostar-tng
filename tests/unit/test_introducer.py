import numpy as np

from src.chronostar.introducer.simpleintroducer import SimpleIntroducer
from tests.unit.fooclasses import CONFIG_PARAMS, DATA, FooComponent, FooMixture


def test_construction() -> None:
    SimpleIntroducer.configure(**CONFIG_PARAMS['introducer'])
    FooComponent.configure(**CONFIG_PARAMS['component'])

    introducer = SimpleIntroducer(   # noqa F401
        component_class=FooComponent,
    )


def test_comp_count_increase() -> None:
    SimpleIntroducer.configure(**CONFIG_PARAMS['introducer'])
    FooComponent.configure(**CONFIG_PARAMS['component'])

    introducer = SimpleIntroducer(
        component_class=FooComponent,
    )

    comp_set_1 = [FooComponent(params=None) for _ in range(1)]
    [c.maximize(None, None) for c in comp_set_1]    # type: ignore
    next_gen = introducer.next_gen(comp_set_1)      # type: ignore
    for comp_set in next_gen:
        assert len(comp_set) == 2

    comp_set_2 = next_gen[0]
    next_gen = introducer.next_gen(comp_set_2)
    for comp_set in next_gen:
        assert len(comp_set) == 3


def test_full_usage() -> None:
    SimpleIntroducer.configure(**CONFIG_PARAMS['introducer'])
    FooComponent.configure(**CONFIG_PARAMS['component'])
    FooMixture.configure(**CONFIG_PARAMS['mixture'])

    introducer = SimpleIntroducer(
        component_class=FooComponent,
    )

    comp = FooComponent(params=None)
    # hacky way to ensure mean and cov are set
    comp.maximize(None, None)       # type: ignore

    best_mixture = FooMixture(
        np.ones(1),
        [comp]
    )

    best_score = best_mixture.bic(DATA)
    prev_best_score = -np.inf

    while best_score > prev_best_score:
        # Produce the next generation and loop over them
        for next_init_cond in introducer.next_gen(
            best_mixture.get_components()
        ):
            ncomps = len(next_init_cond)
            init_weights = np.ones(ncomps) / ncomps
            m = FooMixture(
                init_weights,
                next_init_cond
            )
            m.fit(DATA)
            if m.bic(DATA) > prev_best_score:
                prev_best_score = best_score
                best_mixture = m
                best_score = m.bic(DATA)
        print(best_score)
