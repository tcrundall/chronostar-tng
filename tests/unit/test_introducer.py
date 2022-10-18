import numpy as np

from ..context import chronostar     # noqa

from chronostar.introducer.simpleintroducer import SimpleIntroducer
from .fooclasses import CONFIG_PARAMS, DATA, FooComponent, FooMixture
from chronostar.base import InitialCondition


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

    comp_set_1 = tuple([FooComponent(params=None) for _ in range(1)])
    [c.maximize(None, None) for c in comp_set_1]    # type: ignore
    prev_init_cond = InitialCondition(
        label='first',
        components=comp_set_1,
    )
    next_gen = introducer.next_gen(prev_init_cond)      # type: ignore
    for init_cond in next_gen:
        assert len(init_cond.components) == 2

    init_cond_2 = next_gen[0]
    next_gen = introducer.next_gen(init_cond_2)
    for init_cond in next_gen:
        assert len(init_cond.components) == 3


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
        init_comps=[comp],
        init_weights=np.ones(1),
    )

    best_initial_cond = InitialCondition(
        '0',
        best_mixture.get_components(),
    )

    best_score = best_mixture.bic(DATA)
    prev_best_score = -np.inf

    while best_score > prev_best_score:
        # Produce the next generation and loop over them
        for next_init_cond in introducer.next_gen(
            best_initial_cond,
        ):
            ncomps = len(next_init_cond.components)
            init_weights = np.ones(ncomps) / ncomps
            m = FooMixture(
                init_weights,
                next_init_cond.components,
            )
            m.fit(DATA)
            if m.bic(DATA) > prev_best_score:
                prev_best_score = best_score
                best_mixture = m
                best_score = m.bic(DATA)
                best_initial_cond = InitialCondition(
                    label=next_init_cond.label,
                    components=m.get_components(),
                )
        print(best_score)
