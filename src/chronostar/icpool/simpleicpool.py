import numpy as np
from typing import Generator, Optional, Union

from src.chronostar.base import (
    BaseComponent,
    BaseMixture,
    BaseICPool,
    BaseIntroducer,
    ScoredMixture,
)


class SimpleICPool(BaseICPool):
    def __init__(self, *args, **kwargs) -> None:        # type: ignore
        super().__init__(*args, **kwargs)

        # Perhaps do this in pool()?
        self.introducer: BaseIntroducer = self.introducer_class(
            self.config_params, self.component_class
        )

    def register_result(
        self,
        unique_id: Union[str, int],
        mixture: BaseMixture,
        score: float,
    ) -> None:
        self.registry[unique_id] = ScoredMixture(mixture, score)

    def pool(self) -> Generator[tuple[int, list[BaseComponent]], None, None]:
        best_mixture: Optional[BaseMixture] = None
        prev_best_score: Optional[float] = None
        best_score = -np.inf

        while prev_best_score is None or best_score > prev_best_score:
            prev_best_score = best_score
            self.registry = {}

            # Loop over the next generation of initial conditions
            for ix, init_conditions in enumerate(
                self.introducer.next_gen(
                    None if best_mixture is None else [list(
                        best_mixture.get_components()
                    )]
                )
            ):
                yield ix, init_conditions

            # Once all initial conditions are provided, look for best registry
            best_mixture, best_score = max(
                self.registry.values(),
                key=lambda x: x.score
            )

            # Using best fit, repeat until score ceases to improve

            assert isinstance(best_mixture, BaseMixture)
            self.best_mixture_ = best_mixture

    @property
    def best_mixture(self) -> BaseMixture:
        return self.best_mixture_
