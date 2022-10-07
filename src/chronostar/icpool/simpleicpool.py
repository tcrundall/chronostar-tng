import numpy as np
from typing import Generator, Optional, Union

from ..base import (
    BaseComponent,
    BaseMixture,
    BaseICPool,
    BaseIntroducer,
    ScoredMixture,
)


class SimpleICPool(BaseICPool):
    """Manager and populator of a pool of initial conditions
    """

    def __init__(self, *args, **kwargs) -> None:        # type: ignore
        super().__init__(*args, **kwargs)
        """Constructor method
        """

        # Perhaps do this in pool()?
        self.introducer: BaseIntroducer = self.introducer_class(
            self.component_class
        )

    @classmethod
    def configure(cls, max_components=30, **kwargs):
        """Set class level configuration parameters that will be
        carried through to all instances.

        Parameters
        ----------
        max_components : int
            An upper limit on how many components can make up a
            set of initial conditions, by default 30
        """

        cls.max_components = max_components

        if kwargs:
            print(f"{cls} config: Extra keyword arguments provided:\n{kwargs}")

    def register_result(
        self,
        unique_id: Union[str, int],
        mixture: BaseMixture,
        score: float,
    ) -> None:
        """Register the result of a completed fit

        Parameters
        ----------
        unique_id : Union[str, int]
            A unique identifier
        mixture : BaseMixture
            A mixture object whose fit has been finalised
        score : float
            A score of the fit, where higher means better,
            e.g. -BIC
        """

        self.registry[unique_id] = ScoredMixture(mixture, score)

    def pool(self) -> Generator[tuple[int, list[BaseComponent]], None, None]:
        """Produce a generator which will yields a set of initial conditions,
        one at a time

        Yields
        ------
        Generator[tuple[int, list[BaseComponent]], None, None]
            TODO: understand what i should write here...
        """
        best_mixture: Optional[BaseMixture] = None
        prev_best_score: Optional[float] = None
        best_score = -np.inf

        while prev_best_score is None or best_score > prev_best_score:
            print(f"-----{-best_score=}")
            self.best_mixture_ = best_mixture
            prev_best_score = best_score
            self.registry = {}

            # Loop over the next generation of initial conditions
            for ix, init_conditions in enumerate(
                self.introducer.next_gen(
                    None if best_mixture is None else list(
                        best_mixture.get_components()
                    )
                )
            ):
                # Only yield component sets that are within limits
                if len(init_conditions) < self.max_components:
                    yield ix, init_conditions

            # Once all initial conditions are provided, look for best registry
            # if self.registry:
            best_mixture, best_score = max(
                self.registry.values(),
                key=lambda x: x.score
            )

    @property
    def best_mixture(self) -> BaseMixture:
        """Get the mixture with the best score

        Returns
        -------
        BaseMixture
            The best fitting mixture
        """
        return self.best_mixture_           # type: ignore
