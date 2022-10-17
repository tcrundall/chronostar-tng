import numpy as np
from queue import Queue
from typing import Callable, Optional, Union

from ..base import (
    BaseComponent,
    BaseMixture,
    BaseICPool,
    BaseIntroducer,
    ScoredMixture,
)


class SimpleICPool(BaseICPool):
    """Manager and populator of a pool of initial conditions

    Attributes
    ----------
    max_components : int, default 100
        The max components in an initial condition provided by
        `SimpleICPOol`, configurable
    """
    function_parser: dict[str, Callable] = {}
    max_components = 100

    def __init__(self, *args, **kwargs) -> None:        # type: ignore
        super().__init__(*args, **kwargs)
        """Constructor method
        """

        # Perhaps do this in pool()?
        self.introducer: BaseIntroducer = self.introducer_class(
            self.component_class
        )
        self.queue: Queue[tuple[int, list[BaseComponent]]] = Queue()
        self.best_mixture_: Optional[BaseMixture] = None
        self.best_score_: float = -np.inf

        self.first_pass = True
        self.generation = 0

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

    def try_populate_queue(self) -> None:
        """Attempt to populate the queue of initial conditions
        """
        # If this is our first pass, let our introducer provide starting point
        if self.first_pass:
            print("Letting introducer generate first IC")
            for ix, init_conds in enumerate(self.introducer.next_gen(None)):
                self.queue.put((ix, init_conds))
            self.first_pass = False
            print(f"{self.generation=}")
            self.generation += 1
            return

        # Otherwise, check registry, see if we should generate next generation
        # If we are serial, we can trust that each run has been registered.
        # parallel needs some more thinking
        best_mixture, best_score = max(
            self.registry.values(),
            key=lambda x: x.score
        )

        # If we have improved previous best mixture, save current best and
        # repopulate queue
        if best_score > self.best_score_:
            print(f"{self.generation=}")
            print(f"{best_score=}")
            self.generation += 1

            self.best_mixture_ = best_mixture
            self.best_score_ = best_score

            self.registry = {}

            # Loop over the next generation of initial conditions
            for ix, init_conditions in enumerate(
                self.introducer.next_gen(list(best_mixture.get_components()))
            ):
                print(f"Length {len(init_conditions)} ~ {self.max_components=}")
                if len(init_conditions) <= self.max_components:
                    self.queue.put((ix, init_conditions))
                else:
                    print(f"[SimpleICPool]:"
                          f"Discarded IC, "
                          f"{len(init_conditions)} > {self.max_components=}")

    def has_next(self) -> bool:
        """Return True if (after populating if needed) queue is non-empty

        Returns
        -------
        bool
            True if queue is non-empty
        """
        if not self.queue.empty():
            return True

        self.try_populate_queue()
        return not self.queue.empty()

    def get_next(self) -> tuple[int, list[BaseComponent]]:
        """Get the next initial condition set from queue

        Returns
        -------
        tuple[int, list[BaseComponent]]
            (unique_id, initial condition set)
        """
        return self.queue.get()

    @property
    def best_mixture(self) -> BaseMixture:
        """Get the mixture with the best score

        Returns
        -------
        BaseMixture
            The best fitting mixture
        """
        return self.best_mixture_           # type: ignore
