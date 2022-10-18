import numpy as np
from queue import Queue
from typing import Callable, Optional

from ..base import (
    BaseMixture,
    BaseICPool,
    BaseIntroducer,
    ScoredMixture,
    InitialCondition,
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
        self.queue: Queue[InitialCondition] = Queue()
        self.best_mixture_: Optional[BaseMixture] = None
        self.best_score_: float = -np.inf

        self.first_pass = True
        self.generation = 0

    def register_result(
        self,
        label: str,
        mixture: BaseMixture,
        score: float,
    ) -> None:
        """Register the result of a completed fit

        Parameters
        ----------
        label : str
            A uniquely identifying label with summary information:
            ``unique_id-parent_id-generation-ncomps``
        mixture : BaseMixture
            A mixture object whose fit has been finalised
        score : float
            A score of the fit, where higher means better,
            e.g. -BIC
        """

        self.registry[label] = ScoredMixture(mixture, score, label)

    def try_populate_queue(self) -> None:
        """Attempt to populate the queue of initial conditions
        """
        # If this is our first pass, let our introducer provide starting point
        if self.first_pass:
            print("Letting introducer generate first IC")
            for init_conds in self.introducer.next_gen(None):
                self.queue.put(init_conds)
            self.first_pass = False
            print(f"{self.generation=}")
            self.generation += 1
            return

        # Otherwise, check registry, see if we should generate next generation
        # If we are serial, we can trust that each run has been registered.
        # parallel needs some more thinking
        best_mixture, best_score, best_label = max(
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
            base_init_condition = InitialCondition(
                best_label,
                tuple(best_mixture.get_components())
            )

            for init_condition in self.introducer.next_gen(base_init_condition):
                if len(init_condition.components) <= self.max_components:
                    self.queue.put(init_condition)
                else:
                    print(f"[SimpleICPool]:"
                          f"Discarded IC {init_condition.label} "
                          f"{len(init_condition.components)} > {self.max_components=}")

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

    def get_next(self) -> InitialCondition:
        """Get the next initial condition set from queue

        Returns
        -------
        tuple[int, list[BaseComponent]]
            (unique_id, initial condition set)
        """
        return self.queue.get()

    def provide_start(self, init_conds: InitialCondition):
        self.queue.put(init_conds)
        self.first_pass = False

    @property
    def best_mixture(self) -> BaseMixture:
        """Get the mixture with the best score

        Returns
        -------
        BaseMixture
            The best fitting mixture
        """
        return self.best_mixture_           # type: ignore
