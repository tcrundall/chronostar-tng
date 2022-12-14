import numpy as np
from queue import Queue
from typing import Optional, Type, Union

from ..base import (
    BaseComponent,
    BaseMixture,
    BaseICPool,
    ScoredMixture,
    InitialCondition,
)
from ..utils.bookkeeping import generate_label


class SimpleICPool(BaseICPool):
    """Manager and populator of a pool of initial conditions

    Attributes
    ----------
    max_components : int, default 100
        The max components in an initial condition provided by
        `SimpleICPool`, configurable
    """
    max_components = 100

    def __init__(
        self,
        component_class: Type[BaseComponent],
        start_init_comps: Optional[tuple[BaseComponent, ...]] = None,
    ) -> None:
        """Constructor method
        """
        super().__init__(component_class, start_init_comps)

        self.queue: Queue[InitialCondition] = Queue()
        self.best_mixture_: Optional[BaseMixture] = None
        self.best_score_: float = -np.inf

        self.n_initconds = 0
        self.n_generations = 0

        if start_init_comps is not None:
            self.put_in_queue(start_init_comps, parent_label='XXX', extra='input')
            self.n_generations += 1

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

        If this is the "first pass", then the ICPool object will ask
        its introducer to produce a generation given no starting point.

        Otherwise, it provides its introducer with the previous generation's
        best mixture and adds the next generation to the queue.
        """
        # If this is our first pass, let our introducer provide starting point
        if not self.registry:
            print("Letting introducer generate first IC")
            self.next_gen(None)
            self.first_pass = False
            print(f"{self.n_generations=}")
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
            print(f"{self.n_generations=}")
            print(f"{best_score=}")

            self.best_mixture_ = best_mixture
            self.best_score_ = best_score

            self.registry = {}

            # Loop over the next generation of initial conditions
            base_init_condition = InitialCondition(
                best_label,
                tuple(best_mixture.get_components())
            )

            self.next_gen(base_init_condition)

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

    def put_in_queue(
        self,
        components: tuple[BaseComponent, ...],
        parent_label: str,
        extra: str,
    ) -> None:
        """Put a new InitialCondition in the queue

        Parameters
        ----------
        components : tuple[BaseComponent, ...]
            A tuple of components
        parent_label : str
            The label of the InitialCondition that initialised the
            parent mixture
        extra : str
            An extra piece of information to append at the end
            of this InitialCondition's label
        """
        label = generate_label(
            self.n_initconds,
            self.n_generations,
            components=components,
            parent_label=parent_label,
            extra=extra,
        )
        self.queue.put(InitialCondition(label, components))
        self.n_initconds += 1

    def next_gen(
        self,
        prev_comp_sets: Union[list[InitialCondition], InitialCondition, None],
    ) -> None:
        """Generate the next generation of initial conditions by splitting
        each existing component into two

        Parameters
        ----------
        prev_comp_sets : list[BaseComponent] (optional)
            A list of components from a previous fit. If none provided,
            a single (uninitialised) component will be returned.

        Returns
        -------
        list[list[BaseComponent]]
            A list of initial conditions, where each initial condition
            is a list of components

        Raises
        ------
        UserWarning
            This introducer can only handle one set of components
        """
        if prev_comp_sets is None:
            components = (self.component_class(params=None),)
            self.put_in_queue(components, parent_label='XXX', extra='auto')
            self.n_generations += 1
            return

        if isinstance(prev_comp_sets, list):
            raise UserWarning("This Introducer only accepts one InitialCondition")

        for target_ix in range(len(prev_comp_sets.components)):
            # Perform a deep copy of components, using their class to
            # construct a replica
            next_ic_components = [
                c.__class__(c.get_parameters()) for c in prev_comp_sets.components
            ]

            # Replace the ith component by splitting it in two
            target_comp = next_ic_components.pop(target_ix)
            try:
                c1, c2 = target_comp.split()
            except UserWarning:
                # Target component not a splittable component
                continue

            next_ic_components.insert(target_ix, c2)
            next_ic_components.insert(target_ix, c1)

            if len(next_ic_components) <= self.max_components:
                self.put_in_queue(
                    components=tuple(next_ic_components),
                    parent_label=prev_comp_sets.label,
                    extra=str(target_ix),
                )
            else:
                print(f"[SimpleICPool]:"
                      f"Discarded IC derived from {prev_comp_sets.label} "
                      f"{len(next_ic_components)} > {self.max_components=}")

        self.n_generations += 1
        print(f"[SimpleIntroducer] After next_gen {self.n_generations=}")

        return

    @property
    def best_mixture(self) -> BaseMixture:
        """Get the mixture with the best score

        Returns
        -------
        BaseMixture
            The best fitting mixture
        """
        return self.best_mixture_           # type: ignore
