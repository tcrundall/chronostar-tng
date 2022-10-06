from typing import Union

from src.chronostar.base import BaseIntroducer, BaseComponent


class SimpleIntroducer(BaseIntroducer):
    """An Introducer object that introduces components into
    previous sets and thereby generates new sets of initial
    conditions
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @classmethod
    def configure(cls, **kwargs):
        """Set class level configuration parameters that will apply
        to all instances
        """

        if kwargs:
            print(f"{cls} config: Extra keyword arguments provided:\n{kwargs}")

    def next_gen(
        self,
        prev_comp_sets:
            Union[list[list[BaseComponent]], list[BaseComponent], None],
    ) -> list[list[BaseComponent]]:
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
            return [[self.component_class(params=None)]]

        if isinstance(prev_comp_sets[0], list):
            raise UserWarning("This Introducer accepts one set of components")

        sets: list[list[BaseComponent]] = []
        for i in range(len(prev_comp_sets)):
            next_set: list[BaseComponent] = prev_comp_sets[:]   # type: ignore
            target_comp = next_set.pop(i)

            c1, c2 = target_comp.split()

            next_set.insert(i, c2)
            next_set.insert(i, c1)
            sets.append(next_set)

        return sets
