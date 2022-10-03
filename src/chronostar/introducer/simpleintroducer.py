from typing import Union

from src.chronostar.base import BaseIntroducer, BaseComponent


class SimpleIntroducer(BaseIntroducer):
    def __init__(self, *args, **kwargs) -> None:        # type: ignore
        super().__init__(*args, **kwargs)

    def next_gen(
        self,
        prev_comp_sets:
            Union[list[list[BaseComponent]], list[BaseComponent], None],
    ) -> list[list[BaseComponent]]:
        if prev_comp_sets is None:
            return [[self.component_class(self.config_params)]]

        if isinstance(prev_comp_sets[0], list):
            raise UserWarning("This Introducer accepts one set of components")

        sets = []
        for i in range(len(prev_comp_sets)):
            next_set: list[BaseComponent] = prev_comp_sets[:]   # type: ignore
            target_comp = next_set.pop(i)

            c1, c2 = target_comp.split()

            next_set.insert(i, c2)
            next_set.insert(i, c1)
            sets.append(next_set)

        return sets         # type: ignore
