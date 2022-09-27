from typing import Union

from .base import BaseIntroducer
from ..component.base import BaseComponent, Splittable


class SimpleIntroducer(BaseIntroducer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def next_gen(
        self,
        prev_comp_sets: Union[list[BaseComponent], None],
    ) -> list[list[BaseComponent]]:
        if prev_comp_sets is None:
            return [[self.component_class(self.config_params)]]

        sets = []
        for i in range(len(prev_comp_sets)):
            next_set = prev_comp_sets[:]
            target_comp = next_set.pop(i)

            assert isinstance(target_comp, Splittable)
            c1, c2 = target_comp.split()

            assert isinstance(c1, BaseComponent)
            assert isinstance(c2, BaseComponent)
            next_set.insert(i, c2)
            next_set.insert(i, c1)
            sets.append(next_set)
        return sets
