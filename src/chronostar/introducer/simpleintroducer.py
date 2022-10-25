from typing import Callable, Optional, Union

from ..base import BaseComponent, BaseIntroducer, InitialCondition


def convert_num2alpha(n: int) -> str:
    """Convert an integer count to a three "digit" character counter

    Parameters
    ----------
    n : int
        An integer value

    Returns
    -------
    str
        The input, converted to base 26, and represented in characters

    Examples
    --------
    >>> convert_num2alpha(0)
    'AAA'

    >>> convert_num2alpha(26)
    'ABA'
    """
    res: list[str] = []
    for order in range(1, 4):
        digit = n % 26
        res.insert(0, chr(ord('A') + digit))
        n //= 26

    return ''.join(res)


class SimpleIntroducer(BaseIntroducer):
    """An Introducer object that introduces components into
    previous sets and thereby generates new sets of initial
    conditions
    """

    # Put any configurable functions and their strings here
    function_parser: dict[str, Callable] = {}

    # Put configurable attributes here

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.generation = 0
        self.n_initconds = 0
        print(f"[SimpleIntroducer] After init {self.generation=}")

    def _generate_label(
        self,
        components: tuple[BaseComponent, ...],
        parent_label: str,
        extra: Optional[str] = None,
    ) -> str:
        """Generate a unique and informative label to be returned with
        an InitialCondition object

        For this implementation, the label looks like:
            ``[char_id]_[parents_char_id]_[ngeneration]_[ncomps]_[split_comp]``
        where char_id is the counter of of how many labels generated so far,
        converted to a 3 "digit" alphabet id, parents_char_id is the same
        thing but from the parent initial condition, ngeneration is the
        generation count so far, ncomps is the number of components,
        and split_comp says which component in the parent initial condition
        was split in order to produce this initial condition.

        Parameters
        ----------
        components : tuple of BaseComponent
            A collection of components soon to be returned as an initial condition
        parent_label : str
            The label of the "parent", the initial condition whose result
            was used to generate `components`
        extra : str, optional
            Any extra info to be appended to resulting label, by default None

        Returns
        -------
        str
            A label describing the provided set of components
        """
        this_id = convert_num2alpha(self.n_initconds)
        parent_id = parent_label.split('-')[0]

        self.n_initconds += 1

        label = f"{this_id}-{parent_id}-{self.generation:04}-{len(components):04}"
        if extra is not None:
            label = f"{label}-{extra}"

        return label

    def next_gen(
        self,
        prev_comp_sets:
            Union[list[InitialCondition], InitialCondition, None],
    ) -> list[InitialCondition]:
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
            label = self._generate_label(components, 'XXX', extra='auto')
            self.generation += 1
            return [InitialCondition(label, components)]

        if isinstance(prev_comp_sets, list):
            raise UserWarning("This Introducer only accepts one InitialCondition")

        next_initial_conditions: list[InitialCondition] = []
        for target_ix in range(len(prev_comp_sets.components)):
            # Perform a deep copy of components, using their class to
            # construct a replica
            next_ic_components = [
                c.__class__(c.get_parameters()) for c in prev_comp_sets.components
            ]

            # Replace the ith component by splitting it in two
            target_comp = next_ic_components.pop(target_ix)
            c1, c2 = target_comp.split()

            next_ic_components.insert(target_ix, c2)
            next_ic_components.insert(target_ix, c1)

            label = self._generate_label(
                components=tuple(next_ic_components),
                parent_label=prev_comp_sets.label,
                extra=str(target_ix),
            )
            # next_ic_components
            # next_ic_comps_tup: tuple[BaseComponent] = tuple(next_ic_components)

            next_initial_conditions.append(
                InitialCondition(label, tuple(next_ic_components))
            )

        self.generation += 1
        print(f"[SimpleIntroducer] After next_gen {self.generation=}")

        return next_initial_conditions
