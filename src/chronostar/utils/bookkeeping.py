from typing import Any, Optional


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


def generate_label(
    n_initconds: int,
    n_generations: int,
    components: tuple[Any, ...],
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
    this_id = convert_num2alpha(n_initconds)
    parent_id = parent_label.split('-')[0]

    # self.n_initconds += 1

    label = f"{this_id}-{parent_id}-{n_generations:04}-{len(components):04}"
    if extra is not None:
        label = f"{label}-{extra}"

    return label
