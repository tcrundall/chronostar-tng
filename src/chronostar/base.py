from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Callable, NamedTuple, Optional, Union, Type, Any
# import numpy as np
from numpy.typing import NDArray
from numpy import float64


class ScoredMixture(NamedTuple):
    """Simple dataclass for pairing mixtures with their scores

    Parameters
    ----------
    mixture : BaseMixture
        A mixture model whose fit method has been called
    score : float
        The score of the mixture model
    """
    mixture: BaseMixture
    score: float
    label: str


class InitialCondition(NamedTuple):
    """Simple named tuple for pairing an informative label with
    a list of initial components

    Parameters
    ----------
    label : str or int
        unique identifer combined with extra information
    components : list[BaseComponent]
        A list of components that can initialise a mixture fit
    """
    label: str
    components: tuple[BaseComponent, ...]


class BaseICPool(metaclass=ABCMeta):
    """A pool of sets of initial conditions, stored as a queue

    Parameters
    ----------
    introducer_class : Type[BaseIntroducer]
        A class derived from BaseIntroducer, this determines how
        new sets of initial conditions are generated
    component_class : Type[BaseComponent]
        A class derived from BaseComponent
    """
    function_parser: dict[str, Callable] = {}

    def __init__(
        self,
        introducer_class: Type[BaseIntroducer],
        component_class: Type[BaseComponent],
    ) -> None:
        self.introducer_class = introducer_class
        self.component_class = component_class

        self.registry: dict[Union[str, int], ScoredMixture] = {}

    @classmethod
    def configure(cls, **kwargs) -> None:
        """Set any conofigurable class attributes
        """
        for param, val in kwargs.items():
            if hasattr(cls, param):
                if val in cls.function_parser:
                    setattr(cls, param, cls.function_parser[val])
                else:
                    setattr(cls, param, val)
            else:
                print(f"[CONFIG]:{cls} unexpected config param: {param}={val}")

    @abstractmethod
    def has_next(self) -> bool:
        """Determine if internal queue is non-empty, after attempting to
        repopulate as needed.

        Returns
        -------
        bool
            Whether the queue is non-empty
        """
        pass

    @abstractmethod
    def get_next(self) -> InitialCondition:
        """Pop the next initial conditions of internal queue and
        return it with a unique identifier

        Returns
        -------
        tuple[Union[str, int], list[BaseComponent]]
            (unique_id, set of initialc conditions)
        """
        pass

    @abstractmethod
    def provide_start(self, init_conds: InitialCondition) -> None:
        pass

    @abstractmethod
    def register_result(
        self,
        unique_id: str,
        mixture: BaseMixture,
        score: float
    ) -> None:
        """Register a finished mixture fit with its score

        Parameters
        ----------
        unique_id : Union[str, int]
            Unique identifier that was provided along with initial conditions
        mixture : BaseMixture
            A mixture model that has been fit
        score : float
            The score of the mixture model
        """
        pass

    @property
    @abstractmethod
    def best_mixture(self) -> BaseMixture:
        """Get the best scoring mixture

        Returns
        -------
        BaseMixture
            The best scoring mixture

        Note
        ----
        Perhaps an extension is to keep track of the top N best mixtures
        """
        pass


class BaseIntroducer(metaclass=ABCMeta):
    """A class repsonsible for constructing new sets of
    initial conditions by introducing components

    Parameters
    ----------
    component_class : Type[BaseComponent]
        A derived class from BaseComponent
    """

    function_parser: dict[str, Callable] = {}

    def __init__(
        self,
        component_class: Type[BaseComponent],
    ) -> None:
        self.component_class = component_class

    @classmethod
    def configure(cls, **kwargs) -> None:
        """Set any conofigurable class attributes
        """
        for param, val in kwargs.items():
            if hasattr(cls, param):
                if val in cls.function_parser:
                    setattr(cls, param, cls.function_parser[val])
                else:
                    setattr(cls, param, val)
            else:
                print(f"[CONFIG]:{cls} unexpected config param: {param}={val}")

    @abstractmethod
    def next_gen(
        self,
        prev_comp_sets: Union[
            list[InitialCondition],
            InitialCondition,
            None
        ],
    ) -> list[InitialCondition]:
        """Generate the next "generation" of runs, as sets of
        initial conditions

        Parameters
        ----------
        prev_comp_sets : Union[ list[list[BaseComponent]], list[BaseComponent], None ]
            Takes either multiple previous fits, one previous fit, or none,
            where a previous fit is a list of components

        Returns
        -------
        list[list[BaseComponent]]
            Returns sets of initial conditions as a list, where each set of initial
            conditions is a list of components.
        """
        pass


class BaseComponent(metaclass=ABCMeta):
    """Abstract class for a (assumed-to-be Gaussian)
    component to be used in a mixture model

    Capable of fitting itself to a set of samples and
    responsibilities (membership probabilities)

    Parameters
    ----------
    params : ndarray of shape(n_params), optional
        The model parameters, as a 1 dimensional arra
    """

    function_parser: dict[str, Callable] = {}

    def __init__(self, params: Optional[NDArray[float64]] = None) -> None:
        if params is not None:
            self.set_parameters(params)
            self.parameters_set = True
        else:
            self.parameters_set = False

    @classmethod
    def configure(cls, **kwargs) -> None:
        """Set any conofigurable class attributes
        """
        for param, val in kwargs.items():
            if hasattr(cls, param):
                if val in cls.function_parser:
                    setattr(cls, param, cls.function_parser[val])
                else:
                    setattr(cls, param, val)
            else:
                print(f"[CONFIG]:{cls} unexpected config param: {param}={val}")

    @abstractmethod
    def estimate_log_prob(self, X: NDArray[float64]) -> NDArray[float64]:
        pass

    @abstractmethod
    def maximize(
        self,
        X: NDArray[float64],
        resp: NDArray[float64],
    ) -> None:
        """Maximize the model parameters on a set of data and
        responsibilities

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data
        resp : ndarray of shape (n_samples, n_components)
            Responsibilities (or membership probabilities) of each
            sample to each component
        """
        pass

    @property
    @abstractmethod
    def n_params(self) -> int:
        pass

    @abstractmethod
    def split(self) -> tuple[BaseComponent, BaseComponent]:
        """Split this component into two by some means.

        Popular approach is to split along the primary axis

        Returns
        -------
        tuple[BaseComponent, BaseComponent]
            Two components with identical parameters except
            half as wide and offset from mean
        """
        pass

    @abstractmethod
    def get_parameters(self) -> NDArray[float64]:
        pass

    @abstractmethod
    def set_parameters(self, params: NDArray[float64]) -> None:
        pass


class BaseMixture(metaclass=ABCMeta):
    """A Mixture model (e.g. Gaussian Mixture Model) consisting
    of components

    Parameters
    ----------
    init_weights : ndarray of shape(n_components) or (n_samples, n_components)
        The initial weight of components, ideally normalized such that
        sums to 1. If `init_weights` is 2D the it is interpreted as
        initial membership probabilities
    init_comps : list[BaseComponent]
        A list of component objects, which may optionally already have
        pre-set parameters.
    """

    function_parser: dict[str, Callable] = {}

    def __init__(
        self,
        init_weights: NDArray[float64],
        init_comps: tuple[BaseComponent, ...],
    ) -> None:
        self.init_comps = init_comps
        self.init_weights = init_weights

    @classmethod
    def configure(cls, **kwargs) -> None:
        """Set any conofigurable class attributes
        """
        for param, val in kwargs.items():
            if hasattr(cls, param):
                if val in cls.function_parser:
                    setattr(cls, param, cls.function_parser[val])
                else:
                    setattr(cls, param, val)
            else:
                print(f"[CONFIG]:{cls} unexpected config param: {param}={val}")

    @abstractmethod
    def set_parameters(
        self,
        params: tuple[NDArray[float64], tuple[BaseComponent, ...]],
    ) -> None:
        pass

    @abstractmethod
    def get_parameters(self) -> Any:
        pass

    @abstractmethod
    def fit(self, X: NDArray[float64]) -> None:
        pass

    @abstractmethod
    def bic(self, X: NDArray[float64]) -> float:
        pass

    @abstractmethod
    def get_components(self) -> tuple[BaseComponent, ...]:
        pass

    @abstractmethod
    def estimate_membership_prob(
        self, X: NDArray[float64]
    ) -> NDArray[float64]:
        pass
