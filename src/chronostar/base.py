from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Callable, NamedTuple, Optional, Union, Type
import numpy as np
from numpy.typing import NDArray
from numpy import float64


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

    function_parser: dict[str, Callable] = {}       # type: ignore

    def __init__(self, params: Optional[NDArray[float64]] = None) -> None:
        # If ``parameters_set`` is false, components may get auto-initialized
        if params is not None:
            self.set_parameters(params)
            self.parameters_set = True
        else:
            self.parameters_set = False

    @classmethod
    def configure(cls, **kwargs) -> None:           # type: ignore
        """Set any configurable class attributes
        """
        for param, val in kwargs.items():
            if hasattr(cls, param):
                setattr(cls, param, val)
            else:
                print(f"[CONFIG]:{cls} unexpected config param: {param}={val}")

    @abstractmethod
    def estimate_log_prob(self, X: NDArray[float64]) -> NDArray[float64]:
        """Calculate the log probability of each sample in X given
        this components current estimated parameters

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data

        Returns
        -------
        ndarray of shape (n_samples)
            Log probabilities
        """
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
        """The number of parameters required to describe this component's
        model

        Necessary for calculating information criteria that depend on
        number of parameters, e.g. BIC or AIC

        Returns
        -------
        int
            How many parameters describe this model
        """
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
        """Get parameters

        Returns
        -------
        ndarray of shape (n_params)
            a single 1D array of the component parameters
        """
        pass

    @abstractmethod
    def set_parameters(self, params: NDArray[float64]) -> None:
        """Set parameters

        Parameters
        ----------
        params : ndarray of shape (n_params)
            a single 1D array of the component parameters
        """
        pass


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

    def __init__(
        self,
        init_weights: NDArray[float64],
        init_comps: tuple[BaseComponent, ...],
    ) -> None:
        self.init_comps = init_comps
        self.init_weights = init_weights

    @classmethod
    def configure(cls, **kwargs) -> None:            # type: ignore
        """Set any configurable class attributes
        """
        for param, val in kwargs.items():
            if hasattr(cls, param):
                setattr(cls, param, val)
            else:
                print(f"[CONFIG]:{cls} unexpected config param: {param}={val}")

    @abstractmethod
    def set_parameters(
        self,
        params: tuple[NDArray[float64], tuple[BaseComponent, ...]],
    ) -> None:
        """Set the parameters of the mixture model

        Parameters
        ----------
        params : tuple[NDArray[float64], tuple[BaseComponent, ...]]
            The weights of the components and a tuple of the components
        """
        pass

    @abstractmethod
    def get_parameters(self) -> tuple[NDArray[float64], tuple[BaseComponent, ...]]:
        """Get the parameters of the mixture model

        Returns
        -------
        tuple[NDArray[float64], tuple[BaseComponent, ...]]
            The weights of the components and a tuple of the components
        """
        pass

    @abstractmethod
    def fit(self, X: NDArray[float64]) -> None:
        """Fit the mixture model to the data

        Parameters
        ----------
        X : NDArray[float64] of shape (n_samples, n_features)
            Input data
        """
        pass

    @abstractmethod
    def bic(self, X: NDArray[float64]) -> float:
        """Calculate the Bayesian Information Critereon (BIC)

        Parameters
        ----------
        X : NDArray[float64] of shape (n_samples, n_features)
            Input data

        Returns
        -------
        float
            The BIC
        """
        pass

    @abstractmethod
    def get_components(self) -> tuple[BaseComponent, ...]:
        """Return the components of the mixture model

        Returns
        -------
        tuple[BaseComponent, ...]
            A tuple of the components
        """
        pass

    @abstractmethod
    def estimate_weighted_log_prob(self, X: NDArray[float64]) -> NDArray[float64]:
        """Estimate the weighted log-probabilities, log P(X | Z) + log weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        array, shape (n_samples, n_component)
            weighted_log_prob
        """
        pass

    def estimate_membership_prob(
        self, X: NDArray[float64]
    ) -> NDArray[float64]:
        """Estimate the membership probability of each star to each
        component

        Parameters
        ----------
        X : NDArray[float64] of shape (n_samples, n_features)
            Input data

        Returns
        -------
        NDArray[float64] of shape (n_samples, n_components)
            The membership probability of each star to each component.
            Each row should sum to 1, each column should average to the
            corresponding component's weight
        """
        weighted_log_prob = self.estimate_weighted_log_prob(X)

        # Take exponent
        weighted_prob = np.exp(weighted_log_prob)

        # Normalize such that each row sums to 1
        return np.transpose((weighted_prob.T / weighted_prob.sum(axis=1)))


class ScoredMixture(NamedTuple):
    """Simple named tuple for pairing mixtures with their scores

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


class BaseICPool(metaclass=ABCMeta):
    """A pool of sets of initial conditions, stored as a queue

    Parameters
    ----------
    component_class : Type[BaseComponent]
        A class derived from BaseComponent
    """

    def __init__(
        self,
        component_class: Type[BaseComponent],
        start_init_comps: Optional[tuple[BaseComponent, ...]] = None,
    ) -> None:
        self.component_class = component_class
        self.registry: dict[Union[str, int], ScoredMixture] = {}

    @classmethod
    def configure(cls, **kwargs) -> None:       # type: ignore
        """Set any configurable class attributes
        """
        for param, val in kwargs.items():
            if hasattr(cls, param):
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
        InitialCondition
            An initial condition: a tuple of components, paired with
            a unique, informative id
        """
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
        unique_id : str
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
