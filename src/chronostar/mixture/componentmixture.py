from typing import Any, Callable, Tuple
import numpy as np
from numpy import float64
from numpy.typing import NDArray

from ..base import BaseComponent, BaseMixture
from .sklmixture import SKLComponentMixture


class ComponentMixture(BaseMixture):
    """A mixture model of arbitrary components

    Parameters
    ----------
    init_weights : NDArray[float64] of shape (n_components) or (n_samples, n_components)
        Initial weights of the components. If array is 2 dimensional, the
        init_weights are taken to be initial membership probabilities. If
        this is the case, you must configure `init_params='init_resp'`
    init_components : list[BaseComponent]
        Component objects which will be maximised to the data,
        optionally with pre-initialised parameters

    Attributes
    ----------
    tol : float, default 1e-3
        Used to determine convergence by sklearn's EM algorithm.
        Convergence determined if "change" between EM iterations is
        less than tol, where change is the difference between the
        average log probability of each sample, configurable
    reg_covar : float, default 1e-6
        A regularization constant added to the diagonals of
        covariance matrices, configurable
    max_iter : int, default 100
        The maximum iterations for sklearn's EM algorithm, configurable
    n_init : int, default 1
        (included only for sklearn API compatbility, ignored)
    init_params : str, default 'random'
        The initialization approach used by sklearn if component
        parameters aren't pre set, configurable. Must be one of

        - 'init_resp' : responsibilites are taken from input
        - 'kmeans' : responsibilities are initialized using kmeans.
        - 'k-means++' : use the k-means++ method to initialize.
        - 'random' : responsibilities are initialized randomly.
        - 'random_from_data' : initial means are randomly selected data points.

    random_state: int, default None
        Controls the random seed given to the method chosen to
        initialize the parameters (see init_params). In addition, it
        controls the generation of random samples from the fitted
        distribution. Pass an int for reproducible output across multiple
        function calls, configurable.
    warm_start: bool, default True
        (leave True for correct interactions between `self` and
        `self.sklmixture`)
    verbose: int, default 0
        Whether to print sklearn statements:

        - 0 : no output
        - 1 : prints current initialization and each iteration step
        - 2 : same as 1 but also prints log probability and execution time

    verbose_interval: int, default 10
        If `verbose > 0`, how many iterations between print statements
    """

    function_parser: dict[str, Callable] = {}

    # Configurable class attributes
    tol: float = 1e-3
    reg_covar: float = 1e-6
    max_iter: int = 100
    n_init: int = 1
    init_params: str = 'random'
    random_state: Any = None
    warm_start: bool = True
    verbose: int = 0
    verbose_interval: int = 10

    def __init__(
        self,
        init_weights: NDArray[float64],
        init_components: list[BaseComponent],
    ) -> None:

        # Can handle extra parameters if I want...
        self.sklmixture = SKLComponentMixture(
            init_weights,
            init_components,
            tol=self.tol,
            reg_covar=self.reg_covar,
            max_iter=self.max_iter,
            n_init=self.n_init,
            init_params=self.init_params,
            random_state=self.random_state,
            warm_start=self.warm_start,
            verbose=self.verbose,
            verbose_interval=self.verbose_interval,
        )

    def fit(self, X: NDArray[float64]) -> None:
        """Fit the mixture model to the input data

        Parameters
        ----------
        X : NDArray[float64] of shape (n_samples, n_features)
            Input data
        """
        print("--------------------------------------------------")
        print(f"Fitting {len(self.get_components())}-comp mixture")
        print("--------------------------------------------------")
        self.sklmixture.fit(X)

    def bic(self, X: NDArray[float64]) -> float:
        """Calculate the Bayesian Information Criterion

        Parameters
        ----------
        X : NDArray[float64] of shape (n_samples, n_features)
            Input data

        Returns
        -------
        float
            The calculated BIC
        """

        return float(self.sklmixture.bic(X))

    def set_parameters(
        self,
        params: Tuple[NDArray[float64], list[BaseComponent]],
    ) -> None:
        """Set the parameters that characterise the mixture

        Parameters
        ----------
        params: tuple[NDArray[float64], list[BaseComponent]]
            The weights of the components and the component objects
        """

        self.sklmixture._set_parameters(params)

    def get_parameters(self) -> tuple[NDArray[float64], list[BaseComponent]]:
        """Get the parameters that characterise the mixture

        Returns
        -------
        tuple[NDArray[float64], list[BaseComponent]]
            The weights of the components and the component objects
        """

        return self.sklmixture._get_parameters()

    def get_components(self) -> list[BaseComponent]:
        """Get the list of components fitted to the data

        Returns
        -------
        list[BaseComponent]
            The list of components
        """
        _, components = self.get_parameters()
        return components

    def estimate_membership_prob(self, X: NDArray[float64]):
        """Estimate the membership probabilities of each sample to
        each component

        This method assumes the mixture has already been fit with
        :meth:`fit`

        Parameters
        ----------
        X : NDArray[float64] of shape (n_samples, n_features)
            Input data

        Returns
        -------
        NDArray[float64] of shape (n_samples, n_components)
            The membership probabilities of each sample to each
            component.
        """
        weighted_log_prob = self.sklmixture._estimate_weighted_log_prob(X)

        # Take exponent
        weighted_prob = np.exp(weighted_log_prob)

        # Normalize such that each row sums to 1
        return (weighted_prob.T / weighted_prob.sum(axis=1)).T
