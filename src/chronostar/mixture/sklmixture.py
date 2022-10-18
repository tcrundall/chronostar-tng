from typing import Any, Optional
from numpy.typing import NDArray
from numpy import float64
import numpy as np
from sklearn import cluster
from sklearn.cluster import kmeans_plusplus
from sklearn.mixture._base import BaseMixture as SKLBaseMixture

from ..base import BaseComponent


class SKLComponentMixture(SKLBaseMixture):
    """A derived class utilising much from scikit-learn
    to fit a Gaussian Mixture Model

    Parameters
    ----------
    weights_init : NDArray[float64] of shape (n_components)
        Initial weights of each component, ideally normalized
        to sum to 1. If array is 2 dimensional, the
        init_weights are taken to be initial membership probabilities.
        If this is the case, you must configure
        `init_params='init_resp'`
    components_init : list[BaseComponent]
        Component objects which will be maximised to the data,
        optionally with pre-initialised parameters
    tol : float, optional
        Some tolerance used by sklearn... , by default 1e-3
    reg_covar : float, optional
        Regularisation factor added to diagonal elements of
        covariance matrices, by default 1e-6
    max_iter : int, optional
        Maximum iterations of EM algorithm, by default 100
    n_init : int, optional
        sklearn parameter we don't use, by default 1
    init_params : str, optional
        How to initialise components if not already set,
        'random' assigns memberships randomly then maximises,
        by default 'random'. Options are:

        - 'random': each responsibility is randomly assigned
        - 'init_resp': an initial responsiblity array is provided

    random_state : Any, optional
        sklearn parameter... the random seed?, by default None
    warm_start : bool, optional
        sklearn parameter that we don't use, by default True
    verbose : int, optional
        sklearn parameter..., by default 0
    verbose_interval : int, optional
        sklearn parameter, by default 10
    """

    def __init__(
        self,
        weights_init: NDArray[float64],
        components_init: tuple[BaseComponent, ...],
        *,
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: str = 'random',
        random_state: Any = None,
        warm_start: bool = True,
        verbose: int = 0,
        verbose_interval: int = 10,
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__(
            n_components=len(components_init),
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            random_state=random_state,
            warm_start=warm_start,
            verbose=verbose,
            verbose_interval=verbose_interval,
        )

        self.components_: tuple[BaseComponent, ...] = components_init
        self.init_resp: Optional[NDArray[float64]] = None
        self.m_step_count = 0

        # Check if weights was used to provide membership responsibilities
        if len(weights_init.shape) == 2:
            self.init_resp = weights_init
            self.weights_ = weights_init.sum(axis=0)
            if self.init_params != 'init_resp':
                print("Warning! You provided membership responsibilities"
                      f" but {self.init_params=}. Should be 'init_resp'"
                      " otherwise init membership responsibilities are ignored")
        else:
            self.weights_ = weights_init

        # If components all have attributes, then assume they came from
        # previously converged fit, so hack some SKL parameters to
        # avoid reinitialization.
        # SKL only initializes parameters if not hasattr(self, "converged_")
        # so we set converged here to avoid that
        if all(c.parameters_set for c in self.components_):
            print("All components have set parameters, so skipping"
                  " skl initialization")
            self.converged_ = False
            self.lower_bound_ = -np.inf

        if kwargs:
            print("Extra arguments provided...")

    def _check_parameters(self, X: NDArray[float64]) -> None:
        """Override to do nothing

        Parameters
        ----------
        X : NDArray[float64]
            data
        """
        pass

    def _initialize_parameters(self, X, random_state):
        """Initialize the model parameters.
        Parameters
        ----------
        X : array-like of shape  (n_samples, n_features)
        random_state : RandomState
            A random number generator instance that controls the random seed
            used for the method chosen to initialize the parameters.

        References
        ----------
        Copy pasted from scikit-learn
        TODO: Add explicit author credits
        """
        n_samples, _ = X.shape
        print("[SKLMixture._initialize_parameters]: Initializing parameters!")
        print(f"[SKLMixture._initialize_parameters]: {self.init_params=}")

        if self.init_params == "init_resp":
            if self.init_resp is None:
                raise UserWarning(
                    "init_params is 'init_resp' was set, "
                    "so init_resp cannot be None"
                )
            resp = self.init_resp
            assert resp is not None
        elif self.init_params == "kmeans":
            resp = np.zeros((n_samples, self.n_components))
            label = (
                cluster.KMeans(
                    n_clusters=self.n_components,
                    n_init=1,
                    random_state=random_state,
                )
                .fit(X)
                .labels_
            )
            resp[np.arange(n_samples), label] = 1
        elif self.init_params == "random":
            resp = random_state.uniform(size=(n_samples, self.n_components))
            resp /= resp.sum(axis=1)[:, np.newaxis]
        elif self.init_params == "random_from_data":
            resp = np.zeros((n_samples, self.n_components))
            indices = random_state.choice(
                n_samples, size=self.n_components, replace=False
            )
            resp[indices, np.arange(self.n_components)] = 1
        elif self.init_params == "k-means++":
            resp = np.zeros((n_samples, self.n_components))
            _, indices = kmeans_plusplus(
                X,
                self.n_components,
                random_state=random_state,
            )
            resp[indices, np.arange(self.n_components)] = 1
        else:
            raise ValueError(
                "Unimplemented initialization method '%s'" % self.init_params
            )

        self._initialize(X, resp)

    def _initialize(
        self,
        X: NDArray[float64],
        resp: NDArray[float64]
    ) -> None:
        """Initialize component parameters if not already set

        Parameters
        ----------
        X : NDArray[float64] of shape (n_samples, n_features)
            Input data
        resp : NDArray[float64] of shape (n_samples, n_components)
            Responsibilities (membership probabilities) of each sample
            to each component

        TODO: Actually only initialise randomly when "random" is set
        """

        # We can expect `resp` to be initialized. So:
        for i, component in enumerate(self.components_):
            component.maximize(X, resp[:, i])

        # However, the components may already have their parameters set...
        # then we shouldn't even be here.

    def _m_step(
        self,
        X: NDArray[float64],
        log_resp: NDArray[float64],
    ) -> None:
        """EM's M-step, maximise model parameters based on
        data and estimated responsibilities

        Parameters
        ----------
        X : NDArray[float64] of shape (n_samples, n_features)
            Input data
        log_resp : NDArray[float64] of shape (n_samples, n_components)
            Log responsibilities (membership probabilities) of each sample
            to each component
        """

        resp = np.exp(log_resp)
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        self.weights_ = nk
        self.weights_ /= self.weights_.sum()

        print(f"--- M_STEP {self.m_step_count:03} ---")
        print(f"   - weights {self.weights_}")
        for i, component in enumerate(self.components_):
            print(f"   - comp {i:02}")
            component.maximize(X=X, resp=resp[:, i])

        self.m_step_count += 1

    def _estimate_log_prob(self, X: NDArray[float64]) -> NDArray[float64]:
        """Estimate the log probability of each sample for each
        component

        Parameters
        ----------
        X : NDArray[float64] of shape (n_samples, n_features)
            Input data

        Returns
        -------
        NDArray[float64] of shape (n_samples, n_components)
            Log probabilities
        """
        n_samples = X.shape[0]
        n_components = len(self.components_)

        log_probs = np.zeros((n_samples, n_components))
        for k, component in enumerate(self.components_):
            log_probs[:, k] = component.estimate_log_prob(X)

        return log_probs

    def _estimate_log_weights(self) -> NDArray[float64]:
        """Estimate the log of the component weights

        Returns
        -------
        NDArray[float64] of shape (ncomponents)
            The log of the component weights
        """
        return np.log(self.weights_)

    def _compute_lower_bound(self, _: Any, log_prob_norm: Any) -> Any:
        """Used internally by sklearn

        Parameters
        ----------
        _ : Any
            Argument only here to match API
        log_prob_norm : Any
            Probably a float

        Returns
        -------
        Any
            Same as input
        """
        return log_prob_norm

    def _get_parameters(
        self,
    ) -> tuple[NDArray[float64], tuple[BaseComponent, ...]]:
        """Get the parameters that characterise the mixture

        Returns
        -------
        tuple[NDArray[float64], list[BaseComponent]]
            The weights of the components and the component objects
        """

        return (self.weights_, self.components_)

    def _set_parameters(
        self,
        params: tuple[NDArray[float64], tuple[BaseComponent, ...]],
    ) -> None:
        """Set the parameters that characterise the mixture

        Parameters
        ----------
        params: tuple[NDArray[float64], list[BaseComponent]]
            The weights of the components and the component objects
        """
        (self.weights_, self.components_) = params
        self.n_components = len(self.components_)

    def _n_parameters(self) -> int:
        """Get the number of parameters used to characterise the
        mixture

        Used for BIC and AIC calculations

        Returns
        -------
        int
            The number of parameters used to characterise the
            mixture
        """
        n_params = 0
        for component in self.components_:
            n_params += component.n_params
        n_params += len(self.weights_) - 1
        return int(n_params)

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
        return float(
            -2 * self.score(X) * X.shape[0]
            + self._n_parameters() * np.log(X.shape[0])
        )

    def aic(self, X: NDArray[float64]) -> float:
        """Calculate the Akaike Information Criterion

        Parameters
        ----------
        X : NDArray[float64] of shape (n_samples, n_features)
            Input data

        Returns
        -------
        float
            The calculated AIC
        """
        return float(
            -2 * self.score(X) * X.shape[0]
            + 2 * self._n_parameters()
        )
