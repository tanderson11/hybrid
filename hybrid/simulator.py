from typing import Callable, NamedTuple
from enum import IntEnum
from collections import Counter
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from numpy.typing import ArrayLike

class StepStatus(IntEnum):
    # simulators will subclass this to introduce other statuses.
    def was_rejected(self):
        return self < 0

class Run():
    def __init__(self, t0, y0, history_length=1e6) -> None:
        y0 = np.asarray(y0)
        history_length = int(history_length)
        self.history_index = 0

        self.status_counter = Counter()

        self.t_history = np.zeros(history_length)
        self.y_history = np.zeros((y0.shape[0], history_length))

        self.t_history[0]    = t0
        self.y_history[:, 0] = y0

    def get_t(self):
        return self.t_history[self.history_index]

    def get_y(self):
        return self.y_history[:, self.history_index]

    def current_state(self):
        return self.get_t(), self.get_y()

    def handle_step(self, step):
        self.status_counter.update({step.status:1})
        if step.status.was_rejected():
            return self.get_t()

        n_samples = len(step.t_history)
        self.t_history[self.history_index+1:self.history_index+n_samples+1] = step.t_history
        self.y_history[:, self.history_index+1:self.history_index+n_samples+1] = step.y_history
        self.history_index += n_samples
        return self.get_t()

    def get_history(self):
        t_history = self.t_history[:self.history_index+1]
        y_history = self.y_history[:,:self.history_index+1]
        return History(self.get_t(), self.get_y(), t_history, y_history, self.status_counter)

    def get_step_kwargs(self):
        return {}

@dataclass(frozen=True)
class History():
    """The result of one simulation.

    Attributes
    ----------
    t: float
        The time at the end of simulation.
    y: ArrayLike
        The state vector at t. `y_i` is the quantity of the `i`th species at time `t`.
    t_history: ArrayLike
        The vector of all times where the state was recorded.
    y_history: ArrayLike
        An array of state vectors where the `i`th state vector corresponds to the `i`th entry in `t_history`.
    status_counter: Counter
        A counter object that records all the status of the simulator at the end of each simulation step.
    """
    t: float
    y: ArrayLike
    t_history: ArrayLike
    y_history: ArrayLike
    status_counter: Counter

class Step(NamedTuple):
    t_history: ArrayLike
    y_history: ArrayLike
    status: StepStatus

class Simulator(ABC):
    run_klass = Run
    def __init__(self, k: Union[ArrayLike, Callable], N: ArrayLike, kinetic_order_matrix: ArrayLike, jit: bool=True, propensity_function: Callable=None) -> None:
        """Initialize a simulator equipped to simulate a specific model forward in time with different parameters and initial conditions.

        Parameters
        ----------
        k : ArrayLike | Callable
            Either a vector of unchanging rate constants or a function of time that returns a vector of rate constants.
        N : ArrayLike
            The stoichiometry matrix N such that N_ij is the change in species `i` after unit progress in reaction `j`.
        kinetic_order_matrix : ArrayLike
            The kinetic order matrix such that the _ij entry is the kinetic intensity of species i in reaction j.
        jit : bool, optional
            If True, use numba.jit(nopython=True) to construct a low level callable (fast) version of simulation helper functions, by default True.
        propensity_function : Callable or None, optional
            If not None, use the specified function a_ij(t,y) to calculate the propensities of reactions at time t and state y.
            Specify this if there is a fast means of calculating propensities or if propensities do not obey standard kinetic laws, by default None.
        """
        inhomogeneous = isinstance(k, Callable)
        if not inhomogeneous:
            k = np.asarray(k, dtype=float)
        self.inhomogeneous = inhomogeneous

        assert N.shape[0] == kinetic_order_matrix.shape[0], "N and kinetic_order_matrix should have # rows == # of species"
        assert N.shape[1] == kinetic_order_matrix.shape[1], "N and kinetic_order_matrix should have # columns == # of reaction pathways"

        self.k = k
        self.N = N.astype(float)
        self.kinetic_order_matrix = kinetic_order_matrix.astype(float)

        self.jit = jit
        if propensity_function is not None:
            self.propensity_function = propensity_function
            if jit:
                from numba.core.registry import CPUDispatcher
                assert(isinstance(propensity_function, CPUDispatcher))
        else:
            self.propensity_function = self.construct_propensity_function(k, kinetic_order_matrix, inhomogeneous, jit=jit)


    def initiate_run(self, t0, y0):
        return self.run_klass(t0, y0)

    def simulate(self, t_span: ArrayLike, y0: ArrayLike, rng: np.random.Generator, t_eval: ArrayLike=None, **step_kwargs) -> History:
        """Simulate the reaction manifold between two time points given a starting state.

        Parameters
        ----------
        t_span : ArrayLike
            A tuple of times `(t0, t_end)` to simulate between.
        y0 : ArrayLike
            A vector y_i of the quantity of species i at time 0.
        rng : np.random.Generator
            The random number generator to use for all random numbers needed during simulation.
        t_eval : ArrayLike, optional
            A vector of time points at which to evaluate the system and return in the final results.
            If None, evaluate at points chosen by the simulator, by default None.

        Returns
        -------
        History
            The results of the run with attributes `t`, `y` (the time and state of the system at t_end),
            `t_history` and `y_history` (all time and states evaluated), and `status_counter`,
            which records the kinds of termination that occurred during each step of simulation.
        """
        y0 = np.asarray(y0)
        assert self.N.shape[0] == self.kinetic_order_matrix.shape[0] == y0.shape[0], "N and kinetic_order_matrix should have # rows == # of species"
        assert len(t_span) == 2
        t0, t_end = t_span

        if t_eval is None: t_eval = np.array([])

        run = self.initiate_run(t0, y0)

        t = t0
        while t < t_end:
            step = self.step(*run.current_state(), t_end, rng, t_eval, **step_kwargs, **run.get_step_kwargs())
            t = run.handle_step(step)

        return run.get_history()

    @abstractmethod
    def step(self, t, y, t_end, rng, t_eval):
        ...

    @classmethod
    @abstractmethod
    def construct_propensity_function(cls, k, kinetic_order_matrix, inhomogeneous, jit=True):
        ...

class HybridSimulator(Simulator):
    pass