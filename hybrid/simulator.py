from typing import List, Callable, NamedTuple
from enum import IntEnum, auto
from collections import Counter
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Union
import pandas as pd

import numpy as np
from numpy.typing import ArrayLike

class StepStatus(IntEnum):
    # simulators will subclass this to introduce other statuses.
    def was_rejected(self):
        return self < 0

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
    step_indices: ArrayLike
    status_history: ArrayLike
    pathway_history: ArrayLike
    status_counter: Counter
    pathway_counter: Counter
    step_t_history: ArrayLike
    step_y_history: ArrayLike

    def plot(self, legend, ax=None, **plot_kwargs):
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.subplot()
        ax.plot(self.t_history, self.y_history.T, **plot_kwargs)
        ax.legend(legend)
        return ax

    def restricted_values(self, t, decimal_places=6):
        t_restrict = pd.Series(t).round(decimals=decimal_places)
        t_series = pd.Series(self.t_history).round(decimals=decimal_places)
        mask = (t_series.isin(t_restrict) | (self.t_history == self.t)) & ~t_series.duplicated()
        return self.t_history[mask], self.y_history[:, mask]

class Run():
    def __init__(self, t0, y0, history_length=1e6) -> None:
        y0 = np.asarray(y0)
        history_length = int(history_length)
        self.history_index = 0
        self.step_index = 0

        self.t_history = np.zeros(history_length)
        self.y_history = np.zeros((y0.shape[0], history_length))

        self.step_indices = np.zeros(history_length)
        self.status_history = np.zeros(history_length)
        self.pathway_history = np.zeros(history_length, dtype=int)
        self.step_t_history = np.zeros(history_length)
        self.step_y_history = np.zeros((y0.shape[0], history_length))

        self.status_counter = Counter({})

        self.t_history[0]    = t0
        self.y_history[:, 0] = y0
        self.status_history[0] = None
        self.pathway_history[0] = -1

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

        # for t and y history, we add all the data contained in the step object
        n_samples = len(step.t_history)
        if self.history_index+n_samples+1 > len(self.t_history):
            raise ValueError("exceeded history length")
        self.t_history[self.history_index+1:self.history_index+n_samples+1] = step.t_history
        self.y_history[:, self.history_index+1:self.history_index+n_samples+1] = step.y_history
        self.history_index += n_samples

        # for other histories, we are only recording the fact of 1 step rather than n subsamples
        self.status_history[self.step_index+1] = step.status
        self.pathway_history[self.step_index+1] = step.pathway
        self.step_indices[self.step_index+1] = self.history_index
        self.step_t_history[self.step_index+1] = step.t_history[-1]
        self.step_y_history[:, self.step_index+1] = step.y_history[:, -1]
        self.step_index += 1

        if (self.get_y() < 0).any():
            assert False, "found negative specimen value"

        return self.get_t()

    def get_history(self):
        t_history = self.t_history[:self.history_index+1]
        y_history = self.y_history[:,:self.history_index+1]

        step_indices = self.step_indices[:self.step_index+1]
        status_history = self.status_history[:self.step_index+1]
        pathway_history = self.pathway_history[:self.step_index+1]
        step_t_history = self.step_t_history[:self.step_index+1]
        step_y_history = self.step_y_history[:, :self.step_index+1]
        return History(self.get_t(), self.get_y(), t_history, y_history, step_indices, status_history, pathway_history, self.status_counter, Counter(pathway_history), step_t_history, step_y_history)

    def get_step_kwargs(self):
        return {}

class DummyStatus(StepStatus):
    failure = -1
    t_end = 0
    t_end_for_discontinuity = auto()

def get_updated(ntuple, updates):
    tuple_dict = ntuple._asdict()
    tuple_dict.update(updates)
    return ntuple.__class__(**tuple_dict)

class Step(NamedTuple):
    t_history: ArrayLike
    y_history: ArrayLike
    status: StepStatus
    pathway: int = -1

class Simulator(ABC):
    run_klass = Run
    status_klass = DummyStatus
    def __init__(self, k: Union[ArrayLike, Callable], N: ArrayLike, kinetic_order_matrix: ArrayLike, poisson_products_mask: ArrayLike=None, discontinuities: ArrayLike=None, jit: bool=True, propensity_function: Callable=None, species_labels=None, pathway_labels=None) -> None:
        """Initialize a simulator equipped to simulate a specific model forward in time with different parameters and initial conditions.

        Parameters
        ----------
        k : ArrayLike | Callable
            Either a vector of unchanging rate constants or a function of time that returns a vector of rate constants.
        N : ArrayLike
            The stoichiometry matrix N such that N_ij is the change in species `i` after unit progress in reaction `j`.
        kinetic_order_matrix : ArrayLike
            The kinetic order matrix such that the _ij entry is the kinetic intensity of species i in reaction j.
        poisson_products_mask : ArrayLike, optional
            [DEVELOPMENT FEATURE] A mask with shape equal to the # of pathways. True denotes a reaction whose products will be determined by random draws
            from a Poisson distribution with mean given by the corresponding element in the stoichiometry matrix. By default None.
        discontinuities : ArrayLike, optional
            An array of time coordinates where the time dependent rate constants are discontinuous. Simulation will smartly avoid
            walking over discontinuities and will instead stop at each one. By default None.
        jit : bool, optional
            If True, use numba.jit(nopython=True) to construct a low level callable (fast) version of simulation helper functions, by default True.
        propensity_function : Callable or None, optional
            If not None, use the specified function a_ij(t,y) to calculate the propensities of reactions at time t and state y.
            Specify this if there is a fast means of calculating propensities or if propensities do not obey standard kinetic laws, by default None.
        species_labels : List[str] or None, optional
            A set of strings that matches the shape of y and provides human readable information detailing the system's species. By default None.
        pathway_labels : List[str] or None, optional
            A set of strings that matches the shape of k and provides human readable information detailing the system's reactions. By default None.
        """
        inhomogeneous = isinstance(k, Callable)
        if not inhomogeneous:
            k = np.asarray(k, dtype=float)
        self.inhomogeneous = inhomogeneous

        assert N.shape[0] == kinetic_order_matrix.shape[0], "N and kinetic_order_matrix should have # rows == # of species"
        assert N.shape[1] == kinetic_order_matrix.shape[1], "N and kinetic_order_matrix should have # columns == # of reaction pathways"

        self.k = k
        self.N = N.astype(float)
        self.Nplus  = N * (N > 0)
        self.Nminus = N * (N < 0)
        self.kinetic_order_matrix = kinetic_order_matrix.astype(float)

        self.poisson_products = poisson_products_mask is None
        self.poisson_products_mask = poisson_products_mask

        self.discontinuities = discontinuities if discontinuities is not None else []

        self.jit = jit
        if propensity_function is not None:
            self.propensity_function = propensity_function
            if jit:
                from numba.core.registry import CPUDispatcher
                assert(isinstance(propensity_function, CPUDispatcher))
        else:
            self.propensity_function = self.construct_propensity_function(k, kinetic_order_matrix, inhomogeneous, jit=jit)

        self.species_labels = np.array(species_labels)
        self.pathway_labels = np.array(pathway_labels)

    @classmethod
    def from_model(cls, m, *args, reaction_to_k=None, parameters=None, jit: bool=True, **kwargs):
        """Build a simulator from a `reactionmodel.Model` object. Additional arguments are passed to the simulator's constructor."""
        return cls(m.get_k(reaction_to_k=reaction_to_k, parameters=parameters, jit=jit), m.stoichiometry(), m.kinetic_order(), *args, species_labels=[s.name for s in m.species], pathway_labels=[r.description for r in m.all_reactions], jit=jit, **kwargs)

    def initiate_run(self, t0, y0, history_length=1e6):
        return self.run_klass(t0, y0, history_length=history_length)

    def run_simulations(self, n_trials: int, t_span: ArrayLike, y0: ArrayLike, rng: List[np.random.Generator], end_routine: Callable=None, t_eval: ArrayLike=None, halt: Callable=None, **step_kwargs):
        """Simulate the reaction manifold many times.

        Parameters
        ----------
        n_trials : int
            Number of times to run the experiment.
        t_span : ArrayLike
            The times `(t0, t_end)` between which to simulate as a tuple.
        rng : List[np.random.Generator]
            Either a single random generator, or a list of random generators with length == n_trials.
        y0 : ArrayLike
            A vector y_i of the quantity of species i at time 0.
        end_routine : Callable, optional
            Function with signature Result => Object that is called at the end of each experiment to extract the wanted information.
            Defaults to None, in which case the full result is returned.
        halt : Callable, optional
            A function with signature halt(t, y) => bool evaulated each step that stops execution on a return of True.
            If None, always simulate to t_end. Defaults to None.

        Returns
        -------
        History
            A list of artifacts, one for each trial, where each artifact is given by `end_routine(result_i)`.
        """
        if end_routine is None:
            end_routine = lambda x: x
        single_generator = isinstance(rng, np.random.Generator)
        if not single_generator: assert len(rng) == n_trials
        artifacts = []
        for i in range(n_trials):
            _rng = rng if single_generator else rng[i]

            result = self.simulate(t_span, y0, _rng, t_eval, halt, **step_kwargs)
            artifacts.append(end_routine(result))

        return artifacts

    def simulate(self, t_span: ArrayLike, y0: ArrayLike, rng: np.random.Generator, t_eval: ArrayLike=None, halt: Callable=None, history_length=1e6, **step_kwargs) -> History:
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
        halt : Callable, optional
            A function with signature halt(t, y) => bool evaulated each step that stops execution on a return of True.
            If None, always simulate to t_end. Defaults to None.

        Returns
        -------
        History
            The results of the run with attributes `t`, `y` (the time and state of the system at t_end),
            `t_history` and `y_history` (all time and states evaluated), and `status_counter`,
            which records the kinds of termination that occurred during each step of simulation.
        """
        if isinstance(y0, dict):
            raise ValueError(f"y0 should be an Array or array-like (e.g. list). Not {type(y0)}")
        y0 = np.asarray(y0)

        if (y0.round() != y0).any():
            print("WARNING: initial condition is not on the integer lattice, rounding.")
            y0 = y0.round()
        assert self.N.shape[0] == self.kinetic_order_matrix.shape[0] == y0.shape[0], "N and kinetic_order_matrix should have # rows == # of species"
        assert len(t_span) == 2
        t0, t_end = t_span

        if t_eval is None: t_eval = np.array([])

        run = self.initiate_run(t0, y0, history_length=history_length)

        t = t0
        while t < t_end:
            discontinuity_flag = False
            discontinuity_mask = np.asarray(self.discontinuities) <= t
            # (no discontinuities)
            if len(self.discontinuities) == 0:
                next_discontinuity = np.inf
            else:
                discontinuity_index = np.argmin(discontinuity_mask)
                # (all discontinuities passed because our mininmum satisfying value is True, so all values are true)
                if discontinuity_mask[discontinuity_index]:
                    next_discontinuity = np.inf
                else:
                    next_discontinuity = self.discontinuities[discontinuity_index]

            if next_discontinuity < t_end:
                discontinuity_flag = True

            t_end_step = min(t_end, np.nextafter(next_discontinuity, 0))

            step = self.step(*run.current_state(), t_end_step, rng, t_eval, **step_kwargs, **run.get_step_kwargs())

            if discontinuity_flag and step.status == self.status_klass.t_end:
                print(f"Doing surgery to avoid discontinuity: skipping from {step.t_history[-1]} to {np.nextafter(next_discontinuity, t_end)}")
                step = get_updated(step, {'status':self.status_klass.t_end_for_discontinuity})
                step.t_history[-1] = np.nextafter(next_discontinuity, t_end)

            t = run.handle_step(step)

            if np.isclose(t, t_end):
                t = t_end
            if halt is not None and halt(run.get_t(), run.get_y()):
                break

        return run.get_history()

    @abstractmethod
    def step(self, t, y, t_end, rng, t_eval):
        ...

    @classmethod
    @abstractmethod
    def construct_propensity_function(cls, k, kinetic_order_matrix, inhomogeneous, jit=True):
        ...