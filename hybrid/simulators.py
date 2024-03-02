import numpy as np
from numpy.typing import ArrayLike
from typing import Callable, NamedTuple
from enum import IntEnum
from collections import Counter
from dataclasses import dataclass
from abc import ABC, abstractmethod, abstractclassmethod

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

@dataclass(frozen=True)
class History():
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
    def __init__(self, k, N, kinetic_order_matrix, jit=True, propensity_function=None) -> None:
        inhomogeneous = isinstance(k, Callable)
        if not inhomogeneous:
            k = np.asarray(k, dtype=float)
        self.inhomogeneous = inhomogeneous

        assert N.shape[0] == kinetic_order_matrix.shape[0], "N and kinetic_order_matrix should have # rows == # of species"
        assert N.shape[1] == kinetic_order_matrix.shape[1], "N and kinetic_order_matrix should have # columns == # of reaction pathways"

        self.k = k
        self.N = N
        self.kinetic_order_matrix = kinetic_order_matrix

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

    def simulate(self, t_span: ArrayLike, y0: ArrayLike, rng: np.random.Generator, t_eval: ArrayLike=None, **step_kwargs):
        y0 = np.asarray(y0)
        assert self.N.shape[0] == self.kinetic_order_matrix.shape[0] == y0.shape[0], "N and kinetic_order_matrix should have # rows == # of species"
        assert len(t_span) == 2
        t0, t_end = t_span

        if t_eval is None: t_eval = np.array([])

        run = self.initiate_run(t0, y0)

        t = t0
        while t < t_end:
            step = self.step(*run.current_state(), t_end, rng, t_eval, **step_kwargs)
            t = run.handle_step(step)

        return run.get_history()

    @abstractmethod
    def step(self, t, y, t_end, rng, t_eval):
        ...

    @abstractclassmethod
    def construct_propensity_function(cls, k, kinetic_order_matrix, jit=True):
        ...

class HybridSimulator(Simulator):
    pass