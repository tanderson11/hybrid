import numpy as np
from numpy.typing import ArrayLike
from typing import Callable, NamedTuple
from enum import IntEnum
from collections import Counter

class StepStatus(IntEnum):
    # simulators will subclass this to introduce other statuses.
    def was_rejected(self):
        return self < 0

class Run():
    def __init__(self, t0, y0, history_length=1e6) -> None:
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
        print(self.get_t(), self.history_index)
        return self.get_t()

    def get_history(self):
        t_history = self.t_history[:self.history_index+1]
        y_history = self.y_history[:,:self.history_index+1]
        return History(self.get_t(), self.get_y(), t_history, y_history, self.status_counter)

class History(NamedTuple):
    t: float
    y: ArrayLike
    t_history: ArrayLike
    y_history: ArrayLike
    status_counter: Counter

class Step(NamedTuple):
    t_history: ArrayLike
    y_history: ArrayLike
    status: StepStatus

class Simulator():
    run_klass = Run
    def __init__(self, k, N, kinetic_order_matrix, propensity_function=None) -> None:
        inhomogeneous = isinstance(k, Callable)
        if not inhomogeneous:
            k = np.asarray(k, dtype=float)
        self.inhomogeneous = inhomogeneous

        assert N.shape[0] == kinetic_order_matrix.shape[0], "N and kinetic_order_matrix should have # rows == # of species"
        assert N.shape[1] == kinetic_order_matrix.shape[1], "N and kinetic_order_matrix should have # columns == # of reaction pathways"

        self.k = k
        self.N = N
        self.kinetic_order_matrix = kinetic_order_matrix
        if propensity_function is not None:
            self.propensity_function = propensity_function
        else:
            self.propensity_function = self.construct_propensity_function(k, kinetic_order_matrix)

    def initiate_run(self, t0, y0):
        return self.run_klass(t0, y0)

    def simulate(self, t_span: ArrayLike, y0: ArrayLike, rng: np.random.Generator, t_eval: ArrayLike=None):
        assert self.N.shape[0] == self.kinetic_order_matrix.shape[0] == y0.shape[0], "N and kinetic_order_matrix should have # rows == # of species"
        assert len(t_span) == 2
        t0, t_end = t_span

        if t_eval is None: t_eval = np.array([])

        run = self.initiate_run(t0, y0)

        t = t0
        while t < t_end:
            step = self.step(*run.current_state(), t_end, rng, t_eval)
            t = run.handle_step(step)
            #break

        return run.get_history()

class HybridSimulator(Simulator):
    pass