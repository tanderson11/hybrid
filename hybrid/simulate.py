import numpy as np
from numpy.typing import ArrayLike
from typing import Callable
from dataclasses import dataclass

class Run():
    def __init__(self, t0, y0, history_length=1e7) -> None:
        self.history_index = 0
        self.n_rejections  = 0
        
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
        t = self.get_t()
        if step.rejected:
            self.n_rejections += 1
            return t

        # handle a real update
        self.history_index += 1
        self.t_history[self.history_index]    = step.t
        self.y_history[:, self.history_index] = step.y
        return t

    def get_history(self):
        return 

class Simulator():
    run_klass = Run
    def __init__(self, k, N, kinetic_order_matrix) -> None:
        inhomogeneous = isinstance(k, Callable)
        if not inhomogeneous:
            k = np.asarray(k, dtype=float)
        self.inhomogeneous = inhomogeneous

        assert N.shape[0] == kinetic_order_matrix.shape[0], "N and kinetic_order_matrix should have # rows == # of species"
        assert N.shape[1] == kinetic_order_matrix.shape[1], "N and kinetic_order_matrix should have # columns == # of reaction pathways"

        self.k = k
        self.N = N
        self.kinetic_order_matrix = kinetic_order_matrix

    def initiate_run(self, t0, y0):
        return self.run_klass(t0, y0)

    def simulate(self, t_span: ArrayLike, y0: ArrayLike, rng: np.random.Generator, t_eval: ArrayLike=None):
        assert self.N.shape[0] == self.kinetic_order_matrix.shape[0] == y0.shape[0], "N and kinetic_order_matrix should have # rows == # of species"
        assert len(t_span) == 2
        t0, t_end = t_span

        run = self.initiate_run(t0, y0)

        t = t0
        while t < t_end:
            step = self.step(run.current_state(), rng, t_eval)
            t = run.handle_step(step)

        return run.get_history()

class HybridSimulator(Simulator):
    pass

SIMULATORS = {
    'haseltinerawlings': HybridSimulator,
    'gillespie': GillespieSimulator,
    'tauleap': TauLeapSimulator,
}

def simulate(t_span: ArrayLike, y0: ArrayLike, k: Callable[[float], ArrayLike], N: ArrayLike, kinetic_order_matrix: ArrayLike, rng: np.random.Generator, t_eval: ArrayLike=None, method='haseltinerawlings', **simulator_kwargs):    
    simulator_klass = SIMULATORS[method]
    simulator = simulator_klass(k, N, kinetic_order_matrix, **simulator_kwargs)
    return simulator.simulate(t_span, y0, rng, t_eval)