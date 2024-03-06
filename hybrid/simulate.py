import numpy as np
from numpy.typing import ArrayLike
from typing import Callable

from .gillespie import GillespieSimulator
from .hybrid import HybridSimulator
#from .tau import TauLeapSimulator

SIMULATORS = {
    'haseltinerawlings': HybridSimulator,
    'gillespie': GillespieSimulator,
    #'tauleap': TauLeapSimulator,
}

def simulate(t_span: ArrayLike, y0: ArrayLike, k: Callable[[float], ArrayLike], N: ArrayLike, kinetic_order_matrix: ArrayLike, rng: np.random.Generator, t_eval: ArrayLike=None, method='haseltinerawlings', **simulator_kwargs):    
    simulator_klass = SIMULATORS[method]
    simulator = simulator_klass(k, N, kinetic_order_matrix, **simulator_kwargs)
    return simulator.simulate(t_span, y0, rng, t_eval)