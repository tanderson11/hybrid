import numpy as np
from numpy.typing import ArrayLike
from typing import Callable

from hybrid.gillespie import GillespieSimulator
from hybrid.hybrid import HybridSimulator
from hybrid.tau import TauLeapSimulator

SIMULATORS = {
    'gillespie': GillespieSimulator,
    'haseltinerawlings': HybridSimulator,
    'tauleap': TauLeapSimulator,
}

def simulate(t_span: ArrayLike, y0: ArrayLike, k: Callable[[float], ArrayLike], N: ArrayLike, kinetic_order_matrix: ArrayLike, rng: np.random.Generator, t_eval: ArrayLike=None, halt=None, method='tau', **simulator_kwargs):
    """Simulate a system of reactions over a span of time given an initial state using `method`.

    Parameters
    ----------
    t_span : ArrayLike
        A tuple of times `(t0, t_end)` to simulate between.
    y0 : ArrayLike
        A vector y_i of the quantity of species i at time 0.
    k : ArrayLike | Callable
        Either a vector of unchanging rate constants or a function of time that returns a vector of rate constants.
    N : ArrayLike
        The stoichiometry matrix N such that N_ij is the change in species `i` after unit progress in reaction `j`.
    kinetic_order_matrix : ArrayLike
        The kinetic order matrix such that the _ij entry is the kinetic intensity of species i in reaction j.
    rng : np.random.Generator
        The random number generator to use for all random numbers needed during simulation.
    t_eval : ArrayLike, optional
        A vector of time points at which to evaluate the system and return in the final results.
        If None, evaluate at points chosen by the simulator, by default None.
    halt : Callable, optional
        A function with signature halt(t, y) => bool evaulated each step that stops execution on a return of True.
        If None, always simulate to t_end. Defaults to None.
    method : str, optional
        The method to use for simulation. Options include 'gillespie', 'haseltinerawlings', and 'tau', by default 'tau'.
    **simulator_kwargs
        Options that are passed to the specified simulator class. To see valid configurations, inspect the class that you are using.

    Returns
    -------
    History
        The results of the run with attributes `t`, `y` (the time and state of the system at t_end),
        `t_history` and `y_history` (all time and states evaluated), and `status_counter`,
        which records the kinds of termination that occurred during each step of simulation.
    """
    simulator_klass = SIMULATORS[method]
    simulator = simulator_klass(k, N, kinetic_order_matrix, **simulator_kwargs)
    return simulator.simulate(t_span, y0, rng, t_eval, halt=halt)