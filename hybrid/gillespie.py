import numpy as np
from typing import Callable, NamedTuple
from numba import jit
from scipy.integrate import quad
from scipy.optimize import fsolve

def jit_calculate_propensities_factory(kinetic_order_matrix):
    # assumes constant rate constants
    @jit(nopython=True)
    def jit_calculate_propensities(y, k):
        # Remember, we want total number of distinct combinations * k === rate.
        # we want to calculate (y_i rate_involvement_ij) (binomial coefficient)
        # for each species i and each reaction j
        # sadly, inside a numba C function, we can't avail ourselves of scipy's binom,
        # so we write this little calculator ourselves
        intensity_power = np.zeros_like(kinetic_order_matrix)
        for i in range(0, kinetic_order_matrix.shape[0]):
            for j in range(0, kinetic_order_matrix.shape[1]):
                if y[i] < kinetic_order_matrix[i][j]:
                    intensity_power[i][j] = 0.0
                elif y[i] == kinetic_order_matrix[i][j]:
                    intensity_power[i][j] = 1.0
                else:
                    intensity = 1.0
                    for x in range(0, kinetic_order_matrix[i][j]):
                        intensity *= (y[i] - x) / (x+1)
                    intensity_power[i][j] = intensity
        product_down_columns = np.ones(len(k))
        for i in range(0, len(y)):
            product_down_columns = product_down_columns * intensity_power[i]
        return product_down_columns * k
    return jit_calculate_propensities

class SimulationResult(NamedTuple):
    t: float
    y: np.ndarray
    t_history: np.ndarray
    y_history: np.ndarray
    n_stochastic: int

def forward_time(y0: np.ndarray, t_span: list[float], k: np.ndarray, N: np.ndarray, kinetic_order_matrix: np.ndarray, rng: np.random.Generator, **kwargs):
    calculate_propensities = jit_calculate_propensities_factory(kinetic_order_matrix)

    t,t_end = t_span
    y = y0

    history_length = int(1e6)
    t_history = np.zeros(history_length)
    y_history = np.zeros((len(y), history_length))
    history_index = 0
    n_stochastic = 0
    while t < t_end:
        step_update = homogeneous_gillespie_step(y, k, t, t_end, N, calculate_propensities, rng)
        t += step_update.t_update
        y += step_update.y_update

        t_history[history_index] = t
        y_history[:, history_index] = y
        history_index += 1
        n_stochastic += step_update.was_stochastic_event

    t_history = t_history[:history_index]
    y_history = y_history[:,:history_index]

    return SimulationResult(t_history[-1], y_history[:, -1], t_history, y_history, n_stochastic)

class StepUpdate(NamedTuple):
    t_update: float
    y_update: np.ndarray
    was_stochastic_event: bool

class HittingTimeProposal(NamedTuple):
    tau: float
    propensities: np.ndarray = None
    total_propensity: float = None

def find_hitting_time_inhomogenous(y, k, t, calculate_propensities, rng):
    hitting_point = rng.exponential(1)
    f = inhomogeneous_upper_bound_f_factory(t, hitting_point, calculate_propensities)
    hitting_time = fsolve(f, x0=1)
    return HittingTimeProposal(hitting_time)

def find_hitting_time_homogeneous(propensities, total_propensity, rng):
    hitting_point = rng.exponential(1)
    hitting_time = hitting_point / total_propensity
    return HittingTimeProposal(hitting_time, propensities, total_propensity)

def inhomogeneous_upper_bound_f_factory(t, hitting_point, calculate_propensities):
    # we need to solve for the hitting time
    # which is the time when hitting_point / integral of propensities
    # so we build an objective function that takes a zero when x = hitting_point / integral
    # using numerical minimization we can minimize the objective function to find the time x
    def objective_function(x):
        integral = quad(calculate_propensities, t, x)[0]
        return np.abs(x - hitting_point / integral)

def gillespie_update_proposal(N, propensities, total_propensity, rng):
    selections = propensities.cumsum() / total_propensity
    pathway_rand = rng.random()
    entry = np.argmax(selections > pathway_rand)
    path_index = np.unravel_index(entry, selections.shape)

    # N_ij = net change in i after unit progress in reaction j
    # so the appropriate column of the stoich matrix tells us how to do our update
    update = np.transpose(N[:,path_index])
    update = update.reshape((N.shape[0],))
    return update

def gillespie_step(y, k, t, t_end, N, calculate_propensities, rng):
    hitting_time = find_hitting_time_inhomogenous(y, k, t, calculate_propensities, rng).tau

    endpoint_propensities = calculate_propensities(y, k, hitting_time)
    total_propensity = np.sum(endpoint_propensities)

    update = gillespie_update_proposal(N, endpoint_propensities, total_propensity, rng)

    return StepUpdate(hitting_time, update, was_stochastic_event=True)

def homogeneous_gillespie_step(y, k, t, t_end, N, calculate_propensities, rng):
    propensities = calculate_propensities(y, k)
    total_propensity = np.sum(propensities)
    time_proposal = find_hitting_time_homogeneous(propensities, total_propensity, rng)
    hitting_time, propensities, total_propensity = time_proposal

    if t + hitting_time > t_end:
        return StepUpdate(t_end - t, np.zeros_like(y), was_stochastic_event=False)

    update = gillespie_update_proposal(N, propensities, total_propensity, rng)

    return StepUpdate(hitting_time, update, was_stochastic_event=True)

