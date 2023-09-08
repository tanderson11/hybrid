import numpy as np
from scipy.integrate import solve_ivp
from typing import NamedTuple
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable

@dataclass(frozen=True)
class SimulationOptions():
    approximate_rtot: bool = False
    batch_events: bool = False
    deterministic: bool = True
    round_randomly: bool = True

@dataclass(frozen=True)
class SimulationResult():
    t: float
    y: np.ndarray
    n_stochastic: int
    nfev: int

@dataclass(frozen=True)
class Partition():
    deterministic: np.ndarray
    stochastic: np.ndarray

def partition_by_threshold(propensities, threshold):
    stochastic = propensities <= threshold
    return Partition(~stochastic, stochastic)

def stochastic_event_finder(t, y_expanded, k_of_t,  N, rate_involvement_matrix, partition, hitting_point):
    stochastic_progress = y_expanded[-1]
    return stochastic_progress-hitting_point
stochastic_event_finder.terminal = True

def dydt(t, y_expanded, k_of_t, N, rate_involvement_matrix, partition, hitting_point):
    # by fiat the last entry of y will carry the integral of stochastic rates
    y = y_expanded[:-1]
    #print("y at start of dydt", y)
    propensities = calculate_propensities(y, k_of_t(t), rate_involvement_matrix)
    deterministic_propensities = propensities * partition.deterministic
    stochastic_propensities = propensities * partition.stochastic

    dydt = np.zeros_like(y_expanded)
    # each propensity feeds back into the stoich matrix to determine
    # overall rate of change in the state
    # https://en.wikipedia.org/wiki/Biochemical_systems_equation
    dydt[:-1] = N @ deterministic_propensities
    dydt[-1]  = sum(stochastic_propensities)
    #print("t", t, "y_expanded", y_expanded, "dydt", dydt)
    return dydt

def calculate_propensities(y, k, rate_involvement_matrix):
    # product along column in rate involvement matrix
    # with states raised to power of involvement
    # multiplied by rate constants == propensity
    # dimension of y is expanded to make it a column vector
    return np.prod(np.expand_dims(y, axis=1)**rate_involvement_matrix, axis=0) * k

class StepStatus(IntEnum):
    upper_limit = 0
    stochastic_event = 1
    other_terminal_event = 2

class StepUpdate(NamedTuple):
    t: float
    y: np.ndarray
    status: StepStatus
    nfev: int

def canonicalize_event(t_events, y_events):
    event_index = None
    for i,event_set in enumerate(t_events):
        if event_set is None:
            continue
        if event_index is not None:
            assert False, "that's unusual, we had two different kinds of terminal events in 1 step"
        event_index = i
    # now we have one canonical event that occurred
    t_event = t_events[event_index]
    assert len(t_event) == 1, "we had more than one of the same kind of terminal event!"
    t_event = t_event[0]
    y_event = y_events[event_index][0]

    return t_event, y_event, event_index

def round_with_method(y, round_randomly=False, rng=None):
    if round_randomly:
        # round down if random float is greater than decimal
        # round up otherwise
        rounded = np.round((rng.random(y.shape) <= (y - np.floor(y))) + y)
        return rounded
    else:
        return np.round(y)

def hybrid_step(
        y0: np.ndarray,
        t_span: list,
        partition: Partition,
        k_of_t: Callable[[float], np.ndarray],
        N: np.ndarray,
        rate_involvement_matrix: np.ndarray,
        rng: np.random.Generator,
        events: list[Callable[[np.ndarray], float]],
        simulation_options: SimulationOptions
        ) -> StepUpdate:
    """Integrates the partitioned system forward in time until reaching upper bound of integration or a stochastic event occurs.

    Args:
        y0 (np.ndarray): initial state.
        t_span (list): [t0, t_upper_limit].
        partition (Partition): partition.stochastic and partition.deterministic are masks for pathways.
        k_of_t (f: float -> np.ndarray or np.ndarray): either a callable that gives rate constants at time t or a list of unchanging rate constants.
        N (np.ndarray): the stoichiometry matrix for the system. N_ij = net change in i after unit progress in reaction j.
        rate_involvement_matrix (np.ndarray): A_ij = kinetic order for species i in reaction j.
        rng (np.random.Generator)
        events (list[Callable[[np.ndarray], float]]): a list of continuous functions of the state that have a 0 when an event of interest occurs.
        simulation_options (SimulationOptions): configuration of the simulation.

    Returns:
        StepUpdate: update object:
            update.t: time at end of step,
            update.y: state at end of step,
            update.status: -1 if integration error, 0 if upper limit, 1 if stoch event, 2 if other event,
            update.nfev: number of evaluations of derivative.
    """    
    event_flag = False
    t0, t_max = t_span

    # not really a hitting *time* as this is a dimensionless quantity,
    # a random number drawn from the unit exponential distribution.
    # when the integral of the stochastic rates == hitting_point, an event occurs
    hitting_point = rng.exponential(1)

    # a continuous event function that will record 0 when the hitting point is reached
    extra_events = events
    events = [stochastic_event_finder]
    events.extend(extra_events)
    # all our events should be terminal
    for e in events:
        assert e.terminal

    # integrate until hitting or until t_max
    y0_expanded = np.zeros(len(y0)+1)
    y0_expanded[:-1] = y0
    step_solved = solve_ivp(dydt, t_span, y0_expanded, events=events, args=(k_of_t, N, rate_involvement_matrix, partition, hitting_point))

    # if no event occurred, simply return the current values of t and y
    if step_solved.status == -1:
        assert False, "integration step failed"
    elif step_solved.status == 0:
        print("Reached upper limit of integration")
        y_last = step_solved.y[:,-1]
        # drop last entry: it is the rate integral
        y_last = y_last[:-1]
        y_last = round_with_method(y_last, simulation_options.round_randomly, rng)
        return StepUpdate(step_solved.t[-1], y_last, StepStatus.upper_limit, step_solved.nfev)
    
    #print("t_events:", step_solved.t_events)
    # if we reach here, an event has occured
    assert step_solved.status == 1
    # if an event occured, move to that time point
    # first ensure that our expectations are met: 1 event of 1 kind, because we insist on terminal events
    t, y, event_index = canonicalize_event(step_solved.t_events, step_solved.y_events)
    # we constructed the events list so that the 0th element is always the occurrence of a stochastic event
    if event_index == 0:
        event_flag = True

    # drop expanded term for some of rates
    y = y[:-1]
    # round
    y = round_with_method(y, simulation_options.round_randomly, rng)

    # if the event isn't a stochastic event, then we were halting to reassess partition
    # so: move to the update but don't adjudicate any events
    if not event_flag:
        return StepUpdate(t,y,StepStatus.other_terminal_event, step_solved.nfev)

    # if the event was a stochastic event, cause it to happen
    # first by determining which event happened
    endpoint_propensities = calculate_propensities(y, k_of_t(t), rate_involvement_matrix)

    # OPEN QUESTION: should we recalculate endpoint partition or should we use current partition?
    # I think probably not!
    #endpoint_partition = partition_function(endpoint_propensities)

    valid_selections = endpoint_propensities * partition.stochastic

    # TODO: fix if sum == 0 to have no division by 0
    valid_selections /= valid_selections.sum()
    valid_selections = valid_selections.cumsum()

    # the first entry that is greater than a random float is our event choice
    pathway_rand = rng.random()
    #print(valid_selections > pathway_rand)
    entry = np.argmax(valid_selections > pathway_rand)
    path_index = np.unravel_index(entry, valid_selections.shape)

    # our pathway updater converts an index into rates denoting which change happened
    # into an actual update step of our species quantities
    update = np.transpose(N[:,path_index])
    update = update.reshape(y.shape)
    y += update
    #print("stochastic event index", path_index, "update", update)
    #print(y)
    return StepUpdate(t,y,StepStatus.stochastic_event,step_solved.nfev)

def forward_time(y0: np.ndarray, t_span: list[float], partition_function: Callable[[np.ndarray], Partition], k_of_t: Callable[[float], np.ndarray], N: np.ndarray, rate_involvement_matrix: np.ndarray, rng: np.random.Generator, events=[], simulation_options=SimulationOptions()) -> SimulationResult:
    """Evolve system of irreversible reactions forward in time using hybrid deterministic-stochastic approximation.

    Args:
        y0 (np.ndarray): initial state
        t_span (list): [t0, t_upper_limit]
        partition_function (Callable[[np.ndarray], Partition]): function that takes rates at time t and outputs partition of system
        k_of_t (f: float -> np.ndarray or np.ndarray): either a callable that gives rate constants at time t or a list of unchanging rate constants
        N (np.ndarray): the stoichiometry matrix for the system. N_ij = net change in i after unit progress in reaction j
        rate_involvement_matrix (np.ndarray): A_ij = kinetic order for species i in reaction j.
        rng (np.random.Generator)
        events (list[Callable[[np.ndarray], float]], optional): a list of continuous functions of the state that have a 0 when an event of interest occurs. Defaults to [].
        simulation_options (SimulationOptions): configuration of the simulation. Defaults to SimulationOptions().

    Returns:
        SimulationResult: simulation result object:
            result.t: t_end, should be approximately equal to t_span[-1],
            result.y: system state at t_end,
            result.n_stochastic: number of stochastic events that occured,
            result.nfev: number of evaluations of the derivative.

    """    
    n_stochastic = 0
    nfev = 0
    t0, t_end = t_span
    t = t0
    y = y0
    while t < t_end:
        propensities = calculate_propensities(y, k_of_t(t), rate_involvement_matrix)
        partition = partition_function(propensities)
        step_update = hybrid_step(y, [t, t_end], partition, k_of_t, N, rate_involvement_matrix, rng, events, simulation_options)
        t = step_update.t
        y = step_update.y
        if step_update.status == StepStatus.stochastic_event:
            n_stochastic += 1
        nfev += step_update.nfev
    return SimulationResult(t,y,n_stochastic,nfev)