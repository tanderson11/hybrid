from typing import NamedTuple, Callable
from dataclasses import dataclass
from enum import auto
import json
from typing import Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp
import scipy.special
import numba

from .simulator import Simulator, StepStatus

class HybridStepStatus(StepStatus):
    failure = -1
    t_end = 0
    stochastic_event = auto()
    t_end_for_discontinuity = auto()
    partition_change = auto()
    contrived_no_reaction = auto()
    user_event = auto()

    def event_like(self):
        return self > HybridStepStatus.t_end

class StepUpdate(NamedTuple):
    t_history: np.ndarray
    y_history: np.ndarray
    status: HybridStepStatus
    nfev: int

@dataclass
class PartitionScheme():
    def save(self, file):
        dictionary = {'name': type(self).__name__}
        dictionary.update(self.__dict__.copy())

        with open(file, 'w') as f:
            json.dump(dictionary, f)

@dataclass
class FixedThresholdPartitioner(PartitionScheme):
    threshold: float

    def partition_function(self, y, propensities):
        stochastic = (propensities <= self.threshold) & (propensities != 0)
        return Partition(~stochastic, stochastic, np.full_like(propensities, self.threshold))

@dataclass(frozen=True)
class Partition():
    deterministic: np.ndarray
    stochastic: np.ndarray
    # array of values for what threshold was used to partition each propensity
    propensity_thresholds: np.ndarray = None

def stochastic_event_finder(t, y_expanded, partition, propensity_function, hitting_point):
    stochastic_progress = y_expanded[-1]
    return stochastic_progress-hitting_point
stochastic_event_finder.terminal = True

def partition_change_finder_factory(partition_fraction_for_halt):
    def partition_change_finder(t, y_expanded, partition, propensity_function, hitting_point):
        y = y_expanded[:-1]
        propensities = propensity_function(t, y)
        distance_to_switch = np.where(
            partition.deterministic,
            propensities - partition.propensity_thresholds * partition_fraction_for_halt,
            np.inf
        )
        return distance_to_switch
    partition_change_finder.terminal  = True
    partition_change_finder.direction = -1

    return partition_change_finder

class HybridSimulator(Simulator):
    def __init__(self, k: Union[Callable, ArrayLike], N: ArrayLike, kinetic_order_matrix: ArrayLike, partition_function: Union[Callable, PartitionScheme], discontinuities: ArrayLike=None, jit: bool=True, propensity_function: Callable=None, dydt_function: Callable=None, **kwargs) -> None:
        """Initialize a Haseltine Rawlings simulator equipped to simulate a specific model forward in time with different parameters and initial conditions.

        Parameters
        ----------
        k : ArrayLike | Callable
            Either a vector of unchanging rate constants or a function of time that returns a vector of rate constants.
        N : ArrayLike
            The stoichiometry matrix N such that N_ij is the change in species `i` after unit progress in reaction `j`.
        kinetic_order_matrix : ArrayLike
            The kinetic order matrix such that the _ij entry is the kinetic intensity of species i in reaction j.
        partition_function : Callable | PartitionScheme
            A function p(propensities) that returns a Partition object with attributes `deterministic`
            (a mask for which pathways to consider as deterministic), `stochastic` (a mask for which pathways
            to consider as stochastic), and `propensity_thresholds` (the value that each propensity was compared to
            in order to partition it). In the original Haseltine-Rawlings algorithm, this function compares
            each propensity to the fixed threshold of `100.0` events per unit time. To retrieve this behavior,
            pass the object `FixedThresholdPartitioner(100.0)`.
        discontinuities : ArrayLike, optional
            A vector of time points that correspond to discontinuities in the function k(t). Providing these points
            while allow the simulator to avoid integrating over a discontinuity, by default None.
        jit : bool, optional
            If True, use numba.jit(nopython=True) to construct a low level callable (fast) version of simulation helper functions, by default True.
        propensity_function : Callable or None, optional
            If not None, use the specified function a_ij(t,y) to calculate the propensities of reactions at time t and state y.
            Specify this if there is a fast means of calculating propensities or if propensities do not obey standard kinetic laws, by default None.
        dydt_function : Callable, optional
            If not None, use the specified function a_ij(t,y) to calculate the derivative of the system at time t and state y.
            Specify this if there is a fast means of calculating the derivative, by default None.
        **kwargs : HybridSimulationOptions
            A valid selection of options specific to the Haseltine-Rawlings hybrid simulation algorithm.
            To see documentation of each option, inspect the class HybridSimulationOptions.

        Raises
        ------
        TypeError
            If `k` is specified as a string, possibly due to the lazy evaluation of parameters in the reactionmodel package.
        """
        if isinstance(k, str):
            raise TypeError(f"Instead of a function or matrix, found this message for k: {k}")
        super().__init__(k, N, kinetic_order_matrix, jit, propensity_function)
        if isinstance(partition_function, PartitionScheme):
            partition_function = partition_function.partition_function
        self.partition_function = partition_function
        self.discontinuities = discontinuities if discontinuities is not None else []
        self.simulation_options = HybridSimulationOptions(**kwargs)
        if dydt_function is None:
            self.dydt = self.construct_dydt_function(N, jit)
        else:
            self.dydt = dydt_function
            if self.jit:
                from numba.core.registry import CPUDispatcher
                assert(isinstance(propensity_function, CPUDispatcher))

    @classmethod
    def construct_dydt_function(cls, N, jit=True):
        if jit:
            return cls.jit_dydt_factory(N)
        return cls.dydt_factory(N)

    @staticmethod
    def jit_dydt_factory(N):
        @numba.jit(nopython=True)
        def jit_dydt(t, y_expanded, deterministic_mask, stochastic_mask, propensity_function, hitting_point):
            # by fiat the last entry of y will carry the integral of stochastic rates
            y = y_expanded[:-1]

            propensities = propensity_function(t, y)
            deterministic_propensities = propensities * deterministic_mask
            stochastic_propensities = propensities * stochastic_mask

            # each propensity feeds back into the stoich matrix to determine
            # overall rate of change in the state
            # https://en.wikipedia.org/wiki/Biochemical_systems_equation
            rates = N @ deterministic_propensities
            sum_stochastic = np.sum(stochastic_propensities)

            dydt = np.zeros_like(y_expanded)
            dydt[:-1] = rates
            dydt[-1]  = sum_stochastic
            return dydt

        # need this wrapper to access the properties of partition in nopython mode
        def wrapper(t, y_expanded, partition, propensity_function, hitting_point):
            return jit_dydt(t, y_expanded, partition.deterministic, partition.stochastic, propensity_function, hitting_point)
        return wrapper

    @staticmethod
    def dydt_factory(N):
        def dydt(t, y_expanded, partition, propensity_function, hitting_point):
            # by fiat the last entry of y will carry the integral of stochastic rates
            y = y_expanded[:-1]
            #print("y at start of dydt", y)
            propensities = propensity_function(t, y)
            deterministic_propensities = propensities * partition.deterministic
            stochastic_propensities = propensities * partition.stochastic

            dydt = np.zeros_like(y_expanded)
            # each propensity feeds back into the stoich matrix to determine
            # overall rate of change in the state
            # https://en.wikipedia.org/wiki/Biochemical_systems_equation
            dydt[:-1] = N @ deterministic_propensities
            dydt[-1]  = np.sum(stochastic_propensities)
            #print("t", t, "y_expanded", y_expanded, "dydt", dydt)
            return dydt
        return dydt

    @classmethod
    def construct_propensity_function(cls, k, kinetic_order_matrix, inhomogeneous, jit=True):
        if jit:
            return cls.jit_propensity_factory(k, kinetic_order_matrix, inhomogeneous)
        return cls.propensity_factory(k, kinetic_order_matrix, inhomogeneous)

    @staticmethod
    def propensity_factory(k, kinetic_order_matrix, inhomogeneous):
        def calculate_propensities(t, y):
            if inhomogeneous:
                k_of_t = k(t)
            else:
                k_of_t = k
            # product along column in rate involvement matrix
            # with states raised to power of involvement
            # multiplied by rate constants == propensity
            # dimension of y is expanded to make it a column vector
            return np.prod(scipy.special.binom(np.expand_dims(y, axis=1), kinetic_order_matrix), axis=0) * k_of_t
        return calculate_propensities

    @staticmethod
    def jit_propensity_factory(k, kinetic_order_matrix, inhomogeneous):
        @numba.jit(nopython=True)
        def jit_calculate_propensities(t, y):
            # Remember, we want total number of distinct combinations * k === rate.
            # we want to calculate (y_i kinetic_order_ij) (binomial coefficient)
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

            # then we take the product down the columns (so product over each reaction)
            # and multiply that output by the vector of rate constants
            # to get the propensity of each reaction at time t
            if inhomogeneous:
                k_of_t = k(t)
            else:
                k_of_t = k

            product_down_columns = np.ones(len(k_of_t))
            for i in range(0, len(y)):
                product_down_columns = product_down_columns * intensity_power[i]
            return product_down_columns * k_of_t
        return jit_calculate_propensities

    def step(self, t, y, t_end, rng, t_eval, events=None):
        if events is None: events = []
        """Integrates the partitioned system forward in time until the upper bound of integration is reached or a stochastic event occurs."""
        starting_propensities = self.propensity_function(t, y)
        partition = self.partition_function(y, starting_propensities)

        t_span = [t, t_end]

        discontinuity_flag = False

        discontinuity_mask = np.asarray(self.discontinuities) <= t
        # (no discontinuities)
        if len(self.discontinuities) == 0:
            next_discontinuity = np.inf
        else:
            discontinuity_index = np.argmin(discontinuity_mask)
            # (all discontinuities passed because  our mininmum satisfying value is True, so all values are true)
            if discontinuity_mask[discontinuity_index]:
                next_discontinuity = np.inf
            else:
                next_discontinuity = self.discontinuities[discontinuity_index]

        if next_discontinuity < t_end:
            discontinuity_flag = True
            # integrate only until the last float before the discontinuity
            #print(f"Reducing t_end {t_span[-1]} => {np.nextafter(next_discontinuity, t)}. Discontinuity_index={discontinuity_index}")
            t_span[-1] = np.nextafter(next_discontinuity, t)

        # not really a hitting *time* as this is a dimensionless quantity,
        # a random number drawn from the unit exponential distribution.
        # when the integral of the stochastic rates == hitting_point, an event occurs
        hitting_point = rng.exponential(1)

        event_type_cutoffs = []
        event_types = []
        user_events = events
        # if we approximate the stochastic propensities as constant between events,
        # we know the exact hitting time, otherwise we have to stop at an event
        if self.simulation_options.approximate_rtot:
            events = []
            total_stochastic_event_rate = np.sum(starting_propensities[partition.stochastic]) + self.simulation_options.contrived_no_reaction_rate
            hitting_time = t + hitting_point / total_stochastic_event_rate
            if hitting_time < t_span[-1]:
                #print(f"Removing discontinuity flag: t_end {t_span[-1]} => {hitting_time}")
                t_span[-1] = hitting_time
                # if discontinuity_flag was set, the upper limit was a discontinuity
                # but now that we've lowered the upper limit, that is no longer the case
                discontinuity_flag = False
        else:
            # a continuous event function that will record 0 when the hitting point is reached
            events = [stochastic_event_finder]
            event_type_cutoffs = [0]
            event_types = [HybridStepStatus.stochastic_event]

        if self.simulation_options.halt_on_partition_change:
            partition_change_finder = partition_change_finder_factory(self.simulation_options.partition_fraction_for_halt)
            events.append(partition_change_finder)
            event_type_cutoffs.append(self.kinetic_order_matrix.shape[1])
            event_types.append(HybridStepStatus.partition_change)

        if len(user_events) > 0:
            events.extend(user_events)
            event_type_cutoffs.append(len(events)-1)
            event_types.append(HybridStepStatus.user_event)
        # all our events should be terminal
        for e in events:
            assert e.terminal

        # we have an extra entry in our state vector to carry the integral of the rates of stochastic events, which dictates when an event fires
        y_expanded = np.zeros(len(y)+1)
        y_expanded[:-1] = y

        # we can't call solve_ivp with all the t_eval, we need to call it with only those time points
        # that lie between our start and intended upper limit of integration
        if len(t_eval) > 0:
            relevant_t_eval = t_eval[(t_eval > t) & (t_eval < t_span[-1])]
            relevant_t_eval = list(relevant_t_eval)
            relevant_t_eval.append(t_span[-1])
        else:
            # t_eval will be np.array([]), we demote it to None so that solve_ivp returns normal evaluations
            relevant_t_eval = None

        # integrate until hitting or until t_max
        step_solved = solve_ivp(self.dydt, t_span, y_expanded, events=events, args=(partition, self.propensity_function, hitting_point), t_eval=relevant_t_eval)
        ivp_step_status = step_solved.status
        t_history = step_solved.t
        # if we use t_eval, we will sometimes have no points returned except in events, so we need to dodge that error in indexing
        if len(step_solved.y) > 0:
            # slice to -1 to drop the integral of the rates, which is slotted into the last entry of y
            y_history = step_solved.y[:-1,:]
        else:
            y_history = np.array([])

        # this branching logic tells us what caused integration to stop
        if ivp_step_status == -1:
            status = HybridStepStatus.failure
            #print(step_solved)
            assert False, "integration step failed"
        elif ivp_step_status == 0:
            # if we are approximating the stochastic rates as constant between events, then hitting the upper limit
            # corresponds to a stochastic event, unless the upper limit is also the true end of the integration
            if not discontinuity_flag and self.simulation_options.approximate_rtot and t_end > t_span[-1]:
                status = HybridStepStatus.stochastic_event
                for t_event in step_solved.t_events:
                    assert(len(t_event) == 0)
            else:
                status = HybridStepStatus.t_end
        else:
            # if we reach here, an event has occured
            assert ivp_step_status == 1
            # ensure that our expectations are met: 1 event of 1 kind, because we insist on terminal events
            t_event, y_event, event_index = canonicalize_event(step_solved.t_events, step_solved.y_events)
            # drop expanded term for sum of rates
            y_event = y_event[:-1]

            # add event state to our history if its absent
            # (it might be absent if t_eval is set)
            if len(t_history) == 0:
                t_history = np.array([t_event])
                y_history = np.expand_dims(y_event, axis=1)
            elif t_event != t_history[-1]:
                t_history = np.concatenate([t_history, [t_event]])
                y_history = np.concatenate([y_history, np.expand_dims(y_event, axis=1)], axis=1)


            # use the event_index to find out what our status is (answer, what kind of event was this?)
            #print(event_type_cutoffs, event_types)
            for cutoff, event_status in zip(event_type_cutoffs, event_types):
                if event_index <= cutoff:
                    status = event_status
                    break
            else: # nobreak
                assert False, f"Couldn't assign a status to event. Event index = {event_index}. event_cutoffs={event_type_cutoffs}. event_types={event_types}."

        if status == HybridStepStatus.partition_change:
            print(f"Stopping for partition change. t={t_event}")

        # round the species quantities at our final time point
        y_history[:, -1] = round_with_method(y_history[:, -1], self.simulation_options.round_randomly, rng)

        # if we reached the true upper limit of integration simply return the current values of t and y
        if status == HybridStepStatus.t_end:
            if discontinuity_flag:
                status = HybridStepStatus.t_end_for_discontinuity
                print(f"Doing surgery to avoid discontinuity: skipping from {t_history[-1]} to {np.nextafter(next_discontinuity, t_end)}")
                t_history[-1] = np.nextafter(next_discontinuity, t_end)
            ## FIRST RETURN
            return StepUpdate(t_history, y_history, status, step_solved.nfev)

        assert HybridStepStatus.event_like(status)

        # if the event isn't a stochastic event, then we were halting to reassess partition
        # so: we simply return the state of the system at the time of our event
        if not status == HybridStepStatus.stochastic_event:
            return StepUpdate(t_history, y_history, status, step_solved.nfev)
        assert status == HybridStepStatus.stochastic_event

        # if the event was a stochastic event, cause it to happen
        # first by determining which event happened
        endpoint_propensities = self.propensity_function(t, y)

        # OPEN QUESTION: should we recalculate endpoint partition or should we use current partition?
        # I think we want to use starting partition but endpoint propensities!

        valid_selections = endpoint_propensities * partition.stochastic
        # if we have a contrived rate of no reaction, insert it into our array of transition probabilities as the last element
        if self.simulation_options.approximate_rtot:
            valid_selections = np.hstack([valid_selections, np.array([self.simulation_options.contrived_no_reaction_rate])])

        # TODO: fix if sum == 0 to have no division by 0. WHY DOES THIS HAPPEN?
        selection_sum = valid_selections.sum()
        assert selection_sum != 0.
        valid_selections /= selection_sum
        valid_selections = valid_selections.cumsum()

        # the first entry that is greater than a random float is our event choice
        pathway_rand = rng.random()
        entry = np.argmax(valid_selections > pathway_rand)
        path_index = np.unravel_index(entry, valid_selections.shape)

        # don't apply any update if our selection was the contrived rate of no reaction
        if self.simulation_options.approximate_rtot and np.squeeze(path_index) == len(valid_selections)-1:
            return StepUpdate(t_history, y_history, HybridStepStatus.contrived_no_reaction, step_solved.nfev)

        # N_ij = net change in i after unit progress in reaction j
        # so the appropriate column of the stoich matrix tells us how to do our update
        update = np.transpose(self.N[:,path_index])
        update = update.reshape(y.shape)
        y_history[:, -1] += update

        return StepUpdate(t_history, y_history, status,step_solved.nfev)

class HybridNotImplementedError(NotImplementedError):
    """Attempted to use a hybrid algorithm feature that has not been implemented."""

@dataclass(frozen=True)
class HybridSimulationOptions():
    """This class defines the configuration of a Haseltine-Rawlings hybrid forward simulation algorithm.

    Attributes
    ----------
        jit (bool, optional):
            whether to insist that propensities, derivatives, and rate constants
            are calculated using Numba jit compiled functions
            (faster but tedious to write). Defaults to True.
        approximate_rtot (bool, optional):
            if True, approximate the total of stochastic propensities as constant
            between stochastic events (see discussion in IV of Haseltine and
            Rawlings paper). If True, contrived_no_reaction_rate must be specified.
            Defaults to False.
        contrived_no_reaction_rate (float, optional):
            specified if and only if approximate_rtot=True: a contrived rate
            of no reaction to ensure that no timestep between stochastic events
            becomes too large. Defaults to None.
        deqs (bool, optional):
            if True, approximate the fast reactions as differential equations.
            If False, treat the fast reactions as Langevin equations. Defaults to True.
        round_randomly (bool, optional):
            if True, round species quantities randomly in proportion to decimal
            (so 1.8 copies of X becomes 2 with 80% probability). If False, round conventionally.
            Defaults to True.
        halt_on_partition_change (bool, optional):
            if True, stop integration when the change in state causes a deterministic
            reaction to enter the stochastic regime. If False, don't. When  Defaults to True.
        partition_fraction_for_halt (float, optional):
            float < 1. Only relevant if ``halt_on_partition_change``, in which case integration
            is stopped when the rate of a reaction reaches ``partition_fraction_for_halt * threshold``
            This avoids numerical instability around the threshold. Defaults to 0.9 (90% of threshold)
    """
    approximate_rtot: bool = False
    contrived_no_reaction_rate: float = None
    deqs: bool = True
    round_randomly: bool = True
    halt_on_partition_change: bool = False
    partition_fraction_for_halt: float = 0.9

    def __post_init__(self):
        if not self.deqs: raise HybridNotImplementedError("simulation was configured with deqs=False, but Langevin equations are not implemented.")
        if self.approximate_rtot:
            assert isinstance(self.contrived_no_reaction_rate, float) and self.contrived_no_reaction_rate > 0, "If approximating stochastic rates as constant in between events, contrived_no_reaction_rate must be a FLOAT greater than 0 to prevent overly large steps."
        else:
            assert(self.contrived_no_reaction_rate is None)

        if self.halt_on_partition_change:
            assert isinstance(self.partition_fraction_for_halt, float)

def canonicalize_event(t_events, y_events):
    """For a set of integration events, ensure our expectations are met and return the event.

    We expect exactly 1 event, which is terminal. We return the time and state at the event, and the event index."""
    event_index = None
    for i,event_set in enumerate(t_events):
        if event_set.size == 0:
            continue
        if event_index is not None:
            assert False, f"that's unusual, we had two different kinds of terminal events in 1 step. t_events={t_events}, y_events={y_events}"
        event_index = i
    # now we have one canonical event that occurred
    t_event = t_events[event_index]
    assert len(t_event) == 1, "we had more than one of the same kind of terminal event!"
    t_event = t_event[0]
    y_event = y_events[event_index][0]

    return t_event, y_event, event_index

def round_with_method(y, round_randomly=False, rng=None):
    """Round the state vector `y` either conventionally or randomly."""
    if round_randomly:
        # round up if random float is less than decimal (small decimal ==> rarely round up)
        # round down otherwise
        rounded = (rng.random(y.shape) <= (y - np.floor(y))) + np.floor(y)
        return rounded
    else:
        return np.round(y)