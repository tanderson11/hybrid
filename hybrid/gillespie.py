import numpy as np
from typing import NamedTuple
from numba import jit as numbajit
from scipy.integrate import quad
from scipy.optimize import fsolve

from .simulator import Step, StepStatus, Simulator

class GillespieStepStatus(StepStatus):
    rejected = -1
    t_end = 0
    stochastic = 1

class GillespieSimulator(Simulator):
    def step(self, t, y, t_end, rng, t_eval):
        if self.inhomogeneous:
            return self.gillespie_step(t, y, t_end, rng, t_eval)
        else:
            return self.homogeneous_gillespie_step(t, y, t_end, rng, t_eval)

    @classmethod
    def construct_propensity_function(cls, k, kinetic_order_matrix, inhomogeneous, jit=True):
        assert jit
        @numbajit(nopython=True)
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
            if inhomogeneous:
                k_of_t = k(t)
            else:
                k_of_t = k
            product_down_columns = np.ones(len(k_of_t))
            for i in range(0, len(y)):
                product_down_columns = product_down_columns * intensity_power[i]
            return product_down_columns * k_of_t
        return jit_calculate_propensities

    @classmethod
    def find_hitting_time_inhomogenous(cls, t, y, propensity_function, rng):
        hitting_point = rng.exponential(1)
        f = cls.inhomogeneous_upper_bound_f_factory(t, y, hitting_point, propensity_function)
        hitting_time = fsolve(f, x0=1)
        return hitting_time

    @staticmethod
    def find_hitting_time_homogeneous(total_propensity, rng):
        hitting_point = rng.exponential(1)
        hitting_time = hitting_point / total_propensity
        return hitting_time

    @staticmethod
    def inhomogeneous_upper_bound_f_factory(t, y, hitting_point, propensity_function):
        # we need to solve for the hitting time
        # which is the time when hitting_point / integral of propensities
        # so we build an objective function that takes a zero when x = hitting_point / integral
        # using numerical minimization we can minimize the objective function to find the time x
        def objective_function(x):
            integral = quad(propensity_function, t, x, args=(y))[0]
            return np.abs(x - hitting_point / integral)
        return objective_function

    @staticmethod
    def gillespie_update_proposal(N, propensities, rng):
        cumsum = propensities.cumsum()
        selections = cumsum / cumsum[-1]
        pathway_rand = rng.random()
        entry = np.argmax(selections > pathway_rand)
        path_index = np.unravel_index(entry, selections.shape)

        # N_ij = net change in i after unit progress in reaction j
        # so the appropriate column of the stoich matrix tells us how to do our update
        update = np.transpose(N[:,path_index])
        update = update.reshape((N.shape[0],))
        return update

    @staticmethod
    def expand_step_with_t_eval(t, y0, hitting_time, update, t_eval, t_end):
        #import pdb; pdb.set_trace()
        # gather any intermediate values requested (y is constant at each intermediate)
        t_history = list(t_eval[(t_eval > t) & (t_eval < t + hitting_time)])
        t_history.append(t+hitting_time)
        y_history = np.zeros((len(y0), len(t_history)))
        # back fill all times prior to the hitting time with the initial state
        if len(t_history) - 1 > 0:
            y_history[:,:len(t_history)-1] = np.expand_dims(y0, 1)
        y_history[:,len(t_history)-1]  = y0+update

        return t_history, y_history

    def gillespie_step(self, t, y, t_end, rng, t_eval):
        hitting_time = self.find_hitting_time_inhomogenous(t, y, self.propensity_function, rng)
        endpoint_propensities = self.propensity_function(t+hitting_time, y)

        update = self.gillespie_update_proposal(self.N, endpoint_propensities, rng)
        t_history, y_history = self.expand_step_with_t_eval(t,y,hitting_time,update,t_eval,t_end)

        return Step(t_history, y_history, GillespieStepStatus.stochastic)

    def homogeneous_gillespie_step(self, t, y, t_end, rng, t_eval):
        propensities = self.propensity_function(t, y)
        total_propensity = np.sum(propensities)
        time_proposal = self.find_hitting_time_homogeneous(total_propensity, rng)
        hitting_time = time_proposal

        if t + hitting_time > t_end:
            update = np.zeros_like(y)
            return Step(*self.expand_step_with_t_eval(t,y,t_end-t,update,t_eval,t_end), GillespieStepStatus.t_end)

        update = self.gillespie_update_proposal(self.N, propensities, rng)
        t_history, y_history = self.expand_step_with_t_eval(t,y,hitting_time,update,t_eval,t_end)

        return Step(t_history, y_history, GillespieStepStatus.stochastic)