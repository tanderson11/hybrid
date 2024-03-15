from typing import Callable, Union

import numpy as np
from numpy.typing import ArrayLike

from numba import jit
from .gillespie import GillespieSimulator
from hybrid.simulator import StepStatus, Step, Run

class TauStepStatus(StepStatus):
    gillespie_rejected = -3
    rejected_for_gillespie = -2
    rejected = -1
    leap = 0
    leap_critical_combined = 1
    gillespie_t_end = 2
    gillespie_stochastic = 3

    @classmethod
    def from_gillespie_status(cls, status):
        return cls[f'gillespie_{status.name}']

class TauRun(Run):
    def __init__(self, t0, y0, steps_on_rejection, history_length=1000000) -> None:
        super().__init__(t0, y0, history_length)
        self.steps_on_rejection = steps_on_rejection
        self.forced_gillespie_steps = 0

    def handle_step(self, step):
        if step.status == TauStepStatus.rejected_for_gillespie:
            self.forced_gillespie_steps = self.steps_on_rejection
        if step.status == TauStepStatus.gillespie_stochastic:
            assert self.forced_gillespie_steps > 0
            self.forced_gillespie_steps -= 1

        return super().handle_step(step)

    def get_step_kwargs(self):
        return {'do_gillespie': self.forced_gillespie_steps > 0}

class TauLeapSimulator(GillespieSimulator):
    run_klass = TauRun
    def __init__(self, k: Union[ArrayLike, Callable], N: ArrayLike, kinetic_order_matrix: ArrayLike, jit: bool=True, propensity_function: Callable=None, epsilon=0.01, critical_threshold=10, rejection_multiple=10, gillespie_steps_on_rejection=100) -> None:
        super().__init__(k, N, kinetic_order_matrix, jit, propensity_function)
        self.epsilon = epsilon
        self.n_c = critical_threshold
        self.rejection_multiple = rejection_multiple
        self.gillespie_steps_on_rejection = gillespie_steps_on_rejection

        assert not self.inhomogeneous

    def initiate_run(self, t0, y0):
        return self.run_klass(t0, y0, self.gillespie_steps_on_rejection)

    def find_L(self, y):
        # the maximum permitted firings of each reaction before reducing a population below 0
        # see formula 5 of Cao et al. 2005
        with np.errstate(divide='ignore'):
            # cryptic numpy wizardry to multiply each element of the stoichiometry matrix by the the corresponding element of y
            L = np.expand_dims(y,axis=1) / self.N[None, :]
            # drop positive entries (they are being created not destroyed in stoichiometry)
            # and invert negative entries so we can take the min
            L = np.where(L < 0, -L, np.inf)
        L = np.squeeze(np.min(L, axis=1))
        return L

    def find_critical_reactions(self, y):
        L = self.find_L(y)
        return L < self.n_c

    def gillespie_step_wrapper(self, t, y, t_end, rng, t_eval):
        g_step = self.homogeneous_gillespie_step(t, y, t_end, rng, t_eval)
        status = TauStepStatus.from_gillespie_status(g_step.status)

        return Step(g_step.t_history, g_step.y_history, status)

    def step(self, t, y, t_end, rng, t_eval, do_gillespie=False):
        if do_gillespie:
            return self.gillespie_step_wrapper(t, y, t_end, rng, t_eval)
        return self.tau_step(t, y, t_end, rng, t_eval)

    @staticmethod
    def tau_leap_proposal(y, epsilon, propensities, N, kinetic_order_matrix):
        # equations 1-3 in Cao et al. 2005, which are really drawn from Gillespie and Petzold 2003
        # we will approximate the rate laws as if they were exponentiation rather than binomials

        # we're trying to divide each column of kinetic_order by y
        # and multiply each row by the propensities. That gives da_j / d_x_i
        # j = rows, j prime = columns
        # since numpy's default way of multiplying, say, a (3,2) by a (2,)
        # is to adjudicate along the rows, then we first multiply by propensities
        # then transpose and divide by y (so that y is divided along columns)
        f_jjp = (((kinetic_order_matrix * propensities).T / y) @ N)

        # want to multiply and sum along the rows
        mu_j = np.sum(f_jjp * propensities, axis=1)
        sigma_2_j = np.sum(f_jjp**2 * propensities, axis=1)

        total_propensity = np.sum(propensities)

        tau1 = np.min(epsilon * total_propensity / np.abs(mu_j)) # min'd over j
        tau2 = np.min(epsilon**2 * total_propensity**2 / sigma_2_j) # min'd over j
        tau = min(tau1, tau2)

        return tau


    @staticmethod
    def tau_update_proposal(N, tau, propensities, rng):
        reaction_firings = rng.poisson(tau * propensities)
        update = np.sum(reaction_firings * N, axis=1)
        return update

    def tau_update_proposal_avoiding_negatives(self, N, y, tau, propensities, rng):
        """Get a valid leap, halving time of leap and redrawing if a negative total would be reached."""
        bad_update_flag = True
        while bad_update_flag:
            update = self.tau_update_proposal(N, tau, propensities, rng)

            if not ((y + update) < 0).any():
                bad_update_flag = False
            else:
                # if we get 'exceedingly unlucky' and a species became extinct due to the tau leap
                # (recall that Poisson has non zero probability density for all positive integers)
                # then try again with half the time.
                tau = tau/2
        return tau, update

    def tau_step(self, t, y, t_end, rng, t_eval):
        #import pdb; pdb.set_trace()
        propensities = self.propensity_function(t, y)
        total_propensity = np.sum(propensities)

        critical_reactions = self.find_critical_reactions(y)
        critical_sum = np.sum(propensities[critical_reactions])

        # if all reactions are critical, we won't tau leap, we'll just do gillespie
        if (critical_reactions).all():
            tau_prime = np.inf
        else:
            #import pdb; pdb.set_trace()
            tau_prime = self.tau_leap_proposal(y, self.epsilon, propensities[~critical_reactions], self.N[:, ~critical_reactions], self.kinetic_order_matrix[:, ~critical_reactions])
            tau_prime = min(t_end - t, tau_prime)

        if tau_prime < self.rejection_multiple / total_propensity:
            # reject this step and switch to Gillespie's algorithm for a fixed # of steps
            return Step(None, None, TauStepStatus.rejected_for_gillespie)

        tau_prime_prime = self.find_hitting_time_homogeneous(critical_sum, rng)

        # no critical events took place in our proposed leap forward of tau_prime, so we execute that leap
        if tau_prime < tau_prime_prime:
            tau = tau_prime
            tau, update = self.tau_update_proposal_avoiding_negatives(self.N, y, tau, propensities, rng)

            status = TauStepStatus.leap
        # a single critical event took place at tau_prime_prime, adjudicate that event and the leap of non-critical reactions
        else:
            tau = tau_prime_prime
            print(tau)
            import pdb; pdb.set_trace()
            gillespie_update = self.gillespie_update_proposal(self.N, propensities[critical_reactions], total_propensity, rng)
            tau, tau_update = self.tau_update_proposal_avoiding_negatives(self.N[:, ~critical_reactions], y, tau, propensities[~critical_reactions], rng)
            update = gillespie_update + tau_update
            status = TauStepStatus.leap_critical_combined

        t_history, y_history = self.expand_step_with_t_eval(t,y,tau,update,t_eval,t_end)

        return Step(t_history, y_history, status)

def jit_calculate_propensities_factory(kinetic_order_matrix):
    @jit(nopython=True)
    def jit_calculate_propensities(y, k, t):
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
        k_of_t = k(t)
        product_down_columns = np.ones(len(k_of_t))
        for i in range(0, len(y)):
            product_down_columns = product_down_columns * intensity_power[i]
        return product_down_columns * k_of_t
    return jit_calculate_propensities