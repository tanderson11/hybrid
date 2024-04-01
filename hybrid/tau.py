from typing import Callable, Union
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike

from hybrid.gillespie import GillespieSimulator
from hybrid.simulator import StepStatus, Step, Run

class TauNotImplementedError(NotImplementedError):
    """Attempted to use a tau leaping algorithm feature that has not been implemented."""

class TauLeapers(Enum):
    gp = 'gp'
    corrected = 'corrected'
    species = 'species'

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

        t = super().handle_step(step)
        return t

    def get_step_kwargs(self):
        return {'do_gillespie': self.forced_gillespie_steps > 0}

class TauLeapSimulator(GillespieSimulator):
    run_klass = TauRun
    def __init__(self, k: Union[ArrayLike, Callable], N: ArrayLike, kinetic_order_matrix: ArrayLike, jit: bool=True, propensity_function: Callable=None, leap_type='species', species_creation_is_critical=False, only_reactants_critical=True, epsilon=0.01, critical_threshold=10, rejection_multiple=10, gillespie_steps_on_rejection=100) -> None:
        super().__init__(k, N, kinetic_order_matrix, jit, propensity_function)
        self.epsilon = epsilon
        self.n_c = critical_threshold
        self.rejection_multiple = rejection_multiple
        self.gillespie_steps_on_rejection = gillespie_steps_on_rejection
        self.leap_type = TauLeapers(leap_type)
        if self.leap_type == TauLeapers.gp:
            print("WARNING: using known bad Gillespie-Petzold leap. Is this a test?")
        if self.leap_type == TauLeapers.species:
            self.g = self.build_g_function()

        self.species_creation_is_critical = species_creation_is_critical
        self.only_reactants_critical = only_reactants_critical

        if self.inhomogeneous:
            print("WARNING: inhomogeneous rates in Tau leap simulation. Will assume that rates are constant within leaps.")

    def initiate_run(self, t0, y0):
        return self.run_klass(t0, y0, self.gillespie_steps_on_rejection)

    def find_L(self, y, reactants_only=True):
        # the maximum permitted firings of each reaction before reducing a population below 0
        # see formula 5 of Cao et al. 2005
        with np.errstate(divide='ignore', invalid='ignore'):
            # cryptic numpy wizardry to divide each element of y by the corresponding element of the stoichiometry matrix
            L = np.expand_dims(y,axis=1) / self.N[None, :]
        # drop positive entries (they are being created not destroyed in stoichiometry)
        # and invert negative entries so we can take the min
        if reactants_only:
            L = np.where(L < 0, -L, np.inf)
        else:
            L = np.where(L != 0, np.abs(L), np.inf)
        L = np.squeeze(np.min(L, axis=1))
        return L

    def find_critical_reactions(self, y):
        L = self.find_L(y, self.only_reactants_critical)
        critical = L < self.n_c

        # optionally add reactions that produce any species that currently has 0 specimens to critical reactions
        if self.species_creation_is_critical:
            zero_specimens = y==0
            if zero_specimens.any():
                new_species_mask = ((self.N>0)[zero_specimens]).max(axis=0)
                critical |= new_species_mask

        return critical

    def gillespie_step_wrapper(self, t, y, t_end, rng, t_eval):
        g_step = self.homogeneous_gillespie_step(t, y, t_end, rng, t_eval)
        status = TauStepStatus.from_gillespie_status(g_step.status)

        return Step(g_step.t_history, g_step.y_history, status)

    def step(self, t, y, t_end, rng, t_eval, do_gillespie=False):
        if do_gillespie:
            return self.gillespie_step_wrapper(t, y, t_end, rng, t_eval)
        return self.tau_step(t, y, t_end, rng, t_eval)

    def tau_leap_proposal(self, t, y, propensities, critical_reactions):
        #tau_prime = self.tau_leap_proposal(y, self.epsilon, propensities[~critical_reactions], k[~critical_reactions], self.N[:, ~critical_reactions], self.kinetic_order_matrix[:, ~critical_reactions])

        if self.leap_type == TauLeapers.gp:
            return self.gp_tau_leap_proposal(y, self.epsilon, propensities[~critical_reactions], self.N[:, ~critical_reactions], self.kinetic_order_matrix[:, ~critical_reactions])
        elif self.leap_type == TauLeapers.corrected:
            k = self.k(t) if self.inhomogeneous else self.k
            return self.corrected_tau_leap_proposal(y, self.epsilon, propensities[~critical_reactions], k, self.N[:, ~critical_reactions], self.kinetic_order_matrix[:, ~critical_reactions])
        elif self.leap_type == TauLeapers.species:
            return self.species_tau_leap_proposal(y, self.epsilon, self.g, propensities[~critical_reactions], self.N[:, ~critical_reactions])
        else:
            raise ValueError(f"unknown or unimplemented leap type {self.leap_type}")

    def build_g_function(self):
        # highest order of any REACTION of which species i is a reactant
        reaction_order = self.kinetic_order_matrix.sum(axis=0)
        is_reactant = self.kinetic_order_matrix>0
        hor = (reaction_order * is_reactant).max(axis=1)

        if (hor > 3).any():
            raise TauNotImplementedError("calculation of the g values is not implemented for reactions with order higher than three.")

        # highest order of SPECIES i as reactant in any reaction of the highest order in which it participates
        shohor = np.max(np.equal.outer(hor, reaction_order) * self.kinetic_order_matrix, axis=1)

        if (shohor == 1).all():
            return hor

        def g(y):
            g = np.zeros_like(y)
            g = np.where(
                shohor==1,
                hor,
                g
            )
            g = np.where(
                (shohor == 2) & (hor == 2),
                2 + 1/(y-1),
                g
            )

            g = np.where(
                (shohor == 2) & (hor == 3),
                3/2 * (2 + 1/(y-1)),
                g
            )

            g = np.where(
                (shohor == 3) & (hor == 3),
                3 + 1/(y-1) + 1/(y-2),
                g
            )

            # TK what to do with infinite values?
            return g

        return g

    @staticmethod
    def species_tau_leap_proposal(y, epsilon, g, propensities, N):
        """Propose a tau leap that uniformly bounds changes in propensities by a change in species approximation. Cao et al 2006 formula (33)."""
        # Cao et al. 2006 formula (33)
        mu_hat_i = N @ propensities
        sigma_2_hat_i = (N**2) @ propensities
        # calculate g vector (if it depends on y) or use constant g vector
        g = g(y) if not isinstance(g, np.ndarray) else g

        # WEIRD PROBLEM: if I insist on a smaller upper bound here, the results are WORSE in the mutant emergence problem

        # Currently, I disagree with Cao and Gillespie: I believe that it should be a max of y*epsilon/g and EPSILON
        # it's unfair in one branch insist that the propensity changes by no more than epsilon of its total value
        # and in the other branch insist that it changes by no more than potentially 100% of its value! when we make our leap
        # we're quite likely in that regime to have multiple reactions to fire, but the whole point of having this maximum here
        # is to say "hey, the reaction will change by at LEAST 1 discrete firing, so let's not insist on a time step that is super ultra tiny"
        # but by choosing this 1/|mu_hat_i|, we're saying "hey it changes by at LEAST 1 firing, so let's let it change by 1,2,or3ish firings"
        # and that is whack
        # although I guess with this species partitioning, ... that's essentially defying what Cao proposed entirely, because y >= 1

        # CAO AND GILLESPIE
        tau1 = np.min(np.maximum(np.nan_to_num(y * epsilon / g, 0), 1) / np.abs(mu_hat_i))
        tau2 = np.min(np.maximum(np.nan_to_num(y * epsilon / g, 0), 1)**2 / np.abs(sigma_2_hat_i))
        # THAYER
        # should modify this so we don't get epsilon**2?
        #tau1 = np.min(np.maximum(y * epsilon / g, epsilon) / np.abs(mu_hat_i))
        #tau2 = np.min(np.maximum(y * epsilon / g, epsilon)**2 / np.abs(sigma_2_hat_i))

        return min(tau1, tau2)

    @staticmethod
    def gp_tau_leap_proposal(y, epsilon, propensities, N, kinetic_order_matrix):
        assert False, "Haven't carefully checked if the right thing happens with division by 0"
        # equations 1-3 in Cao et al. 2005, which are really drawn from Gillespie and Petzold 2003
        # we will approximate the rate laws as if they were exponentiation rather than binomials

        # we're trying to divide each column of kinetic_order by y
        # and multiply each row by the propensities. That gives da_j / dx_i
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
    def corrected_tau_leap_proposal(y, epsilon, propensities, k_vec, N, kinetic_order_matrix):
        """Propose an extent of time tau to leap forward that ensures no propensity changes by more than epsilon relative to itself. Cao et al. 2006."""
        assert False, "Haven't carefully checked if the right thing happens with division by 0"

        # see notes under gp_tau_leap_proposal to understand who is transposed and why
        # [derivative is 0 w.r.t any species coordinate that is currently 0]
        quotient = np.divide((kinetic_order_matrix * propensities).T, y, out=np.zeros_like((kinetic_order_matrix * propensities).T), where=y!=0) # divide by 0 => 0
        f_jjp = (quotient @ N)

        # want to multiply and sum along the rows
        mu_j = np.sum(f_jjp * propensities, axis=1)
        sigma_2_j = np.sum(f_jjp**2 * propensities, axis=1)

        quotient = np.divide(propensities, np.abs(mu_j), out=np.zeros_like(propensities), where=mu_j!=0) # division by 0 => sent to 0
        quotient = np.where(np.isinf(quotient), 0, quotient)
        tau1 = np.min(np.maximum(epsilon * quotient, k_vec)) # min'd over j
        quotient = np.divide(propensities**2, np.abs(sigma_2_j), out=np.zeros_like(propensities), where=mu_j!=0) # division by 0 => sent to 0
        quotient = np.where(np.isinf(quotient), 0, quotient)
        tau2 = np.min(np.maximum(epsilon**2 * quotient, k_vec**2)) # min'd over j
        print(min(tau1, tau2))
        return min(tau1, tau2)

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
        #import pudb; pudb.set_trace()
        propensities = self.propensity_function(t, y)
        total_propensity = np.sum(propensities)

        critical_reactions = self.find_critical_reactions(y)
        critical_sum = np.sum(propensities[critical_reactions])

        #import pdb; pdb.set_trace()

        #if critical_reactions.any(): import pudb; pudb.set_trace()

        # if all reactions are critical, we won't tau leap, we'll just do gillespie
        if (critical_reactions).all():
            tau_prime = np.inf
        else:
            #import pdb; pdb.set_trace()
            tau_prime = self.tau_leap_proposal(t, y, propensities, critical_reactions)
            tau_prime = min(t_end - t, tau_prime)

        if tau_prime < self.rejection_multiple / total_propensity:
            # reject this step and switch to Gillespie's algorithm for a fixed # of steps
            return Step(None, None, TauStepStatus.rejected_for_gillespie)

        if critical_sum == 0:
            tau_prime_prime = np.inf
        else:
            tau_prime_prime = self.find_hitting_time_homogeneous(critical_sum, rng)
            #tau_prime_prime = 1 / critical_sum

        # no critical events took place in our proposed leap forward of tau_prime, so we execute that leap
        if tau_prime < tau_prime_prime:
            tau = tau_prime
            tau, update = self.tau_update_proposal_avoiding_negatives(self.N[:, ~critical_reactions], y, tau, propensities[~critical_reactions], rng)

            status = TauStepStatus.leap
        # a single critical event took place at tau_prime_prime, adjudicate that event and the leap of non-critical reactions
        else:
            tau = tau_prime_prime
            #print(tau)
            #if np.sum(critical_reactions) >= 1: import pudb; pudb.set_trace()
            gillespie_update = self.gillespie_update_proposal(self.N[:, critical_reactions], propensities[critical_reactions], rng)
            tau_update = self.tau_update_proposal(self.N[:, ~critical_reactions], tau, propensities[~critical_reactions], rng)
            update = gillespie_update + tau_update

            if ((y+update) < 0 ).any():
                return Step(None, None, TauStepStatus.rejected)
            status = TauStepStatus.leap_critical_combined

        t_history, y_history = self.expand_step_with_t_eval(t,y,tau,update,t_eval,t_end)

        return Step(t_history, y_history, status)