from typing import Callable, Union
from enum import Enum, auto
from dataclasses import dataclass

import numba
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import fsolve, least_squares

from hybrid.gillespie import GillespieSimulator
from hybrid.simulator import StepStatus, Step, Run

class TauNotImplementedError(NotImplementedError):
    """Attempted to use a tau leaping algorithm feature that has not been implemented."""

class Method(Enum):
    explicit = 'explicit'
    implicit = 'implicit'

class TimeHandling(Enum):
    homogeneous = 'homogeneous'
    inhomogeneous_monotonic = 'inhomogeneous_monotonic'

class TauLeapers(Enum):
    gp = 'gp'
    corrected = 'corrected'
    species = 'species'

class TauStepStatus(StepStatus):
    rejected_for_gillespie = -2
    rejected = -1
    t_end = 0
    t_end_for_discontinuity = auto()
    leap = auto()
    leap_critical_combined = auto()
    stochastic = auto()

class TauRun(Run):
    def __init__(self, t0, y0, steps_on_rejection, history_length=1e6) -> None:
        super().__init__(t0, y0, history_length)
        self.steps_on_rejection = steps_on_rejection
        self.forced_gillespie_steps = 0

    def handle_step(self, step):
        if step.status == TauStepStatus.rejected_for_gillespie:
            self.forced_gillespie_steps = self.steps_on_rejection
        if step.status == TauStepStatus.stochastic:
            assert self.forced_gillespie_steps > 0
            self.forced_gillespie_steps -= 1

        t = super().handle_step(step)
        return t

    def get_step_kwargs(self):
        return {'do_gillespie': self.forced_gillespie_steps > 0}

@dataclass(frozen=True)
class TauLeapOptions():
    leap_type: str = 'species'
    method: str = 'explicit'
    time_handling: str = 'homogeneous'
    epsilon: float=0.01
    rejection_multiple: float=10
    gillespie_steps_on_rejection: int=100
    critical_threshold: int=10
    species_creation_is_critical: bool=False
    only_reactants_critical: bool=True

class TauLeapSimulator(GillespieSimulator):
    run_klass = TauRun
    status_klass = TauStepStatus
    def __init__(self, k: Union[ArrayLike, Callable], N: ArrayLike, kinetic_order_matrix: ArrayLike, equilibrium_mask: ArrayLike = None, poisson_product_mask: ArrayLike=None, discontinuities: ArrayLike=None, jit: bool=True, propensity_function: Callable=None, species_labels=None, pathway_labels=None, **option_kwargs) -> None:
        super().__init__(k, N, kinetic_order_matrix, poisson_product_mask=poisson_product_mask, discontinuities=discontinuities, jit=jit, propensity_function=propensity_function, species_labels=species_labels, pathway_labels=pathway_labels)
        self.N2 = self.N**2
        simulator_options = TauLeapOptions(**option_kwargs)

        self.epsilon = simulator_options.epsilon
        self.n_c = simulator_options.critical_threshold
        self.rejection_multiple = simulator_options.rejection_multiple
        self.gillespie_steps_on_rejection = simulator_options.gillespie_steps_on_rejection
        self.method = Method(simulator_options.method)
        self.time_handling = TimeHandling(simulator_options.time_handling)
        self.leap_type = TauLeapers(simulator_options.leap_type)
        if self.leap_type == TauLeapers.gp:
            print("WARNING: using known bad Gillespie-Petzold leap. Is this a test?")
        if self.leap_type == TauLeapers.species:
            self.g = self.build_g_function()

        self.species_creation_is_critical = simulator_options.species_creation_is_critical
        self.only_reactants_critical = simulator_options.only_reactants_critical

        self.equilibrium_mask = equilibrium_mask
        if equilibrium_mask is not None and equilibrium_mask.any():
            assert self.method == Method.implicit, "specifying reaction channels as being near equilibrium is only supported for implicit tau leaping"

        if self.inhomogeneous and self.time_handling == TimeHandling.homogeneous:
            print("WARNING: inhomogeneous rates in Tau leap simulation, but simulator hasn't been told to use inhomogeneous leaping. Is this a test?")

    @classmethod
    def from_model(cls, m, *args, reaction_to_k=None, parameters=None, jit: bool=True, **kwargs):
        from hybrid.model import SimulationAwareModel
        if isinstance(m, SimulationAwareModel):
            kwargs['equilibrium_mask'] = m.equilibrium_mask()
            kwargs['poisson_product_mask'] = m.poisson_product_mask()
        return cls(m.get_k(reaction_to_k=reaction_to_k, parameters=parameters, jit=jit), m.stoichiometry(), m.kinetic_order(), *args, species_labels=[s.name for s in m.species], pathway_labels=[r.description for r in m.all_reactions], jit=jit, **kwargs)

    def initiate_run(self, t0, y0, history_length=1e6):
        return self.run_klass(t0, y0, self.gillespie_steps_on_rejection, history_length=history_length)

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

        # for only 1 reaction, we don't want to lose our dimension information
        if len(critical.shape) == 0:
            critical = np.array([critical])

        return critical

    def gillespie_step_wrapper(self, t, y, t_end, rng, t_eval):
        if self.time_handling == TimeHandling.inhomogeneous_monotonic:
            g_step = self.gillespie_step(t, y, t_end, rng, t_eval)
        else:
            g_step = self.homogeneous_gillespie_step(t, y, t_end, rng, t_eval)

        status = g_step.status

        return Step(g_step.t_history, g_step.y_history, status)

    def step(self, t, y, t_end, rng, t_eval, do_gillespie=False):
        if do_gillespie:
            gs = self.gillespie_step_wrapper(t, y, t_end, rng, t_eval)
            return gs
        return self.tau_step(t, y, t_end, rng, t_eval)

    def candidate_time_inhomogeneous(self, t, y, propensities, reaction_mask, t_end):
        assert self.leap_type == TauLeapers.species, "time inhomogeneous tau leaping is only supported with leap type of species"
        # approximately bound the change in each propensity *due to leaping* to one part in epsilon/2
        tau = self.species_tau_leap_proposal(y, self.epsilon/2, self.g, propensities[reaction_mask], self.N[:, reaction_mask], self.N2[:, reaction_mask])
        # bound the change in each propensity *due to explicit time dependence* to one part in epsilon/2
        tau_prime = self.inhomogeneous_time_proposal(t, y, self.epsilon/2, propensities[reaction_mask], reaction_mask, min(t_end-t, tau))

        return min(tau, tau_prime)

    @staticmethod
    def find_acceptable_inhomogeneous_tau_halving(calculate_nonzero_endpoint_props, calculate_largest_fraction_change, tau_max, epsilon):
        acceptable = False

        while not acceptable:
            tau = tau_max / 2
            end_props = calculate_nonzero_endpoint_props(tau)
            if calculate_largest_fraction_change(end_props) < epsilon:
                acceptable = True

        return tau

    @staticmethod
    def find_acceptable_inhomogeneous_tau_lsqs(calculate_nonzero_endpoint_props, calculate_largest_fraction_change, tau_max, epsilon):
        def objective_function(tau):
            '''Returns 0 when tau maximizes our tolerance for error due to explicit time dependence.'''
            nonzero_end_propensities = calculate_nonzero_endpoint_props(tau)
            part_change = calculate_largest_fraction_change(nonzero_end_propensities)
            return np.abs(part_change - epsilon)
        lsqs = least_squares(objective_function, x0=tau_max/2, bounds=(0.0, tau_max))
        if not lsqs.success:
            print(lsqs)
            raise ValueError('least squares fit in time inhomogeneous leap failed')
        return lsqs.x[0]

    def inhomogeneous_time_proposal(self, t, y, epsilon, propensities, reaction_mask, tau_max):
        """Calculate a forward-leap time that bounds the changes in propensities due to explicit time dependence within one part in epsilon.

        This only works if time dependence is entry-wise monotonic in the interval (t, t+tau_max)."""
        end_propensities = self.propensity_function(t + tau_max, y)[reaction_mask]
        # if a rate constant moves from 0 to a positive number, then it will be impossible to bound the change in propensities
        # by epsilon, so we should instead default to gillespie steps
        if (end_propensities[propensities == 0] > 0).any():
            return 0

        nonzero_mask = propensities != 0
        nonzero_propensities = propensities[nonzero_mask]

        def calculate_nonzero_endpoint_props(tau):
            tau = tau[0]
            end_propensities = self.propensity_function(t + tau, y)[reaction_mask]
            non_zero_end_propensities = end_propensities[nonzero_mask]
            return non_zero_end_propensities

        def calculate_largest_fraction_change(nonzero_end_propensities):
            return np.max(np.abs((nonzero_propensities - nonzero_end_propensities)/nonzero_propensities))

        # if the change at the endpoint doesn't break our tolerance, then accept the maximum time jump
        # (recall monotonicity)
        if calculate_largest_fraction_change(calculate_nonzero_endpoint_props([tau_max])) < epsilon:
            return tau_max

        method = 'lsqs'

        if method == 'lsqs':
            return self.find_acceptable_inhomogeneous_tau_lsqs(calculate_nonzero_endpoint_props, calculate_largest_fraction_change, tau_max, epsilon)
        elif method == 'halving':
            return self.find_acceptable_inhomogeneous_tau_havling(calculate_nonzero_endpoint_props, calculate_largest_fraction_change, tau_max, epsilon)


    def candidate_time(self, t, y, propensities, reaction_mask, t_end):
        """Calculate a value of tau to leap forward in time that satisfies the leap condition for the reactions with value True in reaction_mask."""
        if self.leap_type == TauLeapers.gp:
            return self.gp_tau_leap_proposal(y, self.epsilon, propensities[reaction_mask], self.N[:, reaction_mask], self.kinetic_order_matrix[:, reaction_mask])
        elif self.leap_type == TauLeapers.corrected:
            k = self.k(t) if self.inhomogeneous else self.k
            return self.corrected_tau_leap_proposal(y, self.epsilon, propensities[reaction_mask], k, self.N[:, reaction_mask], self.kinetic_order_matrix[:, reaction_mask])
        elif self.leap_type == TauLeapers.species:
            return self.species_tau_leap_proposal(y, self.epsilon, self.g, propensities[reaction_mask], self.N[:, reaction_mask], self.N2[:, reaction_mask])
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

        @numba.jit(nopython=True)
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
    @numba.jit(nopython=True)
    def calculate_mu_sigma(N, N2, propensities):
        mu_hat_i = N @ propensities
        # N2 = N**2 but precalculated for efficiency
        sigma_2_hat_i = N2 @ propensities

        return mu_hat_i, sigma_2_hat_i

    @classmethod
    def species_tau_leap_proposal(cls, y, epsilon, g, propensities, N, N2):
        """Propose a tau leap that uniformly bounds changes in propensities by a change in species approximation. Cao et al 2006 formula (33)."""
        # Cao et al. 2006 formula (33)
        mu_hat_i, sigma_2_hat_i = cls.calculate_mu_sigma(N, N2, propensities)
        # calculate g vector (if it depends on y) or use constant g vector
        g = g(y) if not isinstance(g, np.ndarray) else g

        # Currently, I disagree with Cao and Gillespie: I believe that it should be a max of y*epsilon/g and EPSILON
        # it's unfair in one branch insist that the propensity changes by no more than epsilon of its total value
        # and in the other branch insist that it changes by no more than potentially 100% of its value! when we make our leap
        # we're quite likely in that regime to have multiple reactions to fire, but the whole point of having this maximum here
        # is to say "hey, the reaction will change by at LEAST 1 discrete firing, so let's not insist on a time step that is super ultra tiny"
        # but by choosing this 1/|mu_hat_i|, we're saying "hey it changes by at LEAST 1 firing, so let's let it change by 1,2,or3ish firings"
        # and that is whack
        # although I guess with this species partitioning, ... that's essentially defying what Cao proposed entirely, because y >= 1

        # CAO AND GILLESPIE
        with np.errstate(divide='ignore'):
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
    def explicit_tau_reaction_firings(tau, propensities, rng):
        return rng.poisson(tau * propensities)

    @classmethod
    def implicit_tau_reaction_firings(cls, N, Nplus, Nminus, y, t, tau, propensities, rng, propensity_calculator, jacboian_calculator):
        # all of this is Sandmann 2009 summarizing Rathinam 2003
        mean_explicit_firings = tau * propensities
        # number of firings from the explicit method
        explicit_update, drawn_explicit_firings = cls.explicit_tau_update_proposal(N, Nplus, Nminus, tau, propensities, rng, return_firings=True)
        # convert firings => mean 0
        mean_0 = explicit_update - tau * (N @ propensities)
        # now solve the implicit equation by first defining an objective function
        def objective_function(y_end):
            y_end_propensities = propensity_calculator(t+tau, y_end)
            mean_y_end_firings = tau * y_end_propensities
            return y_end - y - mean_0 - np.sum(mean_y_end_firings * N, axis=1)

        # more precisely how Sandmann did it
        #def objective_function(y_end):
        #    y_end_propensities = propensity_calculator(t+tau, y_end)
        #    mean_y_end_firings = tau * y_end_propensities
        #
        #    firings = np.round(drawn_explicit_firings - mean_explicit_firings + mean_y_end_firings)
        #    return y_end - y + N @ (firings)

        # closes around reaction_firings
        def calculate_k_j(y_end_propensities):
            mean_y_end_firings = tau * y_end_propensities
            return drawn_explicit_firings + mean_y_end_firings - mean_explicit_firings

        #y_end = fsolve(objective_function, y)
        #y_end_lsqs = least_squares(objective_function, x0=y)
        #y_end_lsqs_jac = least_squares(objective_function, x0=y, jac=jacboian_calculator)
        y_end_lsqs_xtol = least_squares(objective_function, x0=y, xtol=1e-8)
        #y_end_lsqs_xtol_jac = least_squares(objective_function, x0=y, xtol=1e-3, jac=jacboian_calculator)
        y_end_lsqs = y_end_lsqs_xtol
        #y_end_lsqs_3 = least_squares(objective_function, x0=y, ftol=1e-4)
        #y_end_lsqs_4 = least_squares(objective_function, x0=y, )
        assert y_end_lsqs.success
        y_end = y_end_lsqs.x

        y_end_propensities = propensity_calculator(t+tau, y_end)
        # after calculating y_end, we will slightly adjust it to get a stoichiometrically realizable, integer-valued vector
        k_j = calculate_k_j(y_end_propensities)
        k_j = np.round(k_j)

        if (k_j < 0).any():
            print(t, "negative")

        return k_j

    @classmethod
    def _update_from_firings(cls, N, Nplus, Nminus, firings, rng, poisson_product_mask=None):
        #assert (firings >= 0).all()
        if poisson_product_mask is None:
            return N @ firings
        product_update_base = Nplus*firings

        # for each reaction with random products, draw the random number of products created
        # taking advantage of the fact that the sum of poisson is poisson of the sum
        product_update_base[:, poisson_product_mask] = rng.poisson(product_update_base[:,poisson_product_mask])

        update = product_update_base.sum(axis=1)
        update += Nminus @ firings
        return update

    @classmethod
    def implicit_tau_update_proposal(cls, N, Nplus, Nminus, y, t, tau, propensities, rng, propensity_calculator, jacobian_calculator, return_firings=False, poisson_product_mask=None):
        k_j = cls.implicit_tau_reaction_firings(N, Nplus, Nminus, y, t, tau, propensities, rng, propensity_calculator, jacobian_calculator)
        update = cls._update_from_firings(N, Nplus, Nminus, k_j, rng, poisson_product_mask=poisson_product_mask)
        if return_firings:
            return update, k_j
        return update

    @classmethod
    def explicit_tau_update_proposal(cls, N, Nplus, Nminus, tau, propensities, rng, return_firings=False, poisson_product_mask=None):
        reaction_firings = cls.explicit_tau_reaction_firings(tau, propensities, rng)
        update = cls._update_from_firings(N, Nplus, Nminus, reaction_firings, poisson_product_mask)
        if return_firings:
            return update, reaction_firings
        return update

    def tau_update_proposal_avoiding_negatives(self, N, Nplus, Nminus, y, t, tau, propensities, rng, poisson_product_mask=None, method=Method.explicit, propensity_calculator=None, jacobian_calculator=None):
        bad_update_flag = True
        while bad_update_flag:
            if method == Method.explicit:
                update = self.explicit_tau_update_proposal(N, Nplus, Nminus, tau, propensities, rng, poisson_product_mask=poisson_product_mask)
            else:
                assert method == Method.implicit
                update = self.implicit_tau_update_proposal(N, Nplus, Nminus, y, t, tau, propensities, rng, propensity_calculator, jacobian_calculator, poisson_product_mask=poisson_product_mask)

            if not ((y + update) < 0).any():
                bad_update_flag = False
            else:
                # if we get 'exceedingly unlucky' and a species became extinct due to the tau leap
                # (recall that Poisson has non zero probability density for all positive integers)
                # then try again with half the time.
                tau = tau/2
        return tau, update

    def tau_step(self, t, y, t_end, rng, t_eval):
        # a tracker variable if a Gillespie step is made, initialize as None
        pathway = None

        propensities = self.propensity_function(t, y)
        total_propensity = np.sum(propensities)

        critical_reactions = self.find_critical_reactions(y)
        critical_sum = np.sum(propensities[critical_reactions])

        poisson_product_mask = None if self.poisson_product_mask is None else self.poisson_product_mask[~critical_reactions]

        # if all reactions are critical, we won't tau leap, we'll just do gillespie
        if (critical_reactions).all():
            tau_prime = np.inf
        else:
            # when calculating the acceptable leap time, we ignore critical processes
            reaction_mask = ~critical_reactions
            # for the implicit method, we have a mask of those reactions that we should consider to be
            # near equilibrium, which should NOT be used to determine the acceptable leap time
            if self.equilibrium_mask is not None and self.method == Method.implicit:
                reaction_mask = reaction_mask & (~self.equilibrium_mask)
            if self.inhomogeneous:
                tau_prime = self.candidate_time_inhomogeneous(t, y, propensities, reaction_mask, t_end)
            else:
                tau_prime = self.candidate_time(t, y, propensities, reaction_mask, t_end)
            tau_prime = min(t_end - t, tau_prime)

        if tau_prime < self.rejection_multiple / total_propensity:
            # reject this step and switch to Gillespie's algorithm for a fixed # of steps
            return Step(None, None, self.status_klass.rejected_for_gillespie)

        if critical_sum == 0:
            tau_prime_prime = np.inf
        else:
            if self.time_handling == TimeHandling.inhomogeneous_monotonic:
                # we desperately want to avoid the tragedy of calculating the integral for the hitting time
                # so we exploit monotonicity to ask how quickly could the next critical reaction firing happen
                # if, even at its fastest occurrence, it would not happen before our proposed leap,
                # then we leap
                worst_case_critical_sum = max(critical_sum, np.sum(self.propensity_function(t + tau_prime, y)))
                tau_prime_prime_upper_bound = self.find_hitting_time_homogeneous(worst_case_critical_sum, rng)

                # if, however, the worst case does happen before our leap,
                # then we refine our estimate of the arrival time to the exact arrival time of the
                # inhomogeneous poisson process
                if tau_prime < tau_prime_prime_upper_bound:
                    tau_prime_prime = self.find_hitting_time_inhomogenous(t, t_end, y, self.propensity_function, rng)
                else:
                    # this represents the knowledge that the worst case arrival time of the critical process
                    # is after the proposed leap. by setting tau_prime_prime to inf,
                    # the conditional on the future line will always be True, and a non-critical leap will be executed
                    tau_prime_prime = np.inf
            else:
                tau_prime_prime = self.find_hitting_time_homogeneous(critical_sum, rng)
            #tau_prime_prime = 1 / critical_sum

        # no critical events took place in our proposed leap forward of tau_prime, so we execute that leap
        if tau_prime < tau_prime_prime:
            tau = tau_prime

            propensity_calculator = None
            jacobian_calculator = None
            if self.method == Method.implicit:
                def propensity_calculator(t, y):
                    propensities = self.propensity_function(t, y)
                    return propensities[~critical_reactions]

                def jacobian_calculator(y):
                    powers = (self.kinetic_order_matrix - 1)
                    y = np.expand_dims(y, axis=1)
                    derivatives = np.power(y, powers)

                    # derivative is 0 in every y_i with no rate involvement
                    derivatives = np.where(
                        powers < 0,
                        0,
                        derivatives
                    )
                    derivatives *= self.kinetic_order_matrix

                    # transpose so that rows are reactions
                    derivatives = derivatives.T
                    assert derivatives.shape[1] == len(y)
                    # the actual changes in the *Ys* due to the changes in the propensities
                    jacobian = self.N @ derivatives
                    return jacobian

            tau, update = self.tau_update_proposal_avoiding_negatives(self.N[:, ~critical_reactions], self.Nplus[:, ~critical_reactions], self.Nminus[:, ~critical_reactions], y, t, tau, propensities[~critical_reactions], rng, poisson_product_mask=poisson_product_mask, method=self.method, propensity_calculator=propensity_calculator, jacobian_calculator=jacobian_calculator)

            if t+tau_prime == t_end:
                status = self.status_klass.t_end
            else:
                status = self.status_klass.leap
        # a single critical event took place at tau_prime_prime, adjudicate that event and the leap of non-critical reactions
        else:
            tau = tau_prime_prime
            #print(tau)
            #if np.sum(critical_reactions) >= 1: import pudb; pudb.set_trace()
            pathway, gillespie_update = self.gillespie_update_proposal(self.N[:, critical_reactions], self.Nplus[:, critical_reactions], self.Nminus[:, critical_reactions], propensities[critical_reactions], rng, poisson_product_mask=poisson_product_mask)
            tau_update = self.explicit_tau_update_proposal(self.N[:, ~critical_reactions], self.Nplus[:, ~critical_reactions], self.Nminus[:, ~critical_reactions], tau, propensities[~critical_reactions], rng)
            update = gillespie_update + tau_update

            if ((y+update) < 0 ).any():
                return Step(None, None, self.status_klass.rejected)
            status = self.status_klass.leap_critical_combined

        t_history, y_history = self.expand_step_with_t_eval(t,y,tau,update,t_eval,t_end)

        if pathway is not None:
            return Step(t_history, y_history, status, pathway=pathway)
        return Step(t_history, y_history, status)