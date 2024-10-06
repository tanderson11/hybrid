from enum import auto
from typing import Callable, Union
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike
from numba import jit as numbajit
from scipy.integrate import quad
from scipy.optimize import fsolve, least_squares

from .simulator import Step, StepStatus, Simulator

class GillespieStepStatus(StepStatus):
    t_end = 0
    t_end_for_discontinuity = auto()
    stochastic = auto()

@dataclass(frozen=True)
class GillespieOptions():
    pass

class GillespieSimulator(Simulator):
    status_klass = GillespieStepStatus
    def __init__(self, k: Union[ArrayLike, Callable],
                 N: ArrayLike,
                 kinetic_order_matrix: ArrayLike,
                 poisson_products_mask: ArrayLike=None,
                 discontinuities: ArrayLike=None,
                 jit: bool=True,
                 propensity_function: Callable=None,
                 species_labels=None,
                 pathway_labels=None,
                 reaction_index_to_hooks=None
                ) -> None:
        super().__init__(k, N, kinetic_order_matrix, poisson_products_mask=poisson_products_mask, discontinuities=discontinuities, jit=jit, propensity_function=propensity_function, species_labels=species_labels, pathway_labels=pathway_labels)
        self.reaction_index_to_hooks = reaction_index_to_hooks

    @classmethod
    def from_model(cls, m, *args, reaction_to_k=None, parameters=None, jit: bool=True, **kwargs):
        reaction_index_to_hooks = m.reaction_index_to_hooks if m.has_hooks else None
        return cls(m.get_k(reaction_to_k=reaction_to_k, parameters=parameters, jit=jit), m.stoichiometry(), m.kinetic_order(), *args, species_labels=[s.name for s in m.species], pathway_labels=[r.description for r in m.all_reactions], reaction_index_to_hooks=reaction_index_to_hooks, jit=jit, **kwargs)

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
    def find_hitting_time_inhomogenous(cls, t, t_end, y, propensity_function, rng):
        hitting_point = rng.exponential(1)
        #hitting_point = 1
        #hitting_point = np.log(1/rng.random())

        f = cls.inhomogeneous_upper_bound_f_factory(t, y, hitting_point, propensity_function)

        x0 = min(0.1, (t_end-t)/2)
        least_sqs = least_squares(f, x0=x0, bounds=(0.0, t_end-t))
        hitting_time = least_sqs.x[0]

        if not least_sqs.success:
            raise ValueError('least squares failed')

        # we didn't really find the hitting time, we stopped ourselves from exceeding the end of the time span
        # (the time dependence might not be well defined there) return inf to avoid causing a final event
        if np.isclose(hitting_time, t_end-t):
            return np.inf

        return hitting_time

    @staticmethod
    def inhomogeneous_upper_bound_f_factory(t, y, hitting_point, propensity_function):
        # we need to solve for the hitting time
        # which is the time when hitting_point = integral of propensities
        # so we build an objective function of tau that takes a zero when hitting_point = integral
        # using numerical minimization we can minimize the objective function to find the time tau
        def propensity_wrapped(t):
            propensities = propensity_function(t, y)
            return np.sum(propensities)

        def objective_function(tau):
            integral = quad(propensity_wrapped, t, t+tau)[0]
            if integral == 0.:
                return np.inf
            return np.abs(integral - hitting_point)
        return objective_function

    @staticmethod
    def find_hitting_time_homogeneous(total_propensity, rng):
        hitting_point = rng.exponential(1)
        with np.errstate(divide='ignore'):
            hitting_time = hitting_point / total_propensity
        return hitting_time

    @staticmethod
    def _pick_gillespie_pathway(propensities, rng):
        cumsum = propensities.cumsum()
        selections = cumsum / cumsum[-1]
        pathway_rand = rng.random()
        entry = np.argmax(selections > pathway_rand)
        path_index = np.unravel_index(entry, selections.shape)
        assert len(path_index) == 1
        pathway = path_index[0]

        return pathway

    @staticmethod
    def _gillespie_update(N, pathway):
        # N_ij = net change in i after unit progress in reaction j
        # so the appropriate column of the stoich matrix tells us how to do our update
        update = np.transpose(N[:,pathway])
        update = update.reshape((N.shape[0],))
        return update

    @classmethod
    def gillespie_update_proposal(cls, N, propensities, rng, poisson_products_mask=None, Nplus=None, Nminus=None):
        pathway = cls._pick_gillespie_pathway(propensities, rng)
        update = cls._gillespie_update(N, pathway)
        if poisson_products_mask is not None and poisson_products_mask[pathway]:
            positive_update = cls._gillespie_update(Nplus, pathway)
            negative_update = cls._gillespie_update(Nminus, pathway)
            update = rng.poisson(positive_update) + negative_update

        return pathway, update

    @staticmethod
    def expand_step_with_t_eval(t, y0, hitting_time, update, t_eval, t_end):
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
        """Potentially inhomogeneous Gillespie step."""
        zero_pop_reactant = np.broadcast_to(np.expand_dims((y == 0), axis=1), self.kinetic_order_matrix.shape)
        kinetically_involved = (self.kinetic_order_matrix > 0)
        noncontributing_reactions = (zero_pop_reactant & kinetically_involved).any(axis=0)
        if noncontributing_reactions.all():
            hitting_time = np.inf
        else:
            def propensity_function(t, y):
                propensities = self.propensity_function(t, y)
                return propensities[~noncontributing_reactions]

            hitting_time = self.find_hitting_time_inhomogenous(t, t_end, y, propensity_function, rng)

        if t + hitting_time > t_end:
            update = np.zeros_like(y)
            return Step(*self.expand_step_with_t_eval(t,y,t_end-t,update,t_eval,t_end), self.status_klass.t_end)

        endpoint_propensities = self.propensity_function(t+hitting_time, y)

        pathway, update = self.gillespie_update_proposal(self.N, endpoint_propensities, rng, self.poisson_products_mask, Nplus=self.Nplus, Nminus=self.Nminus)
        hook = self.reaction_index_to_hooks.get(pathway, None)
        if hook is not None:
            assert self.poisson_products_mask is None
            hook_pathway, hook_update = self.gillespie_update_proposal(hook.N, hook.p, rng)
            update = update + hook_update

        t_history, y_history = self.expand_step_with_t_eval(t,y,hitting_time,update,t_eval,t_end)

        return Step(t_history, y_history, self.status_klass.stochastic, pathway=pathway)

    def homogeneous_gillespie_step(self, t, y, t_end, rng, t_eval):
        propensities = self.propensity_function(t, y)
        total_propensity = np.sum(propensities)
        time_proposal = self.find_hitting_time_homogeneous(total_propensity, rng)
        hitting_time = time_proposal

        if t + hitting_time > t_end:
            update = np.zeros_like(y)
            return Step(*self.expand_step_with_t_eval(t,y,t_end-t,update,t_eval,t_end), self.status_klass.t_end)

        pathway, update = self.gillespie_update_proposal(self.N, propensities, rng, self.poisson_products_mask, Nplus=self.Nplus, Nminus=self.Nminus)
        hook = self.reaction_index_to_hooks.get(pathway, None)
        if hook is not None:
            assert self.poisson_products_mask is None
            hook_pathway, hook_update = self.gillespie_update_proposal(hook.N, hook.p, rng)
            update = update + hook_update
        t_history, y_history = self.expand_step_with_t_eval(t,y,hitting_time,update,t_eval,t_end)

        return Step(t_history, y_history, self.status_klass.stochastic, pathway)