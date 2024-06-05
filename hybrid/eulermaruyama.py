from dataclasses import dataclass
import numpy as np
import hybrid.util as util

@dataclass
class EMResult():
    t: np.ndarray
    y: np.ndarray
    nfev: int
    t_events: np.ndarray
    status: int = 0

def dW_f(a, rng):
    return rng.normal(loc=0.0, scale=np.sqrt(a))

def mu_sigma_f(self, t, y, dt, dW, rng, partition, propensity_function, _):
    propensities = propensity_function(t,y)
    propensities = propensities * partition.deterministic

    mu = self.N @ (propensities)
    sigma = self.N @ (np.sqrt(propensities * dt) * rng.normal(loc=0.0, scale=1.0, size=propensities.shape))
    #print(np.sum(sigma))
    return mu, sigma


def em_solve_ivp(N, propensity_function, partition, t_span, y0, rng, t_eval=None, events=None, args=None, dt=0.005, rounding_method='randomly'):
    assert events is None or len(events) == 0
    if args is None: args = ()

    t, t_end = t_span
    y = y0
    nfev = 0
    t_history = [t]
    y_history = [y0]

    # cumulative firings of each reaction in order to generate stoichiometrically realizable outcomes
    total_firings = np.zeros(N.shape[1])

    while t < t_end:
        if t + dt > t_end:
            dt = t_end - t

        propensities = propensity_function(t,y,*args) * partition.deterministic

        # mean process
        pathway_drift = propensities * dt
        # zero-mean Wiener process
        pathway_diffusion = np.sqrt(propensities * dt) * rng.normal(loc=0.0, scale=1.0, size=propensities.shape)
        nfev += 1

        # to achieve actual changes in the state, we matrix multiply mean and random firings by the stoichiometry matrix
        dy = N @ pathway_drift + N @ pathway_diffusion
        # but we use the # of firings in order to make sure the final change in state is stoichiometrically realizable
        total_firings += pathway_drift + pathway_diffusion

        y = y + dy
        t = t + dt
        t_history.append(t)
        y_history.append(y)

    # round to stoichiometrically realizable change:
    # this prevents pathways in equilibrium from randomly walking (due to random rounding)
    # (recall: random rounding is necessary to prevent consistent windfalls/shortfalls in small time steps from biasing simulator)
    if round:
        y_history[-1] = y0 + N @ (util.round_with_method(total_firings, rounding_method, rng))

    result = EMResult(
        np.array(t_history),
        np.array(y_history).T,
        nfev,
        t_events=np.array([])
    )
    return result