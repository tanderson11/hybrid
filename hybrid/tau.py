import numpy as np
from numba import jit
from .gillespie import find_hitting_time_homogeneous, gillespie_update_proposal

def jit_calculate_propensities_factory(kinetic_order_matrix):
    @jit(nopython=True)
    def jit_calculate_propensities(y, k, t):
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

        # then we take the product down the columns (so product over each reaction)
        # and multiply that output by the vector of rate constants
        # to get the propensity of each reaction at time t
        k_of_t = k(t)
        product_down_columns = np.ones(len(k_of_t))
        for i in range(0, len(y)):
            product_down_columns = product_down_columns * intensity_power[i]
        return product_down_columns * k_of_t
    return jit_calculate_propensities

def forward_time(y0: np.ndarray, t_span: list[float], k: np.ndarray, N: np.ndarray, kinetic_order_matrix: np.ndarray, rng: np.random.Generator, **kwargs):
    calculate_propensities = jit_calculate_propensities_factory(kinetic_order_matrix)

def find_L(y, N):
    # the maximum permitted firings of each reaction before reducing a population below 0
    # see formula 5 of Cao et al. 2005
    with np.errstate(divide='ignore'):
        # cryptic numpy wizardry to multiply each element of the stoichiometry matrix by the the corresponding element of y
        L = np.expand_dims(y,axis=1) / N[None, :]
        # drop positive entries (they are being created not destroyed in stoichiometry)
        # and invert negative entries so we can take the min
        L = np.where(L < 0, -L, np.inf)
    L = np.min(L, axis=1)
    return L

def find_critical_reactions(y, N, n_c):
    L = find_L(y, N)
    return L < n_c

def tau_leap_proposal():
    pass

def tau_update_proposal():
    pass

rejection_multiple = 10
def tau_step(y, t, t_end, k, N, calculate_propensities, n_c, rng):
    propensities = calculate_propensities(y, k, t)
    total_propensity = np.sum(propensities)

    critical_reactions = find_critical_reactions(y, N, n_c)

    if (~critical_reactions).all():
        tau_prime = np.inf
    else:
        tau_prime = min(t_end - t, tau_leap_proposal(propensities[~critical_reactions]))

    if tau_prime < rejection_multiple / total_propensity:
        # reject this step and switch to Gillespie for 100 steps
        return

    tau_prime_prime = min(t_end - t, find_hitting_time_homogeneous(propensities[critical_reactions], rng).tau)

    # no critical events took place in our proposed leap forward of tau_prime, so we execute that leap
    if tau_prime < tau_prime_prime:
        tau = tau_prime
        update = tau_update_proposal(tau_prime, propensities, N, rng)
    # a single critical event took place at tau_prime_prime, adjudicate that event and the leap of non-critical reactions
    else:
        tau = tau_prime_prime
        gillespie_update = gillespie_update_proposal(N, propensities[critical_reactions], total_propensity, rng)
        tau_update = tau_update_proposal(tau_prime_prime, propensities[~critical_reactions], N[:, ~critical_reactions])

        update = gillespie_update + tau_update

    return tau, update