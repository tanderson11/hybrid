import numpy as np
import hybrid.hybrid as hybrid
from statsmodels.stats import proportion
import unittest
from numba import jit, float64
from numba.types import Array

def get_k_one_species(birth_rate, death_rate):
    @jit(Array(float64, 1, "C")(float64), nopython=True)
    def k(t):
        return np.array([birth_rate, death_rate]).astype(float64)
    return k

def get_k_two_species(birth_rate, death_rate):
    @jit(Array(float64, 1, "C")(float64), nopython=True)
    def k(t):
        return np.array([1.1, 1.0, birth_rate, death_rate]).astype(float64)
    return k

N_one_species = np.array([[1,-1]], dtype=float)
rate_involvement_one_species = np.array([[1,1]])

N_two_species = np.array([[1,-1,0,0],[0,0,1,-1]], dtype=float)
rate_involvement_two_species = np.array([[1,1,0,0],[0,0,1,1]])

class BirthDeathTest(unittest.TestCase):
    n = 10000
    # initial_pop, y0, birth_rate, death_rate, t_span, get_k, N, kinetic_order_matrix, options
    one_species_configuration = (10, 11, 10, [0, 15.0], get_k_one_species, N_one_species, rate_involvement_one_species)
    two_species_configuration = (10, 11, 10, [0, 15.0], get_k_two_species, N_two_species, rate_involvement_two_species)

    approximate_options = {
        'jit':True,
        'approximate_rtot':True,
        'contrived_no_reaction_rate':100.0,
        'partition_function':hybrid.FixedThresholdPartitioner(1000).partition_function,
    }

    simple_jit = {
        'jit':True,
        'partition_function':hybrid.FixedThresholdPartitioner(1000).partition_function,
    }

    simplest_options = {
        'partition_function':hybrid.FixedThresholdPartitioner(1000).partition_function,
    }

    def test_birth_death_one_species_simplest(self):
        y0 = np.array([self.one_species_configuration[0]])
        self.bd_test(*self.one_species_configuration, y0, self.simplest_options)

    def test_birth_death_one_species_jit(self):
        y0 = np.array([self.one_species_configuration[0]])
        self.bd_test(*self.one_species_configuration, y0, self.simple_jit)

    def test_birth_death_one_species_jit_with_dc(self):
        y0 = np.array([self.one_species_configuration[0]])
        options = {'discontinuities':[4.0]}
        options.update(self.simple_jit.copy())
        self.bd_test(*self.one_species_configuration, y0, options)

    def test_birth_death_one_species_approximate(self):
        y0 = np.array([self.one_species_configuration[0]])
        self.bd_test(*self.one_species_configuration, y0, self.approximate_options)

    def test_birth_death_two_species_simplest(self):
        y0 = np.array([1e9, self.two_species_configuration[0]])
        self.bd_test(*self.two_species_configuration, y0, self.simplest_options)

    def test_birth_death_two_species_jit(self):
        y0 = np.array([1e9, self.two_species_configuration[0]])
        self.bd_test(*self.two_species_configuration, y0, self.simple_jit)

    def test_birth_death_two_species_jit_with_dc(self):
        y0 = np.array([1e9, self.two_species_configuration[0]])
        options = {'discontinuities':[4.0]}
        options.update(self.simple_jit.copy())
        self.bd_test(*self.two_species_configuration, y0, options)

    def test_birth_death_two_species_approximate(self):
        y0 = np.array([1e9, self.two_species_configuration[0]])
        self.bd_test(*self.two_species_configuration, y0, self.approximate_options)

    def tearDown(self):
        with open(f"{self.id().split('.')[2]}.txt", 'w') as f:
            f.write(f"Analytic extinction probability={self.analytic_extinction_p} Extinctions={self.extinctions}/{self.n}. 95% confidence=({self.ci_lower, self.ci_upper})")

    def bd_test(self, initial_pop, birth_rate, death_rate, t_span, get_k, N, kinetic_order_matrix, y0, options):
        rng = np.random.default_rng()

        k = get_k(birth_rate, death_rate)

        extinctions = 0
        simulator = hybrid.HybridSimulator(k, N, kinetic_order_matrix, **options)
        for i in range(self.n):
            result = simulator.simulate(t_span, y0, rng)
            extinctions += result.y[-1] == 0.0
            del result
            if i % 100 == 0:
                print("i", i, "Extinctions:", extinctions, "extinction probability:", extinctions/(i+1))

        print("Observed exctinction probability:", extinctions/(i+1), "+/-", proportion.proportion_confint(extinctions, self.n, alpha=0.05, method='jeffreys'))
        print("Expected extinction probability:", (1/(birth_rate/death_rate))**initial_pop)
        expected_extinction = (1/(birth_rate/death_rate))**initial_pop
        observed_extinction = extinctions/(i+1)
        ci_lower, ci_upper = proportion.proportion_confint(extinctions, self.n, alpha=0.05, method='jeffreys')

        self.extinctions = extinctions
        self.analytic_extinction_p = (1/(birth_rate/death_rate))**initial_pop
        self.ci_lower, self.ci_upper = proportion.proportion_confint(extinctions, self.n, alpha=0.05, method='jeffreys')
        self.assertTrue((expected_extinction > ci_lower) and (expected_extinction < ci_upper)), f"Analytic extinction probability fell outside 95% confidence interval for observation. Analytic probability={expected_extinction}. Observed extinctions={observed_extinction}. Confidence={proportion.proportion_confint(extinctions, self.n, alpha=0.05, method='jeffreys')}"

if __name__ == '__main__':
    unittest.main()