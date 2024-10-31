import unittest
import numpy as np
import hybrid.hybrid as hybrid

class PartitionBreakTests(unittest.TestCase):
    n = 10
    def test_decay_regime_change(self):
        threshold = 100.
        y0 = np.array([1000.])
        k = np.array([1.0])
        # go below the stochastic threshold
        t_span = np.array([0.0, -1 * np.log(threshold * 0.8/y0[0])])
        print(t_span)
        N = np.array([[-1.0]]) # first order exponential decay
        kinetic_order_matrix = np.array([[1.0]])
        rng = np.random.default_rng()

        options = {
            'jit':True,
            'partition_function': hybrid.FixedThresholdPartitioner(threshold).partition_function
        }

        simulator = hybrid.HybridSimulator(k, N, kinetic_order_matrix, **options)

        y_ends = []
        for i in range(self.n):
            result = simulator.simulate(t_span, y0, rng)
            y_ends.append(result.y[0])
        
        print(y_ends)
        print(np.var(y_ends))

        self.assertGreater(np.var(y_ends), 0)

    def test_birth_death_regime_change(self):
        threshold = 100.
        y0 = np.array([1000.])
        k = np.array([1.1, 0.9])
        # go below the stochastic threshold
        t_span = np.array([0.0, -1 * np.log(threshold * 0.7/y0[0]) / (k[0] - k[1])])
        print(t_span)
        N = np.array([[-1.0, 1.0]])
        kinetic_order_matrix = np.array([[1.0, 1.0]])
        rng = np.random.default_rng()

        options = {
            'jit':True,
            'partition_function': hybrid.FixedThresholdPartitioner(threshold).partition_function
        }

        simulator = hybrid.HybridSimulator(k, N, kinetic_order_matrix, **options)

        y_ends = []
        for i in range(self.n):
            result = simulator.simulate(t_span, y0, rng)
            y_ends.append(result.y[0])
        
        print(y_ends)
        print(np.var(y_ends))

        #plt.plot(result.t_history, result.y_history[0,:].T)
        #plt.title("Decline")
        #plt.axhline(y=100, color='r', linestyle='-')
        #plt.show()
        self.assertGreater(np.var(y_ends), 0)

    def test_decay_regime_change_approximate(self):
        threshold = 100.
        y0 = np.array([1000.])
        k = np.array([1.0])
        # go below the stochastic threshold
        t_span = np.array([0.0, -1 * np.log(threshold * 0.8/y0[0])])
        print(t_span)
        N = np.array([[-1.0]]) # first order exponential decay
        kinetic_order_matrix = np.array([[1.0]])
        rng = np.random.default_rng()

        options = {
            'jit':True,
            'partition_function': hybrid.FixedThresholdPartitioner(threshold).partition_function,
            'approximate_rtot': True,
            'contrived_no_reaction_rate': 100.,
            'halt_on_partition_change': False,
        }

        simulator = hybrid.HybridSimulator(k, N, kinetic_order_matrix, **options)

        y_ends = []
        for i in range(self.n):
            result = simulator.simulate(t_span, y0, rng)
            y_ends.append(result.y[0])
        
        print(y_ends)
        print(np.var(y_ends))

        #plt.plot(result.t_history, result.y_history[0,:].T)
        #plt.title("Decline")
        #plt.axhline(y=100, color='r', linestyle='-')
        #plt.show()
        self.assertGreater(np.var(y_ends), 0)